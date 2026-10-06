# Global scheduling: PD colocation vs PD disaggregation

Target deployment for every argument below: **4 nodes x 8 H100** (32 GPUs), one
sglang router in front, hierarchical cache enabled with a distributed KV pool.
The comparison is against PD disaggregation on the same hardware, i.e. some
nodes designated prefill and some decode.

This document goes through the four claims in order. Each one is marked with
what is **measured**, what is **derived** from something measured, and what is
**a hypothesis that needs an experiment** — because the value of this story
depends on not blurring those together. A summary of that status is at the end.

Multi-node experiments named here must run on **H20 nodes, at most 4**, trial
named `vibecuda`. H200 is scarce and reserved for single-GPU kernel work.

---

## 0. What the fused kernel actually buys the scheduler

The global-scheduling argument rests on one kernel-level property, so it is
worth stating precisely what is and is not established.

**Measured.** The fused prefill/decode overlap kernel admits a closed-form step
cost (`python/sglang/srt/managers/overlap_cost_model.py`):

```
T_attn(layer) = max(T_p, T_d) + beta * min(T_p, T_d)
T_step        = c0 + c1 * total_tokens + n_layers * T_attn
```

Seven per-GPU physical constants, fit once from 35 kernel microbenchmarks,
**3.1% MAPE** (3.2% leave-one-out) on the H800 sweep. The split-kernel baseline
has no such closed form: its mixed cost is super-additive in its parts (up to
8.1x their sum), and the best parts-based fit is 13.3% MAPE / 43% worst case.
On identical data, features and model class, the baseline is **1.7-3x harder to
predict** (GBDT 10.1% vs 3.3% MAPE, 103% vs 63% worst case), and closing the gap
needs ~1,300 profiled compositions per GPU rather than 35 kernel measurements.

**Not established, and now measured on H200 (2026-10-06).** The *fused FA3+FA2
SM-pre-assignment* kernel — the one this goal is about — is a regression at its
best configuration: at `sm_fa3=64`, 482.19 tok/s and 22.39 ms ITL against a
586.42 / 18.39 baseline (0.82x throughput, 1.22x ITL, 1.32x TTFT). Earlier
reports of 0.335x / 3.00x were an artefact of a pinned split of 8, which the
sweep showed to be far from optimal.

Per-step tracing places the entire gap in MIXED steps (1901 of 2121; p50 21.27 ms
at split 64 against 5.11 ms for DECODE and 16.50 ms for EXTEND, the latter two
unchanged across splits). The cause is structural rather than a defect: a mixed
step must divide SMs between the prefill chunk and the decode rows, while the
baseline runs both classes across all 132 SMs with perfect sharing. So the fused
design can only win if FA3 beats FA2 on FA3's own class by enough to pay for
partitioning the grid, and at kernel level it is currently 1.83x slower there
(772 us against 421 us on a long-prefix fixture). **Everything below depends on
the predictability property, not on the SM-pre-assignment kernel winning.** The
predictability already holds for the shipped overlap kernel. If the
SM-pre-assignment kernel ends up slower, the scheduling story survives; if it
wins, the story gets a throughput term as well. Keeping those separate is what
stops this document from being contingent on an unfinished kernel.

### Update 2026-10-06: two executors, and why that *helps* the predictability argument

The single-kernel design was abandoned on measurement, not preference. Co-residency
constrains it twice, and neither constraint comes from either role's own work:

- **registers** — 384 threads at 1 CTA/SM caps the allocation at 170/thread and the two
  roles' live ranges share it. Upstream FA3 spills 33 676 B inside the fused kernel
  against **52 B** compiled alone in the identical template. Outlining the FA2 body
  takes FA3 to 64 B but moves 17 500 B onto FA2. Halving FA2's O accumulator (8 warps_q,
  `num_mma_q=1`) takes its isolated spill to **88 B** but only to 1 504 B inside the
  fused kernel. So one `__global__` can give *either* role its native allocation, never
  both.
- **shared memory** — FA3's 98 352 B leaves 134 048 B of the 232 448 B opt-in limit, and
  the FA2 role's natural tile-16 configuration needs 135 168 B. Over by 1 120 bytes;
  `cudaFuncSetAttribute` refuses it.

The kernel therefore splits into two executors — flash-attention's FA3 and the
persistent LPT FA2, each at native geometry (measured standalone at 52 B and **0 B**
spill, 168 and **255** registers) — ordered by PDL, reading **one** plan. The
scheduling contribution is unaffected, because all of it lives in the plan rather than
in sharing a `__global__`: 4-mode tile-class assignment, per-queue split length now
sized by each queue's executor geometry, the smem-handoff JIT reduction (intra-role, as
a split row's chunks are all one task's items), and the spec tree-build epilogue.

**Consequence for §3.** The predictability claim gets *stronger*, not weaker. §3 argued
the EDF predictor is more accurate "because the fused kernel is easier to model". Two
executors at native geometry are easier still:

- each kernel's cost is a function of its own queue's work, with no register-starvation
  term that varies with what the other role happens to be compiled alongside;
- the queue assignment and per-queue chunk length are explicit plan outputs, so the
  model's inputs are known before the step runs rather than inferred;
- with PDL the step cost is close to the closed form already fitted at 3.1% MAPE:
  `T_step ≈ c0 + c1 * tokens + n_layers * (T_fa3 + max(0, T_fa2 - overlap))`, which is
  the same `max`-plus-residual shape as `max(T_p, T_d) + beta * min(T_p, T_d)`.

What is **given up** is cross-role backfill inside a CTA — which is itself the construct
that forces both mainloops into one function and causes the 33 676 B — and PDL recovers
part of it temporally. Cross-role multicast is *not* given up: the measured 1.587x DSMEM
result is q-tiles of one prefill row re-gathering the same KV, all FA3-role items.

**Still a regression as of this date**, and the figure is worse than the one above on
the harder fixture: 155.9 us fused against 72.3 us for FA2-on-everything on fixture 2,
**2.16x**, both numerics-correct from one build. The 1.83x cited above is the
16 823-token-KV long-prefix fixture, so the gap is workload-dependent and the shorter
fixture is the harder case. Nothing in §§1-4 depends on this number turning around.

---

## 1. Route to the highest cache-hit node first, and let SLO prediction veto it

### The policy

Disaggregation forces a two-phase placement: find a prefill node, then find a
decode node, then move KV between them. Colocation has no such ordering
constraint, so the *first* policy can be cache affinity rather than role:

```
candidates    = nodes ranked by prefix-cache hit length for this request
for node in candidates (descending hit length):
    if admissible(node, req):  route here
fallback      = argmin over nodes of predicted completion-time violation
```

`admissible(node, req)` is where the perf model enters. It must answer: *if I
put this request on this node, does anything miss its deadline?* — not just this
request, but the requests already queued ahead of it. With EDF that set is
well-defined: the requests whose deadline is **tighter than this one's**.

```
admissible(node, req):
    head = {q in node.queue : deadline(q) < deadline(req)}      # EDF prefix
    # cost of draining the EDF-prefix plus this request's prefill
    t_now = now()
    for q in head ordered by deadline:
        t_now += T_step_predicted(node_state_after(q))
        if t_now > deadline(q): return False       # we would break a tighter req
    t_now += T_prefill_predicted(req, hit_len(node, req))
    return t_now <= deadline(req)
```

Two things make this cheap enough to run per request per candidate node:

- `T_step_predicted` is the closed form above — seven constants, a few flops,
  no checkpoint load, no GPU. The MLP alternative would also work but costs a
  forward pass and a per-workload training set.
- The EDF prefix is usually short. Only requests with tighter deadlines matter,
  and under EDF admission the queue is kept near the feasibility boundary.

### Why this is a real advantage over disaggregation, not a reframing

The honest version of the claim is about **the size of the feasible set**, not
about cache hits per se. Under disaggregation, a request with a long cached
prefix on node A must still be prefilled on a prefill-designated node; if A is a
decode node the hit is unusable without migrating the cache the wrong way. The
set of nodes that can serve the request at full cache benefit is
`{prefill nodes} ∩ {nodes holding the prefix}`. Under colocation it is
`{nodes holding the prefix}`. With 4 nodes and a typical 1:1 P:D split that is
a 2x difference in the candidate set before any SLO filtering.

That is the scheduling-flexibility argument, and it is a *combinatorial* claim
that does not depend on the perf model being accurate. The perf model determines
how much of that larger feasible set you can safely use.

### What to measure

- **Cache hit rate and TTFT** under `cache_aware` affinity-first vs the
  disagg two-phase placement, same trace, 4 x H20.
- **The veto rate**: how often the highest-hit node is rejected by
  `admissible`. If it is near 0 the SLO check is free insurance; if it is near 1
  the policy degenerates to load balancing and the cache argument is weak.
- **Counterfactual regret**: when we veto the top-hit node, how much TTFT did
  the fallback cost? This is the number that says whether the veto is worth it.

Baseline to compare against is upstream `main`'s cache-aware router gate
(`b98a577f`): it switches to minimum `worker.load()` when both an absolute and
a relative imbalance threshold trigger, and uses prefix affinity otherwise.
Note `worker.load()` is *active request count*, not tokens and not predicted
cost. The proposal here is to keep that gate but compare **incremental predicted
server cost** inside the permitted imbalance region.

---

## 2. Best-effort cache migration, and why colocation replicates better

### The mechanism

When we do *not* route to the highest-hit node — because `admissible` vetoed it
— the prefix is still useful: pull it to the node we did choose. sglang already
has the machinery, so this is wiring rather than new infrastructure:

- `python/sglang/srt/mem_cache/hiradix_cache.py` — hierarchical radix cache
  (GPU / CPU / storage tiers)
- `python/sglang/srt/mem_cache/hicache_storage.py` — `HiCacheStorage` backend
  interface, with `mooncake_store`, `nixl`, `lmcache`, `hf3fs`, `eic`,
  `aibrix_kvcache` backends under `mem_cache/storage/`

Best-effort means: issue the pull, serve the request from whatever has arrived
when the prefill is scheduled, and never block on it. A late arrival degrades to
a partial hit, which is strictly better than the miss we would otherwise take.

### Why this replicates better than disaggregation

This is the point where the ThunderAgent argument applies, and it needs to be
stated carefully to be more than an analogy.

ThunderAgent's setting is agentic serving: the same prefix (system prompt, tool
definitions, conversation history) is re-read many times by many requests, and
the thrashing problem is that a cache segment lives on one node while requests
for it arrive at others. Under disaggregation the prefix must live on a
**prefill** node to be useful, so the replica set is confined to the prefill
tier. Every miss outside that tier is a full recompute or a cross-tier pull.

Under colocation every node is a legal home for every prefix, so repeated
access *naturally* replicates: each time a request for prefix P is admitted to a
node that lacks it, best-effort migration seeds a replica there. The replica set
grows toward the set of nodes that actually receive traffic for P, which is the
right shape. Under disaggregation the replica set is capped by the prefill tier
size regardless of demand.

**Hypothesis, needs measurement.** That replicas accumulate where demand is does
not by itself prove lower cost: replication consumes KV capacity that decode
needs, and the eviction policy (`mem_cache/evict_policy.py`) decides whether a
replica survives long enough to be reused. The experiment is a repeated-prefix
agentic trace on 4 x H20 measuring (a) hit rate over time, (b) bytes migrated,
(c) KV capacity displaced, (d) whether the hit-rate gain pays for the eviction
pressure. If replication is net negative under capacity pressure, the honest
conclusion is that the policy needs a replication budget, not that the argument
fails.

---

## 3. A better EDF predictor, with acceptance length as an input

> **MEASURED 2026-10-05, and it contradicts this section's premise for the
> SM-pre-assignment kernel.** Per-step tracing on mixed batches
> (`SGLANG_STEP_TRACE=1`, 2121 steps per arm, fit `t = c0 + c1*tokens + c2*bs`)
> gives, on the MIXED steps that matter:
>
> | arm | MIXED MAPE | p90 |
> |---|---|---|
> | fused FA3+FA2 (`sm_fa3=8`, symmetric) | **13.54%** | 23.39% |
> | unfused baseline | **9.79%** | 18.97% |
>
> The baseline is **1.38x more predictable**, not less. Pure-EXTEND and
> pure-DECODE classes are indistinguishable between arms, as expected since the
> fused kernel only acts on mixed batches.
>
> Mechanism: the fused kernel's step cost contains a makespan term,
> `max(FA3 role, FA2 role)` over a *static* SM partition, which is not linear in
> tokens. The persistent baseline's single shared work queue keeps cost closer to
> linear. The same structure that makes the split sharply tuned (16 -> -22.4%,
> 64 -> -3.6%, 112 -> -22.0%) is what makes the cost non-linear.
>
> **This section therefore stands only on the *overlap* kernel's closed form
> (3.1% MAPE), as section 0 already states.** It must not be re-attributed to the
> SM-pre-assignment kernel. Details in
> `debug_kernel_server_perf_4_mode.md`, commit `613c54016`.


### Why the predictor is better here

> **Caveat added 2026-10-06, from measurement on this kernel.** The 1.7-3x
> predictability advantage quoted above is from the *overlap* kernel's
> microbenchmark sweep. Fitting the SM-pre-assignment kernel's own serving trace
> (2121 steps, six model classes, out-of-fold CV) found fused and baseline at
> **parity** — roughly 1.5% MAPE for either with an adequate class — and the
> earlier "1.38x harder" reading was a model-class plus contaminated-target
> artefact. So for *this* kernel the predictability edge is not yet demonstrated,
> and point 3's argument currently rests on the overlap kernel, where it is.
> Worth resolving before the claim is made about the fused SM-split design.

Already measured, and it is the strongest single result behind this story: the
fused overlap kernel's step cost is closed-form at 3.1% MAPE where the
split-kernel baseline needs a learned model at 9.9-13.3% and 1,300 profiled
compositions per GPU. The downstream consequence is already demonstrated too —
analytic-model EDF pins pooled P99 ITL at **100.24 ms against a 100 ms budget**
(`tasks/realistic_workload_analysis.md`).

Spending an ITL budget precisely requires knowing the marginal cost of a prefill
token *in the presence of* the decode pool. The closed form gives it directly:
below the balance point (`T_p <= T_d`) prefill tokens cost only
`beta * dT_p` — the overlap discount — so the controller can admit chunk tokens
until the budget is exactly spent. With a super-additive baseline cost there is
no such marginal quantity to spend against, which is why chunk control is weaker
there and not merely less accurate.

### Adding acceptance length

For an MTP spec model the step cost depends on how many draft tokens are
accepted, and TPOT depends on it twice: accepted tokens per step sets the
denominator, and the verify batch width sets the attention cost.

The parameter is easy to justify as an online input:

- The draft model is **fixed** in online serving, so acceptance rate is a
  property of the (target, draft, workload) triple rather than a free variable.
- It is directly observable per step — the verifier already knows how many
  tokens it accepted.
- An **EWMA** per request (or per workload class) is therefore a legitimate
  estimator with no offline calibration, and it adapts if a workload shifts.

So the TPOT predictor becomes

```
TPOT = T_step(n_p, r_p, r_d, decode_bs, packed_qo(alpha_hat)) / alpha_hat
alpha_hat = EWMA(accepted tokens per step)
```

where `packed_qo = (k+1) * gqa_ratio` sets which tile class the verify rows land
in. That last coupling is measured and sharp, and it matters for the predictor:
on a GQA model with MTP `k=1`, `packed_qo = 2 * 7 = 14`, so **94.5% of rows are
tile-16 and the tile-64 class is empty** (13 rows in 80,196 over 4,000 steps of
the real MTP trace). The tile class is effectively a step function of `k` and
`gqa_ratio`, which means the predictor should take it as a *discrete feature*,
not interpolate through it:

```
 gqa  k=1  k=2  k=3  k=4  k=8        FA3 gate (packed_qo >= 48)
   7  16   64   64   64   64           gqa=7 -> verify reaches FA3 at k>=6
   8  16   64   64   64   128          gqa=8 -> k>=5
 128  128  128  128  128  128        (MLA: everything is tile-128)
```

**Experiment.** Fit TPOT with and without `alpha_hat`, on the real MTP trace, and
report MAPE for both plus SLO attainment under EDF. The claim to test is not
"acceptance length helps" (it must, mechanically) but *how much of the residual
it removes* — if the closed-form model is already at 3.1%, the headroom is small
and the honest framing is robustness to acceptance drift rather than accuracy.

---

## 4. Network: a complete graph beats a bipartite one

### The structural argument, stated as a bounded constant and not a scaling law

Disaggregation's KV transfers form a **bipartite** graph (prefill nodes to decode
nodes); colocation's form a **complete** graph. The arithmetic of that difference
matters, because it is smaller and flatter than "scales better with #nodes"
suggests:

| n | colocation pairs `n(n-1)/2` | disagg cut `(n/2)^2` | ratio `2(n-1)/n` |
|---|---|---|---|
| 2 | 1 | 1 | **1.00** -- degenerate, bipartite *is* complete |
| 4 | 6 | 4 | 1.50 |
| 6 | 15 | 9 | 1.67 |
| 8 | 28 | 16 | 1.75 |
| 16 | 120 | 64 | 1.88 |
| inf | -- | -- | 2.00 |

The ratio **asymptotes at 2x and is 88% of the way there by n = 8**. The same
holds for viable sources per prefix: disagg caps at `|prefill tier|` = n/2,
colocation at n, so again 2x. So this is a *bounded constant-factor* advantage,
not growth in n, and the step from n = 4 to n = 6 is 1.50 -> 1.67 -- about 11%,
comfortably inside serving-measurement noise.

An earlier draft of this document claimed the approach "scales better as #nodes
increases". That was wrong, and no node count available to us could have shown
it, because there is no unbounded trend to show.

**What is actually measurable, and is the real claim: hotspot formation.** As
offered load rises, disagg's prefill tier saturates while colocation spreads the
same bytes over more pairs. The quantity to report is therefore the *distribution*
of per-link load and its tail, not a slope in n:

- p99 / max per-link utilization at matched offered load
- whether disagg's busiest link saturates before colocation's does
- two-way usage: fraction of bytes on the reverse direction of each link, which
  is structurally 0 for disagg and nonzero for colocation

Those separate the two topologies at any n >= 4 and get sharper at 8.

### Choosing among multiple hit nodes

When several nodes hold a usable prefix and their hit lengths are within a
threshold, picking the longest is pointless precision and concentrates load.
The proposal is the cheap thing:

```
sources = {node : hit_len(node) >= (1 - eps) * max_hit_len}
pull from uniform_random(sources)
```

Randomization is the right default here for a reason worth being explicit about:
it needs **no state, no telemetry, and no router round-trip**, and it provably
spreads load across the candidate set. This is the same reason
power-of-two-choices and ECMP are the standard answers in networking — ECMP
hashes flows across equal-cost paths without any congestion feedback at all.

### Should we model congestion instead?

The user's own caveat is the correct one, and it should be recorded as the
reason *not* to start here:

- **Stale information on rarely-used links.** Congestion-aware routing needs a
  recent estimate per link. Links we rarely pull from have no recent samples,
  which is exactly the classic problem with load-sensitive routing — ARPANET's
  delay-based routing oscillated for this reason, and the fix (Khanna & Zinky's
  revised metric) was to *damp* responsiveness rather than sharpen it.
- **Query latency.** Asking the central router for link state puts an RPC on the
  critical path of a cache pull, which is the thing we are trying to make fast.

A defensible middle option, if randomization proves insufficient:
**power-of-two-choices on locally-observed latency.** Sample two nodes from
`sources`, pull from whichever has the lower EWMA of *our own* recent transfer
latency to it. No router query, no global state, locally maintained, and the
d-choices result says two samples capture most of the benefit of full
information.

**Hypothesis, needs measurement.** Everything in this section is structural
reasoning, not data. On 4 nodes the bisection argument is weak — `n(n-1)/2 = 6`
pairs vs a `2x2` cut — so a 4-node experiment can measure *mechanism* (link
utilization spread, transfer latency distribution, two-way usage) but cannot
demonstrate the scaling claim. The scaling claim should either be stated as
analysis with a simulator, or dropped from the paper's empirical claims and kept
as motivation. Overstating a 4-node result as evidence of `n`-scaling is the
main risk in this section.

---

## Status summary

| Claim | Status | What would settle it |
|---|---|---|
| Fused overlap kernel is closed-form predictable; baseline is 1.7-3x harder | **Measured** (3.1% vs 9.9-13.3% MAPE) | done |
| Analytic-model EDF holds P99 ITL at budget | **Measured** (100.24 ms vs 100 ms) | done |
| MTP collapses the tile taxonomy, so tile class is a discrete predictor feature | **Measured** (94.5% tile-16, tile-64 empty) | done |
| Colocation's feasible candidate set is ~2x disagg's | **Derived** (combinatorial, no data needed) | state as analysis |
| Affinity-first + SLO veto beats disagg placement on TTFT | **Hypothesis** | 4 x H20, veto rate + counterfactual regret |
| Best-effort migration makes replication net-positive | **Hypothesis** | 4 x H20 repeated-prefix trace, hit rate vs KV displaced |
| Acceptance-length input improves TPOT prediction | **Hypothesis** (mechanically must; size unknown) | fit with/without `alpha_hat` on the MTP trace |
| Fused kernel is easier to perf-model than unfused (section 3's premise) | **MEASURED FALSE** for the SM-pre-assignment kernel -- mixed-step MAPE 13.54% vs baseline 9.79% | done; section 3 now rests on the overlap kernel only |
| Randomized source selection suffices vs congestion modelling | **Hypothesis** | 4 x H20, link utilization spread |
| Complete-vs-bipartite is a bounded ~2x constant, 88% saturated by n=8 | **Derived** (arithmetic; corrects an earlier "scales with n" claim) | state as analysis |
| Disagg forms link hotspots that colocation does not, at matched load | **Hypothesis** | 8 x 2 H20 with verified distinct hosts; p99/max per-link utilization |

### Experiment order

1. `alpha_hat` in the TPOT predictor — single node, no new infrastructure,
   directly extends a measured result.
2. Affinity-first + SLO veto, 4 x H20 — needs the veto-rate and regret
   instrumentation, which does not exist yet.
3. Best-effort migration on a repeated-prefix trace, 4 x H20 — reuses (2)'s
   harness.
4. Source selection and link utilization, 4 x H20 — reuses (3)'s traffic.

Items 2-4 share one harness, so build it once. Item 1 is independent and should
go first because it is cheapest and least speculative.
