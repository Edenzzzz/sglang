# Current goal

Make the case for **PD colocation with an EDF/SLO-aware router** against **PD disaggregation**,
for a 4x8 H100/H200 cluster serving agentic traffic (TraceLab-shaped: 64-100K contexts, ~3.8K
appends, ~480-token outputs, mixed with chat and math).

Tasks, in priority order:

- [ ] **1. Cache-hit-first routing with an SLO veto** — router BUILT (2026-10-07,
      `tasks/pd_colocation/router.py`; policies rr / cache / cache_rand / cache_slo, optional EDF
      priorities). Fake-worker check (`router_test.py`): cache sends all 24 requests to the holder
      (the hotspot); cache_slo keeps the holder while predicted TTFT stays under budget
      (220 -> 295 ms), vetoes at 300 ms, and the request goes to another worker, which then holds the
      prefix too. That is a second replica, created by the veto without any migration step (point
      2). Needs the multi-node run.
- [ ] **2. Best-effort cache migration and why colocation replicates better** — design below;
      the replication claim needs measurement.
- [ ] **3. EDF predictor using EWMA acceptance length to predict TPOT** — perf model fitted
      (2026-10-07, *measured*): mixed-step MLP MAPE 6.0% (split) vs 6.4% (LPT), linear 13.4% vs
      16.9%, so a cheap predictor suffices. See debug_kernel_server_perf_4_mode.md#perf-model-fit--2026-10-07.
      Still needed: the acceptance-length input (MTP traces) and the EDF-vs-baseline scheduling run.
- [ ] **4. Two-way RDMA, complete vs bipartite graph, randomized source selection** — design
      below; needs a multi-node congestion measurement (H20 only, <=16 GPUs, trial `vibecuda`).

**Nothing in this document is measured yet.** It is the design and the argument, written so the
experiments have something to confirm or refute. Every quantitative claim is labelled either
*measured* (with a pointer) or *predicted*. Treat the predictions as hypotheses.

---

## What the kernel work licenses us to assume — 2026-10-06

The scheduling argument below leans on one property: **attention cost on a colocated batch is
predictable from the batch's shape**. Two measured results support that, and one undermines it.

- *Measured.* After the `seqused` fix, FA3's cost is **linear in KV** at fixed q:
  `us/kv` constant to 4.5% across a 9.8x range (10 240 -> 100 352). Before the fix it was
  `Theta(kv^2)`. A linear, low-variance cost is what makes an MLP-style predictor viable at
  all. See [the root cause entry](debug_kernel_server_perf_4_mode.md#fa3-root-cause-seqused).
- *Measured.* The two-executor split beats FA2-on-everything on a TraceLab-shaped batch by
  **1.225x (18.4%)**, 12 399.7 us against 15 190.9 us. Kernel-batch time, **not** serving.
- *Against.* A row whose KV exceeds the ~1024 chunk limit **corrupts later rows** in the
  current plan. Until that is fixed, no serving measurement is trustworthy, including anything
  below. This is the gating item.

---

## 1. Route to the highest cache-hit node first, with an EDF/SLO veto

**The disagg baseline.** PD disaggregation picks a *prefill* node first, because prefill and
decode live on different pools. Prefix cache locality is then a second-order concern: the
request goes where prefill capacity is, and whatever prefix that node happens to hold is luck.
With agentic traffic this is the wrong order of preference, because the prefix is enormous
(64-100K) relative to the new tokens (~3.8K), so a hit is worth far more than a marginally
emptier queue.

**The colocated policy.** Rank candidate nodes by prefix-hit length, then **veto** by SLO
feasibility rather than by load:

    candidates = sort(nodes, key = -hit_len(node, request))
    for n in candidates:
        if feasible(n, request):  route(request, n); return
    route(request, argmin_over_nodes(predicted_slo_slack))   # least-bad

`feasible(n, r)` must account for more than the node's current occupancy. Under EDF, admitting
`r` can only delay requests whose deadline is **later** than `r`'s; the ones that can be hurt
by `r` are those with *tighter* deadlines already queued ahead of it. So:

    feasible(n, r):
        head = {q in queue(n) : deadline(q) < deadline(r)}
        # r is served after `head` under EDF, so r's own completion is pushed by all of head
        t_r = now + sum(predicted_service_time(q) for q in head) + predicted_service_time(r, n)
        if t_r > deadline(r): return False
        # and admitting r must not push any LATER-deadline request past its own deadline
        for q in queue(n) with deadline(q) > deadline(r):
            if predicted_completion(q | r admitted) > deadline(q): return False
        return True

**Why this is a structural advantage, not a tuning difference.** In disagg the prefill pool and
decode pool are scheduled separately, so an SLO decision at admission cannot trade prefill
delay against decode delay -- they are different machines with different queues. Colocation
puts both on one node, so the router is choosing within a single queue discipline and can
express "take the cache hit, pay a little TTFT, still make the deadline". That trade is simply
not available to a disagg router.

**Risk to measure, not assume.** Cache-hit-first concentrates load on whichever nodes hold
popular prefixes. The SLO veto is the only thing preventing hotspotting, so the experiment must
report the **hit-rate vs p99-TTFT frontier**, not just mean TTFT. If the veto fires often, the
policy degenerates toward load-balancing and the advantage shrinks -- that is the honest
failure mode and it should be looked for.

## 2. Best-effort cache migration, and why colocation replicates better

When the veto rejects the best-hit node, we route to a worse-hit node **and** pull the missing
prefix to it, using sglang's hierarchical cache plus the distributed cache pool. The pull is
best-effort: if it does not complete before the request is scheduled, the node recomputes the
tail, so correctness never depends on the transfer and a slow link degrades latency rather
than breaking the request.

**The replication argument.** Every migration leaves a copy. Under cache-hit-first routing with
an SLO veto, popular prefixes are therefore *pulled toward wherever load forces requests to
go*, so replica count grows with demand for that prefix. This is the behaviour ThunderAgent
argues for in agentic serving, where the same long session prefix is revisited repeatedly and a
single-copy cache thrashes: the copy is evicted between turns and re-prefilled from scratch,
over and over.

Disagg has a weaker version of this. Its prefill pool holds the prefixes, and decode nodes do
not accumulate them, so replication is bounded by the prefill pool's capacity and the pool is
the thing that must be scaled to reduce thrashing. Colocation makes every node a potential
replica host, so the effective cache is the whole cluster's memory rather than one pool's.

**What must be measured.** Replica count per popular prefix over time, re-prefill rate (tokens
recomputed because of a miss), and whether migration traffic displaces useful work. The claim
"colocation replicates better" is a prediction; the number that would falsify it is a
re-prefill rate no better than disagg's at equal memory.

## 3. A more accurate EDF predictor, with acceptance length as an input

EDF only works if `predicted_service_time` is accurate; a scheduler with a bad predictor
misses deadlines it could have met and rejects requests it could have taken.

**Why colocation plus one attention kernel helps.** A colocated step is one batch through one
attention implementation whose cost is now **linear in total KV** (*measured*, above). Features
that are known at admission:

    TTFT  ~ f(sum of append lengths, sum of context lengths, #prefill rows, #decode rows)
    TPOT  ~ g(#decode rows, sum of context lengths, predicted acceptance length)

A small MLP (two hidden layers, tens of units) over those features is sufficient; the point is
not model capacity but that the target is a smooth low-variance function. In disagg the same
prediction spans two machines, two queues and a KV transfer, so its variance is dominated by
cross-node effects the router does not observe.

**Acceptance length as a feature, and why it is fair.** With an MTP draft model the step cost
depends on how many drafted tokens are accepted. Online, the draft model is **fixed**, so
acceptance length is a slowly-varying property of the traffic, and an EWMA over recent accepts
estimates it without any oracle:

    a_hat <- (1 - alpha) * a_hat + alpha * a_observed        # per model, optionally per workload class
    TPOT_hat = g(..., a_hat)

This is legitimate because nothing about the *future* is assumed -- only that the recent past
predicts the next step, which is exactly the assumption every production scheduler already
makes about arrival rates. It should be reported with the EWMA's own error bars, and the
ablation that matters is `g` with and without `a_hat`.

**Validation plan.** Fit on one traffic mix, test on another (held-out ratio of chat to
TraceLab). Report predictor MAPE, then the thing that actually matters: SLO attainment under
EDF with this predictor against EDF with a naive linear model and against the unfused baseline
schedule. A better MAPE that does not improve attainment is not a result.

## 4. Two-way RDMA: a complete graph instead of a bipartite one

**The topology argument.** Disagg's KV traffic is structurally **bipartite**: prefill nodes
produce, decode nodes consume, and the cut between the two pools carries all of it. Colocation
makes every node both producer and consumer, so the transfer graph is **complete** -- any node
can serve any other. For the same total bytes, a complete graph spreads them over
`O(N^2)` links instead of `O(N_p * N_d)` across one cut, and it scales with `N` instead of with
the smaller pool.

This also removes the "few nodes hold the hot segment" failure mode: in disagg, a popular
prefix lives in the prefill pool and every decode node needing it pulls across the same cut.

**Congestion control, and a deliberate preference for the dumb algorithm.** When several nodes
hold hits within a threshold of each other, pick one **uniformly at random**:

    hits = [n for n in nodes if hit_len(n) >= (1 - eps) * max_hit_len]
    source = uniform_random(hits)

Randomisation needs no state, adds no latency, and spreads load across sources -- the standard
power-of-choices argument from load-balancing. Compared with modelling link congestion, it is
far more robust for this workload:

- a congestion model needs per-link state for links we **rarely use**, and those estimates are
  the stalest exactly when we suddenly need them;
- querying a central router adds a round trip on the critical path of a cache pull, which is
  latency we are trying to save;
- classic congestion-aware routing (ECMP's hashing, CONGA's leaf-to-leaf load sensing,
  HULA/Hedera-style adaptive schemes) works because switches observe link utilisation
  continuously and locally; our router observes neither continuously nor locally.

So the plan is: **random-among-ties first**, and treat congestion-aware source selection as an
ablation that must *beat* random to justify its state and its round trip. "Random is within
noise of the modelled policy" is a perfectly good outcome and should be reported as such.

**What must be measured.** Per-link utilisation spread (max/mean), p99 cache-pull latency, and
whether aggregate attainment degrades more slowly with `N` than disagg's -- the scaling claim
is the real one, and it needs at least 4 nodes to say anything, hence the H20 budget (<=16 GPUs,
4x4 or 8x2, trial `vibecuda`).

---

## Honest summary of standing

The kernel-level result that motivates all of this (**1.225x on a TraceLab-shaped batch**) is
real but is batch time, not serving. The publishability bar is **>=30% TTFT/ITL and >=10%
throughput on a realistic workload**, and there is **no serving data at all** yet. The gating
blocker is the split-row corruption in point 0 above, because serving batches routinely contain
rows above the chunk limit, so a serving A/B today would measure a corrupted baseline.

Order of work: fix the corruption, run the serving A/B, then the four experiments above in the
listed order.
