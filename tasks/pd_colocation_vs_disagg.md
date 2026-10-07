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

## Multi-node measurements (H20, 8 pods x 2 GPUs = 16 workers) — 2026-10-07, in progress

Setup: merlin job `c2fb3965f36cd15f` (cn-seed, trial vibecuda), 8 pods x 2 H20, one
Qwen2.5-7B-Instruct TP1 worker per GPU (persistent FA2 kernel in every arm, so arms differ only
in placement and routing). Pods 0/5/6/7 share one physical host. Workload
(`tasks/pd_colocation/mn_bench.sh`, start barrier): 64 TraceLab sessions x 6 turns at
0.5 sessions/s, replaying recorded inter-turn gaps x0.1, where each turn extends the previous
prompt (16-49K tokens); plus random-ids chat at 16 req/s (3 200 requests, 512 -> 128 tokens).
Driver: `SESSIONS=64 TURNS=6 SESSION_RATE=0.5 SLOWDOWN=0.1 CHAT_RATE=16 CHAT_N=3200 python
tasks/pd_colocation/mn_run.py <arms>`; audit: `mn_audit.py` (every pod's worker logs).
"Worker-reported hit" is the sum of sglang's own `meta_info.cached_tokens` over the sum of
`prompt_tokens`, recorded per request by the router.

**RETRACTED pending re-run (2026-10-07):** every arm below ran with the pods' soft
`ulimit -n` of 1024 and the table does not record completed-request counts. Under the chat burst
the stock sglang_router ran out of sockets. Its circuit breaker then rejected most requests:
disagg_2p2d served 163/384 agent and 2 265/3 200 chat requests. The coloc arms used the same
harness, so their completeness is unverified. All arms are being re-run with the limit raised
(commit a8bd07337), with completed counts recorded.

| arm | agent TTFT ms (mean / P99) | agent ITL ms | chat TTFT ms (mean / P99) | chat ITL ms | worker-reported hit | requests/worker |
|---|---|---|---|---|---|---|
| coloc_rr (round-robin) | 13 185 / 60 744 | 24.2 | 9 832 / 58 533 | 27.1 | (not logged; re-run queued) | 224-226 |
| coloc_cache (affinity, load tie-break) | **1 573 / 8 693** | **7.4** | **109 / 1 473** | **6.3** | **72.9%** (router predicted 71.4%) | 117-365 |
| coloc_rand (affinity, RANDOM tie-break) | 1 705 / 11 011 | 9.5 | 487 / 7 931 | 8.7 | 72.8% | 203-250 |

All arms: 0 tracebacks, 0 fallbacks across all 16 workers.

**Reading so far.**
- **Cache affinity is the dominant effect**: agent TTFT 8.4x lower and chat TTFT 90x lower than
  round-robin, with about 73% of all prompt tokens served from cache. Round-robin recomputes every
  session's prefix wherever its next turn lands, and that prefill interferes with decode
  everywhere (ITL 24-27 ms vs 6-7 ms).
- The router's block-hash prefix estimate matches the workers' own count within 1.5 points, so
  it is good enough to route on.
- **Affinity concentrates load** (117-365 requests per worker) -- the hotspot the SLO veto exists
  to bound.
- **coloc_rand result is from a flawed tie-break and is superseded.** Purely random choice
  among equal-hit workers sent prefix-less chat requests (all workers tie at hit 0) to busy
  workers. Load is spread better (203-250), but chat TTFT is 4.5x worse. The router now breaks
  ties by load, then randomly (`router.py`); coloc_rand will be re-run with it.

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

### Measured on the 4-mode MTP server — 2026-10-07 (H200, one GPU)

Setup: Qwen2.5-7B + EAGLE3 draft (3 steps, topk 1, 4 verify tokens), 4-mode step on the FA3/FA2
split (`MTPMIX_SPLIT`), `agent_chat_rated` at chat 24/s + codex 0.1 sessions/s, start barrier.
Driver `tasks/fused_specdec_workload/drivers/mtp_edf.sh 24 2000 50`.
Traced run: `SGLANG_STEP_TRACE=1 SGLANG_EDF_ACCEPT_TRACE=1`, 0 fallbacks, 0 tracebacks.

- [x] **Tokens per decode row per step are 2.10 on average, not 1.** bench_serving (and a user)
      measure ITL per token: a chunk's gap divided by its tokens. A step-time budget of
      `ITL_SLO` (a=1) is therefore about 2.1x too tight under MTP.
- [x] **Online estimators, scored against the tokens each step then emitted** (175 000
      row-steps, mean absolute error in tokens): a=1 **1.10**, global EWMA (alpha 0.2) **1.00**,
      per-request EWMA **0.89**. Per-step acceptance is noisy (1-4 tokens), so no estimator gets
      close to zero, but the per-request EWMA removes the 2.1x bias and is 19% closer than a=1.
- [x] **Step model** (`tasks/perf_model/fit_mtp_step.py`: linear in chunk tokens, chunk
      attention work, verify rows and verify KV; fit on the first half of 11 837 back-to-back
      steps, scored on the second): held-out MAPE **21.9%**, against **98.6%** for a constant
      (mixed steps 14.8%, pure-decode steps 23.4%).
- [x] **EDF attainment at a non-binding ITL SLO (2026-10-07).** A request attains its SLO when its
      TTFT and its P90 per-token ITL both meet target. ITL SLO 50 ms. The TTFT SLO is the one EDF
      schedules against: 2 s + 1 s per 5 000 prompt tokens beyond 4 096
      (`slo_attain.py --scaled`). Under a flat 2 s TTFT SLO, codex looks worse for every arm
      (baseline 0.861). One round per arm; 0 fallbacks and 0 tracebacks in every arm.

      | arm (tokens/step estimate) | chat SLO | chat goodput /s | chat TTFT ms | chat P99 ITL ms | codex SLO | codex TTFT ms | codex P99 ITL ms |
      |---|---|---|---|---|---|---|---|
      | no EDF | **0.876** | **20.62** | **446** | 59.4 | **1.000** | **978** | 49.6 |
      | EDF, a=1 | 0.117 | 2.07 | 40 883 | 56.8 | 0.111 | 55 815 | 69.3 |
      | EDF, global EWMA | 0.866 | 20.37 | 637 | 64.6 | 0.861 | 3 399 | **36.6** |
      | EDF, per-request EWMA | 0.696 | 16.36 | 1 264 | **53.5** | 0.778 | 4 468 | 47.3 |

      Reading:
      - **Without the acceptance term, EDF collapses on an MTP server.** a=1 spends the 50 ms
        per-token budget on one token per step. It refuses almost every prefill chunk, and chat TTFT
        reaches 41 s. The global EWMA restores attainment from 0.117 to 0.866. That is the
        ablation the design asked for, and the gain is large.
      - **EDF does not beat no-EDF at this SLO.** Without EDF, 93-100% of requests already meet
        the 50 ms ITL target, so the controller has almost no violations to prevent. It pays in
        TTFT and buys ITL headroom that the metric does not reward (codex P99 ITL 49.6 -> 36.6 ms).
      - **The lower-error estimator schedules worse.** Per-request EWMA has the lowest MAE (0.84
        vs 0.95 global in its own run), yet attainment is lower. The chunk budget is a MINIMUM
        over ~40 running requests' slack, and a minimum over noisy per-request estimates picks
        whichever request just saw a run of rejected drafts. The global estimate has no such
        order-statistic bias. Being checked with an `EDF_BIND` log (binding request's estimate vs
        global). This is the design section's warning in concrete form: better MAPE that does not
        improve attainment is not a result.
- [x] **Binding ITL SLO (30 ms per token) — the strict per-step controller loses (2026-10-07).**
      `TAG=_itl30 STEP_MODEL=/dev/shm/w/mtpedf/rate24/mtp_step.json mtp_edf.sh 24 2000 30`
      (commit 937792218). 0 fallbacks and 0 tracebacks in every arm. Scored with
      `slo_attain.py --scaled --itl 30`.

      | arm | chat SLO | chat ITL ok | chat TTFT ms | codex SLO | queue mean | running mean |
      |---|---|---|---|---|---|---|
      | no EDF | **0.706** | **0.723** | **433** | **0.972** | 3 | 16.3 |
      | EDF, a=1 | 0.019 | 0.575 | 163 063 | 0.028 | 1 345 | 20.3 |
      | EDF, global EWMA | 0.359 | 0.409 | 1 637 | 0.417 | 23 | 21.3 |
      | EDF, per-request EWMA | 0.016 | 0.478 | 90 187 | 0.028 | 234 | 24.3 |

      - **The order-statistic bias is confirmed.** `EDF_BIND` logs the tokens/step estimate of
        the request that sets the budget. Under the per-request EWMA it averages 1.12-1.19,
        while the global mean is 1.57-1.91. The minimum over ~20 running requests' noisy
        estimates is close to 1, so the arm behaves like a=1 and collapses the same way. Under
        the global EWMA the two are equal by construction (2.04).
      - **Even the global EWMA makes ITL worse than no EDF** (ITL attainment 0.409 vs 0.723),
        with a short queue (mean 23). So the cause is not queueing. The running batch grows
        (21.3 vs 16.3 rows) at about the same token throughput, and mixed steps rise 19%
        (3 800 vs 3 200). The per-step cap spreads the same prefill over more mixed steps. Each
        pays a fixed cost and slows every decode row, while the cap only sees this step's slack
        and ignores that deferred prefill still has to run later.
- [x] **Ablations: deadline ORDER helps; the per-step chunk CAP is what hurts (2026-10-07).**
      Same load, same SLO (ITL 30 ms, scaled TTFT), same command with `ARMS=...`; one round each.

      | arm | priority | chunk cap | chat SLO | chat ITL ok | chat TTFT ms | chat P99 ITL ms | codex SLO | codex P99 ITL ms |
      |---|---|---|---|---|---|---|---|---|
      | no EDF | FCFS | none | 0.706 | 0.723 | 433 | 58.8 | 0.972 | 43.5 |
      | **EDFP** | deadline | none | **0.750** | **0.750** | **178** | **53.6** | **1.000** | **28.9** |
      | EDFGC | FCFS | global EWMA | 0.381 | 0.499 | 2 657 | 58.4 | 0.861 | 43.3 |
      | EDFG | deadline | global EWMA | 0.359 | 0.409 | 1 637 | 56.5 | 0.417 | 52.3 |
      | EDFGQ | deadline | global EWMA, P90 allowance | 0.301 | 0.357 | 2 227 | 57.8 | 0.389 | 53.3 |

      - **Deadline ordering alone wins on every metric**: chat TTFT -59%, codex P99 ITL -34%,
        chat SLO 0.706 -> 0.750, codex 0.972 -> 1.000.
      - [x] **Replicated, 3 rounds** (`rate24_itl30{,_r2,_r3}`; rounds 2-3 used
        `ARMS="MTPMIX_SPLIT MTPMIX_SPLIT_EDFP"`, commit 863fb3212; 0 fallbacks and 0 tracebacks in all
        six runs). Mean [min-max] over the rounds:

        | metric | no EDF | EDFP (deadline order) | change |
        |---|---|---|---|
        | chat SLO attainment | 0.677 [0.658-0.706] | **0.735** [0.714-0.750] | +8.6%, ranges disjoint |
        | chat goodput /s | 15.93 | **17.31** | +8.6% |
        | chat mean TTFT ms | 479 [433-534] | **180** [172-191] | **-62%** |
        | chat P99 ITL ms | 60.3 | **54.1** | -10% |
        | codex P99 ITL ms | 49.2 [43.5-54.8] | **35.8** [28.9-39.7] | **-27%** |
        | codex mean TTFT ms | 1 018 | 1 457 | +43%, still inside its length-scaled SLO |
        | codex SLO attainment | 0.981 | **1.000** | |

        Codex pays in TTFT for chat's TTFT. That is the trade the SLO definition licenses: a
        40K-token prompt is allowed ~9 s.
      - [x] **Full stack vs stock sglang MTP, 3 rounds (2026-10-08).** The stack is the 4-mode step
        on the FA3/FA2 split plus deadline order (EDFP). Stock is `MTP_LPT`: same EAGLE3 draft, no
        prefill/verify mixing, FCFS. Same load (chat 24/s + codex 0.1 sessions/s), barrier. Stock
        rounds: `TAG=_itl30{,_r2,_r3} ARMS=MTP_LPT mtp_edf.sh 24 2000 30` (commit bf96b6cd3).
        0 fallbacks and 0 tracebacks; every request completed in every run. Means:

        | metric | stock MTP | full stack | change |
        |---|---|---|---|
        | chat TTFT ms | 3 684 | **180** | **-95.1%** |
        | chat mean ITL ms | 22.15 | **12.86** | **-41.9%** |
        | chat P99 ITL ms | 83.2 | **54.1** | **-35.0%** |
        | codex TTFT ms | 2 991 | **1 457** | **-51.3%** |
        | codex mean ITL ms | 11.25 | **7.01** | **-37.7%** |
        | codex P99 ITL ms | 54.5 | **35.8** | **-34.3%** |
        | chat SLO attainment (TTFT scaled, ITL 30 ms) | 0.216 | **0.735** | 3.4x |
        | chat goodput /s | 5.08 | **17.31** | 3.4x |
        | output tok/s (chat / codex) | 1 686.7 / 126.1 | 1 691.2 / 125.8 | equal: both serve the offered rate |

        Every TTFT and ITL metric clears the 30% bar on both tenants. Throughput cannot differ at
        a fixed offered rate below both arms' capacity. The throughput claim is the closed-loop
        capacity run (debug_kernel_server_perf_4_mode.md, v0.16.0: +19.8% total). Correctness: the
        4-mode split passed `drivers/srv_correctness.py`. EDFP changes only admission order, not
        the computation.
      - **Any per-step chunk cap loses, with either ordering.** Splitting a chunk into k pieces
        adds k-1 mixed steps. Each costs ~7 ms plus 0.32 ms per running row (`mtp_step.json`) and
        delays every decode row. The deferred prefill still has to run, and under steady load
        there is no idle step to defer it into.
      - **Gating the cap on a P90 slow-step allowance does not rescue it.** More than 10% of steps
        already exceed 30 ms per token, so most requests use up their allowance early. From then
        on the cap applies, and it creates more slow steps.
      - **What this means for point 3.** The predictor does not earn its keep by throttling chunks
        inside a node. The accurate step model and acceptance EWMA matter where a decision is a
        real choice between alternatives, not a delay of work that must happen anyway: the
        router's admission/SLO veto across nodes (point 1). The in-node scheduler should order by
        deadline and stay work-conserving.

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
