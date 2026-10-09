# Current goal

Make the case for **PD colocation with an EDF/SLO-aware router** against **PD disaggregation**,
for a 4x8 H100/H200 cluster serving agentic traffic (TraceLab-shaped: 64-100K contexts, ~3.8K
appends, ~480-token outputs, mixed with chat and math).

Tasks, in priority order (status 2026-10-09; evidence in the sections below).

**The thesis in one line:** our policy space (4-mode colocation + EDF/SLO router + tunable P/D
admission ratio) is a **strict superset** of every baseline's fixed policy, so we should match
the best baseline at any static workload and beat all of them when the workload moves.
Baselines are two-tier: **single node** = prefill-prioritized (the default) and mixed chunked
prefill; **multi node** = PD multiplex and PD disagg (1P3D, 2P2D).

**Item 5 carries that claim and is the weakest-evidenced part of the thesis:** nothing measured
so far varies the workload mixture, shifts it mid-run, or compares against any baseline on
goodput. Items 1-3 measure routing, cache replication and EDF ordering against 2P2D at a single
static mixture.

- [x] **1. Cache-hit-first routing with an SLO veto** (2026-10-08, H20 16 GPUs) — cache-affinity
      colocation vs the best disagg (2P2D): agent TTFT -93% (21.6 -> 1.6 s), chat TTFT -99.5%,
      prefix hit 73% vs 23-31%. Disagg keeps lower ITL (3.7 vs 6.4 ms). The calibrated veto fires on 2% of
      requests and trims chat P99 TTFT 15%.
      Hot-prefix stress (4 shared 16K prompts piled on their holders): a TTFT-only veto barely
      fires, because a cache holder's PREFILL stays fast while its DECODE degrades. Pricing the
      decode slowdown (`--decode-grid`) cuts chat ITL **-65%** and P99 **-80%** for a 25% chat TTFT
      cost (v0.18.0). Sub-items of this point:
      - [x] route to the highest-hit node first (affinity beats rr and disagg by 8-99x on TTFT)
      - [x] veto on predicted SLO violation, counting only EARLIER-deadline in-flight requests
            (`router.py feasible()`: it sums the prefill ahead of this request's deadline, and
            re-checks that admitting it does not push later-deadline requests past their own)
      - [x] the veto must price the term the overload appears in (TPOT, not just TTFT)
      - [x] load balancing: tie-break on hit bucket, then in-flight load, then random (a bare
            `min()` created a 2-worker hotspot; retracted and fixed, daaedef6a)
- [x] **2. Best-effort cache migration / replication** (2026-10-08) — a cluster-wide mooncake L3
      store (write-through, best-effort prefetch) lifts round-robin's hit rate 12.7% -> 22.3% (chat
      TTFT -41%, P99 ITL -83%). Under affinity it sits idle (16 pulls): affinity already serves 73%
      locally, so migration pays only where routing must place a request elsewhere.
- [x] **3. EDF with an EWMA acceptance length** (2026-10-08, H200) — measured 2.10 tokens/step. a=1
      collapses EDF; a per-request EWMA is biased low by the min over requests; a global EWMA is
      right, but every per-step chunk cap loses. **Deadline ORDER wins**: chat TTFT -62%, codex
      P99 ITL -27%. Full stack vs stock MTP: TTFT -95%/-51%, ITL -42%/-38% (chat/codex, 24/s, 3 rounds).
- [ ] **5. THE SUPERSET CLAIM: adapt to any workload the baselines are each tuned for.**
      Added 2026-10-09 on the user's correction; **this is the capacity/throughput axis and
      nothing measured so far tests it.**

      **The claim.** Our method is 4-mode colocation plus an EDF/SLO router with a tunable
      **P/D admission ratio**. Each baseline is one fixed point in that policy space, so our
      policy space is a strict superset of all of them, and the EDF predictor plus per-class
      SLOs choose the operating point *per workload and per step* instead of at deploy time.

      | baseline | scope | the fixed choice it makes | our method reduces to it by |
      |---|---|---|---|
      | **prefill-prioritized** (sglang DEFAULT) | single node | prefill always preempts decode | admission ratio -> prefill absolute priority |
      | **mixed chunked prefill** | single node | always co-admit a prefill chunk with decode | admitting a chunk every step |
      | **PD multiplex** (`arxiv 2504.14489v2`) | multi node | time-share P and D on every node | alternating admission in time |
      | **PD disagg 1P3D** | multi node | 1 node pure-prefill, 3 pure-decode | per-node ratio -> {1.0} u {0.0 x3} |
      | **PD disagg 2P2D** | multi node | 2 pure-prefill, 2 pure-decode | per-node ratio -> {1.0 x2} u {0.0 x2} |

      **Why this is the right framing:** each baseline is optimal for *some* mixture and wrong
      for others — prefill-priority wins when prefill is scarce and starves decode when it is
      not; 2P2D is right at its provisioned ratio and strands half the cluster when traffic
      turns decode-heavy. A fixed policy cannot follow the workload; a tunable ratio can.

      **How the claim is falsified, stated up front so the experiment can fail honestly:**
      the superset argument predicts (a) at *every* static mixture we are within noise of the
      **best** baseline for that mixture, and (b) at mixtures that **shift mid-run** we are
      strictly better than all of them. If there is a static mixture where we lose to a
      baseline by more than run-to-run noise, the superset claim fails there — a fixed policy
      has less bookkeeping and that overhead is real, as the 6.8% kernel regression in
      `debug_kernel_server_perf_4_mode.md` already shows for a different mechanism.

      - [ ] **5a. SINGLE-NODE baselines** — vs **prefill-prioritized** (the default) and
            **mixed chunked prefill**. 1 GPU; runnable on the current H200 pod.
      - [ ] **5b. MULTI-NODE baselines** — vs **PD multiplex** and **PD disagg (1P3D and 2P2D)**
            at equal total GPU count. Per the standing authorization: **H20**, trial named
            **vibecuda**, <= 16 GPUs (4 nodes x 4, or <= 8 nodes x 2).
      - [ ] **5c. Metric is GOODPUT under per-class SLOs, not raw tok/s.** A TTFT target for
            interactive chat and a TPOT/ITL target for agentic streams; a request counts only if
            it meets its OWN class's SLO. Raw throughput is what made the colocated-vs-colocated
            kernel A/B look flat and is the wrong number for a capacity claim.
      - [ ] **5d. STATIC sweep: mixture ratio x request rate.** Prefill-heavy -> decode-heavy at
            several rates; report the **capacity frontier** (max sustainable rate per mixture),
            not a single operating point. This is the test of prediction (a): the best baseline
            changes across the sweep, and we must track it everywhere. Mixtures blended from the
            datasets already in use (TraceLab agentic replay + chat + math), with the blend
            recorded so it is reproducible.
      - [ ] **5e. DYNAMIC: mid-run mixture shift and load spike.** The core of the argument and
            the test of prediction (b): step-change the mixture (prefill share down, decode share
            up) and inject a rate spike; measure SLO attainment **through** the transition and
            time-to-recover, with the transition visible in the time series rather than averaged
            away. A fixed split cannot re-provision; we re-balance per step. This is also where a
            null result would most damage the thesis, so it gets run most carefully.
      - [ ] **5f. Read `arxiv 2504.08784`** for its SLO definitions, goodput metric and how it
            constructs and shifts mixtures. The arxiv PDF does not text-extract (compressed
            streams only); use the ar5iv/HTML rendering.

- [ ] **4. Two-way RDMA / congestion, randomized source selection** — **not demonstrated on the
      full 4-NIC fabric, and reported as such.** No congestion exists to relieve at 16 GPUs on 4
      hosts: random tie-break among near-equal hits is neutral, the L3 store moved ~236 GB without
      measurable contention, and disagg's 8 prefill GPUs replicate 4 hot prefixes to 82% hit. The
      store also COSTS ~19% chat ITL once the veto spreads load. Sub-items:
      - [x] randomized pull among near-equal-hit nodes (`cache_rand`): neutral at 16 GPUs
      - [x] the same on a NARROWED fabric (`ONE_NIC=1`) -- **randomized selection LOSES even under
            deliberate contention.** Measured below; this is a negative result for the proposed
            algorithm, in the regime built to favour it.
      - [ ] congestion modelled from router-observed past link latency (the alternative in the
            goal). NOT built, and the measurements argue against building it here: with no
            contention to observe there is nothing to fit, and a per-request router query would add
            latency to every decision. Revisit only if the narrowed-fabric run shows an effect.
      - [ ] scaling with node count (the actual claim) -- needs more nodes than the 16-GPU cap
      - [ ] **Simulated-node topology (user proposal, 2026-10-08): treat each GPU as a node and
            shape its interconnect.** 2x8 or 1x8 H200, one TP1 worker per GPU = 8 independent cache
            holders instead of 16 workers paired on 4 host NICs. More holders means hit lengths tie
            more often, which is the only case the random-pull policy differs at all.
            *User authorized H200 for this despite the standing "H20 only / H200 is scarce" note in
            the goal; record it as an explicit exception.*
            **Caveat that decides the implementation: `tc qdisc ... netem` cannot shape RoCE.**
            RDMA bypasses the kernel network stack, so delay/rate rules on the interface do not
            touch mooncake's RDMA transport. Two workable knobs instead:
            (a) run the store over TCP (`MOONCAKE_PROTOCOL=tcp`), which puts transfers back in the
                kernel path where `tc qdisc add dev <if> root netem delay <d> rate <r>` applies
                per node -- this is the way to emulate cross-node RDMA latency AND bandwidth;
            (b) keep RDMA and vary the NIC count as a coarse bandwidth divisor (`ONE_NIC=1`,
                measured: tails 2.3-3.2x worse at unchanged means), which is real contention but
                only one point on the curve.
            Prefer (a) for a latency/bandwidth sweep, and cross-check one point against (b) so the
            TCP emulation is anchored to a real RDMA measurement.
- [x] **MTP on every role** (2026-10-08) — colocation + MTP + 4-mode keeps TTFT and cuts agent ITL
      47%, and vs disagg + MTP cuts TTFT 94% / 99.6%. (An apparent regression was a router
      tie-break hotspot, retracted and fixed.) The fixed-router rerun confirms it: agent ITL -46%
      (7.2 -> 3.9 ms) at unchanged TTFT, and the 4-mode split edges stock MTP on every metric.
      - [x] the H20 serving gate now forces real mixed steps (2026-10-08). Cause of the weak pass
            was the GATE, not the kernel: background rows asked for 96 output tokens and finished
            decoding before the long prefills arrived, so nothing mixed. Fixed (1 200-token
            background rows with `ignore_eos`, staggered long prompts): **10 mixed steps, one with
            2 prefill + 11 verify rows**, vs 1 before. Both comparisons PASS, 0 fallbacks.
      - [x] **The weak pass margin is bf16 noise, not a fault -- RESOLVED 2026-10-08, MTP numbers
            stand.** Split-vs-stock min common prefix is 5 (short) / 15 (long) against a
            stock-vs-stock floor of 285 / 101, identically on H200 and H20, so it is systematic.
            `SGLANG_SPLIT_VERIFY=1` (in-server fp32 recompute of every row of every split step from
            the paged cache, `drivers/verify_gate.sh`) gives the answer:

            | rows | side | qo | kv | rel err vs fp32 |
            |---|---|---|---|---|
            | 0-3 | FA3 (prefill chunks) | 2 048 | 2 048 | 0.0032-0.0040 |
            | 4 | FA2 (**verify rows**) | 4 | 2 052 | 0.0030-0.0034 |

            0.3-0.4% is bf16 accumulation noise, and `|got|` tracks `|ref|` (2.8906 vs 2.8921). Row
            4 at `qo=4` is a VERIFY row beside 2 048-token prefill chunks, so a real 4-mode step was
            verified on both sides. A 0.3% logit difference flips a greedy argmax whenever the top
            two are close, after which sequences separate completely -- which is why the floor is
            high (identical kernels -> identical numerics) and why min-common-prefix is a WEAK
            discriminator across kernel paths. The fp32 per-row check, not prefix length, is the
            real gate; prefix length only catches gross faults like the voided per-request-array bug
            (which showed rel err ~4.0, three orders of magnitude worse).
            *Caveat:* this run logged 1 mixed step (4 verified split steps), because the fp32
            recompute slows the server enough to shift the gate's overlap timing. The step count
            came from the separate strengthened-gate run (10 mixed steps); use that for coverage and
            this for numerics.

**Labels.** Every quantitative claim below is either *measured* (with the run and commit) or
*predicted*. The design sections that predate the measurements say so where the measurements
contradicted them.

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

## Multi-node measurements (H20, 8 pods x 2 GPUs = 16 workers) — 2026-10-08

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

**Re-run with the fd limit fixed (2026-10-08).** The first table (2026-10-07) ran with the pods' soft
`ulimit -n` of 1024. Under the chat burst the stock router ran out of sockets, its circuit breaker
opened, and it rejected most requests: disagg_2p2d served 163/384 agent and 2 265/3 200 chat. That
table is withdrawn. The re-run raises the limit to the hard limit (`ulimit -n $(ulimit -Hn)`, verified
1 024 768 in the router process) and gives disaggregation its best stock configuration: circuit
breaker off, health ejection off, `--prefill-policy cache_aware --decode-policy power_of_two`
(commits a8bd07337, b6ac0c2ad). Command:
`python mn_run.py disagg_2p2d disagg_1p3d coloc_cache coloc_rr coloc_rand` with the knobs above.
Audit: `mn_audit.py <arms>`. 0 tracebacks and 0 fallbacks on all 16 workers in every arm.

| arm | completed agent / chat | agent TTFT ms mean / P99 | agent ITL ms | chat TTFT ms mean / P99 | chat ITL ms | chat out tok/s | prefix hit | requests/worker |
|---|---|---|---|---|---|---|---|---|
| disagg_2p2d (8 prefill + 8 decode GPUs) | 384 / 3 173 | 21 581 / 77 164 | 5.8 | 20 147 / 77 131 | **3.7** | 152.5 | 23.3% | 19-885 |
| disagg_1p3d (4 prefill + 12 decode GPUs) | 384 / 3 176 | 49 257 / 97 041 | **5.2** | 50 301 / 94 592 | **3.5** | 152.6 | 30.8% | 19-936 |
| coloc_rr (round-robin) | 384 / 3 200 | 11 702 / 61 465 | 22.4 | 8 783 / 68 555 | 25.5 | 1 338.2 | 12.7% | 225-226 |
| **coloc_cache** (affinity, load tie-break) | 384 / 3 200 | **1 596 / 8 371** | 7.2 | **101 / 1 539** | 6.4 | **1 481.6** | **72.9%** | 88-329 |
| coloc_rand (affinity, load then random tie-break) | 384 / 3 200 | **1 530 / 8 190** | 7.5 | 121 / 2 613 | 6.3 | 1 481.1 | 72.8% | 111-307 |

Prefix hit: for coloc, the worker-reported `cached_tokens / prompt_tokens` summed over requests;
for disagg, the prefill workers' own logged cache hit. The disagg arms' failed chat requests
(27 and 24) were all streams cut off mid-response, consistent with aborted KV transfers.

**Reading.**
- **Against the best disaggregated layout, cache-affinity colocation cuts mean TTFT 93% for the
  agent (21.6 s -> 1.6 s) and 99.5% for chat (20.1 s -> 0.10 s).** It serves the full offered chat
  rate (1 482 vs 153 output tok/s over the run) and completes every request.
- **Disagg is prefill-bound here, and its cache-aware prefill routing finds little cache.** It hits
  23-31% of prompt tokens against colocation's 73%. With 4-8 prefill GPUs serving all 64 sessions'
  appends, each turn's prefix is on a prefill node only if the last turn ran there and survived
  eviction. Colocation can place a turn on any of 16 GPUs: point 1's flexibility, measured.
- **Disagg keeps the lower ITL** (3.5-5.8 ms vs 6.3-7.5 ms). Its decode GPUs never run prefill.
  This is the price of colocation on this kernel stack, and it is what the 4-mode kernel and
  deadline ordering attack on a single node (point 3: -38% to -42% ITL vs stock MTP).
- **Routing is most of colocation's win.** Round-robin colocation is only 1.8-2.3x better than
  disagg on TTFT and 4-7x worse on ITL. With 12.7% hit, every node recomputes session prefixes, and
  that prefill stalls decode everywhere.
- **Random choice among near-equal hits costs nothing at this scale** (agent TTFT -4%, chat TTFT
  +20%, ITL equal). It does not help yet either: 16 workers on 4 hosts do not congest links (point 4).
  The earlier purely random tie-break, which sent hit-0 chat to busy workers, is fixed by breaking
  ties on in-flight load first.

**Shared L3 prefix store: cross-node pulls on a complete graph (points 2 and 4), 2026-10-08.**
`<arm>_hc` arms add a cluster-wide mooncake store. One `mooncake_master` runs on pod 0. Every
worker contributes 32 GB of host DRAM over RDMA on the back-end NICs and runs
`--enable-hierarchical-cache --hicache-storage-backend mooncake` (write-through, best-effort
prefetch). Any worker can therefore pull a prefix that any other worker computed. Commands:
`python mn_run.py coloc_rr_hc coloc_cache_hc` (commits 5eae86c26, a8465ecc7). 0 tracebacks and
0 fallbacks.

| arm | completed agent / chat | agent TTFT ms mean / P99 | chat TTFT ms mean / P99 | chat ITL ms mean / P99 | worker-reported hit | store gets (batched) |
|---|---|---|---|---|---|---|
| coloc_rr | 384 / 3 200 | 11 702 / 61 465 | 8 783 / 68 555 | 25.5 / 909.5 | 12.7% | - |
| coloc_rr_hc | 384 / 3 199 | 9 982 / 37 938 | 5 163 / 38 004 | 20.7 / 153.6 | **22.3%** | 12 649 (3.2 M items) |
| coloc_cache | 384 / 3 200 | 1 596 / 8 371 | 101 / 1 539 | 6.4 / 11.8 | 72.9% | - |
| coloc_cache_hc | 384 / 3 200 | 1 631 / 8 224 | 122 / 1 549 | 6.3 / 11.7 | 72.9% | 16 |

- **Migration works where routing cannot keep a prefix local.** Under round-robin the store
  lifts the hit rate from 12.7% to 22.3% by pulling prefixes other nodes computed. Chat TTFT
  falls 41%, chat P99 ITL 83%, and agent P99 TTFT 38%. Replication happens as a side effect of
  serving (write-through), with no migration step and no knowledge of the holder.
- **With affinity routing the store is idle (16 pulls in the whole run) and changes nothing.**
  Affinity already serves 73% of tokens locally. This is the intended ordering: route to the
  holder first (point 1), and pull from the store only when the router must place a request
  elsewhere (an SLO veto or an overloaded holder, point 2).
- **Not yet shown: the congestion advantage (point 4).** At 16 GPUs on 4 hosts the store moved
  ~236 GB without measurable link contention, so random vs load-aware source choice cannot
  differ yet. That needs either more nodes than the 16-GPU cap allows, or a deliberately
  hot-prefix workload where many nodes pull one session's prefix at once. The bipartite
  disagg design forces exactly that on its few prefill nodes.
- The single failed chat request in coloc_rr_hc was a worker dropping a reused keep-alive
  connection (`ServerDisconnectedError`). The router now retries once before streaming starts
  (a8465ecc7), as the stock router does.

**The SLO veto, calibrated, with worker-side EDF (point 1), 2026-10-08.** The router's
`PerfModel` was fitted on one H20 worker (`mn_run.py calibrate`, `calibrate_h20.py`): prefill
9.0K tok/s at 4-8K falling to 4.9K at 49K, so 5 531 tok/s over 16-49K, and decode ITL 5.05 ms +
0.065 ms per running sequence. The router uses the same length-scaled TTFT deadline as the
workers (2 s + 1 s per 5K tokens beyond 4 096). Workers run deadline-ordered admission with no
chunk cap (`_edf` suffix; the configuration that won on the H200). Command:
`ROUTER_ARGS="--prefill-tok-s 5531 --decode-step-ms 5.05 --decode-ms-per-seq 0.065 --ttft-slo-ms 2000"
python mn_run.py coloc_cache_edf coloc_slo_edf` (commit bf96b6cd3). 0 tracebacks and 0 fallbacks;
every request completed.

| arm | agent TTFT ms mean / P99 | chat TTFT ms mean / P99 | ITL ms agent / chat | vetoes |
|---|---|---|---|---|
| coloc_cache | 1 596 / 8 371 | 101 / 1 539 | 7.2 / 6.4 | - |
| coloc_cache_edf (worker EDF) | 1 611 / 8 102 | 117 / 2 269 | 7.4 / 6.4 | - |
| **coloc_slo_edf** (veto + worker EDF) | **1 582** / 8 132 | 108 / **1 313** | 7.2 / 6.4 | 71 of 3 587 (2.0%) |

- **The veto fires once calibrated** (the uncalibrated run vetoed nothing). It moves 2% of
  requests off their highest-hit node and trims the chat TTFT tail (P99 -15% vs affinity alone),
  with means unchanged. At this load affinity rarely pushes a worker past its SLO, so there is
  little to correct. The veto is a tail guard, and its value should grow with hotspots, which the
  HOT experiment below creates.

**Hot-prefix stress (points 1, 2, 4), 2026-10-08.** `HOT=1`: the chat tenant becomes 4 groups
sharing 16K-token system prompts (`generated-shared-prefix`, 512-token questions, 16/s, 3 200
requests); the agent tenant is unchanged. Command: `HOT=1 RUN_TAG=_hot ROUTER_ARGS="<calibrated>
--ttft-slo-ms 2000" python mn_run.py coloc_cache coloc_slo_edf coloc_slo_edf_hc disagg_2p2d`
(commits 498d60155, daaedef6a; the cache tie-break was already randomized). 0 tracebacks, 0
fallbacks, every request completed.

| arm | agent TTFT ms mean / P99 | chat TTFT ms mean / P99 | ITL ms agent / chat (chat P99) | prefix hit | vetoes | max requests/worker |
|---|---|---|---|---|---|---|
| coloc_cache_hot | 1 673 / 8 602 | 340 / 2 263 | 7.1 / 23.4 (224) | 93.5% | - | 803 |
| coloc_slo_edf_hot | 1 605 / 8 408 | 347 / 3 290 | 7.8 / 19.1 (230) | 93.5% | 96 (2.7%) | 808 |
| coloc_slo_edf_hc_hot | 1 648 / 8 189 | 377 / 3 414 | 7.5 / **18.9 (200)** | 93.4% | 87 (2.4%) | 802 |
| disagg_2p2d_hot | 27 776 / 76 129 | 29 513 / 85 682 | 6.3 / **3.8 (30.5)** | 81.9% | - | 861 |

- **Colocation still wins TTFT by an order of magnitude** (agent -94%, chat -99%), even when the
  workload is designed to create hotspots.
- **CORRECTED 2026-10-08 (see the decode-aware veto below): the "veto is too weak" reading in
  this bullet held only while the veto priced TTFT alone. Pricing the decode slowdown cuts chat
  mean ITL to 8.3 ms and P99 to 44 ms.** Original reading: **Disagg wins chat ITL by 5x here.** A hot group's decode concentrates on its holder in
  colocation (~800 requests on the busiest worker), while disagg spreads decode over 8 dedicated
  GPUs. This is colocation's real weakness, and the veto is too weak to fix it: it moves only
  ~2.5% of requests, because the calibrated model predicts TTFT, and TTFT stays fine on a holder
  whose prefix is cached. The overload shows up as ITL, which the veto does not price. The veto
  plus store trims chat ITL 19% (23.4 -> 18.9 ms) and P99 to 200 ms, and does not close the gap.
- **What point 4 needs and does not yet have:** a veto that prices the decode (TPOT) slowdown a
  request causes on its holder. The router's TPOT model exists (`decode_step_ms`,
  `decode_ms_per_seq`), but with the calibrated slope (0.065 ms/seq) it never crosses the
  100 ms ITL budget. Calibration measured ITL vs concurrency at short contexts, while the holders'
  ITL is driven by 16K-prefix decode rows plus their prefill.
- **Disagg's prefill pods hit 82%** because 4 hot prefixes replicate across 8 prefill GPUs, so the
  bipartite graph is not the bottleneck at 16 GPUs either. No congestion effect is measurable
  within the cap.

**Narrowing the fabric makes congestion measurable within the GPU cap (point 4), 2026-10-08.**
`ONE_NIC=1` pins every pod's store traffic to mlx5_1 instead of spreading over mlx5_1..4, cutting
aggregate cross-node bandwidth ~4x (verified per worker from `/proc/<pid>/environ`:
`MOONCAKE_DEVICE=mlx5_1`, `MC_MS_AUTO_DISC=0`). Same workload, same arm, 4 NICs vs 1:

| coloc_rr_hc (the heaviest puller, ~12.6K batched gets) | 4 NICs | 1 NIC |
|---|---|---|
| agent TTFT ms mean / P99 | 9 982 / 37 938 | 9 861 / 34 510 |
| chat TTFT ms mean / P99 | 5 163 / 38 004 | 4 787 / 31 849 |
| agent ITL ms mean / **P99** | 20.1 / **110.7** | 21.9 / **358.1** |
| chat ITL ms mean / **P99** | 20.7 / **153.6** | 21.5 / **351.4** |

**Tails degrade 2.3-3.2x while means and TTFT are unchanged** -- the signature of link contention,
not compute: a pull that queues behind others stalls one decode step badly rather than slowing
every step. This is the regime point 4's argument is about, and it did not exist on the full
fabric. The source-selection comparison (`coloc_cache_hc_1nic` deterministic best-holder vs
`coloc_rand_hc_1nic` randomized among near-equal holders) runs on this narrowed fabric.
*Caveat on scope:* this shows randomized selection can be tested, not that colocation scales
better with node count. One NIC at 16 GPUs emulates per-link pressure, not a larger cluster's
topology.

**That pair could not isolate source selection -- design flaw, found on the second arm.** With
affinity the narrowed fabric changes nothing (`coloc_cache_hc_1nic`: agent TTFT 1 589 ms, chat P99
ITL 11.8 ms, against 1 631 ms / 11.7 ms on 4 NICs), because affinity keeps 93% of tokens local and
the store is idle. So the only arm that congests (`rr`) ignores cache hits entirely, while the
arms that select on hits never pull. Randomized source selection has no choice to make in either.
The isolating configuration needs both at once: several workers holding the SAME prefix (so hit
lengths genuinely tie) and local misses that pull cross-node. `HOT=1 ONE_NIC=1` with the
decode-aware veto does that -- the veto spreads each shared-prefix group over several workers, which
both replicates the prefix and forces the spread requests to pull. New policy `cache_slo_rand`
(c87aa83b9) is `cache_slo` with the load term dropped inside a hit bucket, i.e. the goal's "pick
randomly among near-equal holders"; it differs from `cache_slo` only when holders really tie.
Running: `HOT=1 ONE_NIC=1 mn_run.py coloc_slo_edf_hc coloc_slo_rand_hc`.

**Randomized source selection under contention: measured, and it LOSES (2026-10-08).** The
isolating pair, both arms identical except the policy -- hot shared prefixes, `ONE_NIC=1`,
decode-aware veto (`--decode-grid`, 12 points), L3 store, worker EDF on both sides. Command:
`HOT=1 ONE_NIC=1 RUN_TAG=_hotnic ROUTER_ARGS="<grid> --ttft-slo-ms 2000 --itl-slo-ms 40"
mn_run.py coloc_slo_edf_hc` then `coloc_slo_rand_edf_hc` (commits c87aa83b9, f6eab46b3).
0 tracebacks, 0 fallbacks, all requests completed in both.

| arm | chat ITL mean / P99 ms | chat TTFT mean / P99 ms | agent ITL mean / P99 ms | agent TTFT ms |
|---|---|---|---|---|
| `coloc_slo_edf_hc_hotnic` (load-ordered within a hit bucket) | **8.9 / 66.5** | **484 / 6 196** | **9.5 / 52.2** | **1 985** |
| `coloc_slo_rand_edf_hc_hotnic` (random within a hit bucket) | 11.0 / 108.1 | 558 / 6 512 | 10.4 / 95.5 | 2 099 |

**Randomizing among near-equal holders is worse on every metric**: chat mean ITL +24%, chat P99 ITL
+63%, agent P99 ITL +83%, chat TTFT +15%. **Why, and why this is the third time:** dropping the
load term is what the policy DOES -- and load, not link bandwidth, is the binding constraint even
on a 4x-narrowed fabric. A holder chosen at random may already be decoding a dozen 16K rows, and
that costs far more than the pull it saves. The same mechanism appeared twice before at full
fabric width (`coloc_rand`, where purely random choice sent hit-0 chat to busy workers, and the
`min()` tie-break hotspot). The fix each time was to order by in-flight load first and use
randomness only to break what remains.

**Point 4's algorithm, as proposed, is therefore refuted at this scale**: "when several nodes have
hit lengths within a threshold, pull from one at random" is dominated by "pull from the least
loaded of them". Randomness is not harmless here; it discards the signal that matters. What the
goal's underlying intuition gets RIGHT is that concentration is the risk -- but the measured cure
is the load-aware tie-break plus the decode-aware veto (chat ITL -65%, P99 -80%, v0.18.0), not
randomization. A congestion-dominated regime where randomization wins would need link cost to
exceed holder-compute cost; one NIC at 16 GPUs does not reach it, and the simulated-node sweep
below is the way to look for it.

Arm 3 of the flawed pair closes it as predicted:Arm 3 of the flawed pair closes it as predicted: `coloc_rand_hc_1nic` (randomized tie-break,
narrowed fabric) measures agent TTFT 1 611 ms / ITL 7.3 ms / P99 13.2 ms and chat 134 ms / 6.5 /
12.2 -- indistinguishable from `coloc_cache_hc_1nic`'s 1 589 / 7.3 / 13.5 and 148 / 6.5 / 11.8.
With affinity keeping 93% of tokens local, neither policy pulls, so the tie-break has nothing to
spread and the narrowed fabric nothing to congest. Both arms: 0 tracebacks, 0 fallbacks, all
requests completed.

A first attempt at this run was DISCARDED before use: the node script was pushed to the pods in
the background while the arms launched, and the push is sequential, so the last pod's workers
started on the old 4-NIC assignment (2 of 16 workers). The identical script mtime on every pod
was misleading -- `tar` preserves mtimes, so it showed the local file's time, not extraction time.
Later arms would have been fully pinned, making the A/B inconsistent.

**The veto has to price TPOT, not TTFT (point 1, resolved 2026-10-08).** The hot-prefix holders'
overload shows up as slow decode, not slow prefill, so a TTFT-only veto never fired on them. Two
fixes, in order:

1. *Calibration.* `calibrate_h20.py` measured decode ITL at 256-token context only, fitting
   0.065 ms/seq. Re-measured at 256 / 4K / 16K context x 1 / 8 / 32 / 64 concurrency, ITL is
   **flat to ~131K resident tokens** (9.1 ms at 32x4K, 8.3 ms at 8x16K) and then jumps to
   **136.6 ms at 524K** (32x16K) and 347.8 ms at 64x16K.
   **CORRECTED 2026-10-08: that knee is PREFILL INTERFERENCE in the measurement, not a capacity or
   bandwidth effect.** (The earlier reading in this bullet -- "a capacity effect, eviction/
   retraction or an attention-path change, root cause not yet found" -- was wrong.) The calibration
   submits all N requests at once, so with 16K prompts the later requests' prefill chunks share
   steps with the earlier ones' decode; at 256-token context prefill is trivial and no
   interference appears. Checked directly: a server held at 32 concurrent requests and 543K
   resident tokens logs **pure-decode** steps at 1 716 tok/s = **~18.6 ms per step**, with token
   usage 0.44 and CUDA graphs on -- 7x faster than the 136 ms the grid records for that shape, and
   no retraction. (The confirming script's own summary was lost: the chain's `KILL` sweep matches
   `launch_server` and killed its server mid-run. The server log carries the numbers.)
   **This does not invalidate the grid for routing**, which needs "what ITL will a request see on
   this worker", and a worker serving a hot prefix really is running prefill beside decode. It
   does mean the grid is a *loaded-worker* cost, not a pure decode cost, so it should not be reused
   as a decode-only model.
2. *Model form.* No closed form fitted that surface: a 3-term linear fit returned a NEGATIVE base
   cost (-4.64 ms), and `max(flat, kv/bw)` overpredicted the middle of the grid 3-4x. The router
   now **interpolates the measured grid** (`--decode-grid`, bilinear, KV-scaled past the corner;
   reproduces every measured point). 16 rows x 16K predicts 51 ms where the old slope predicted
   ~6 ms. Commits 7010f40fe, f6eab46b3.

| hot-prefix arm | chat ITL mean / P99 ms | chat TTFT mean / P99 ms | agent ITL mean ms | vetoes |
|---|---|---|---|---|
| coloc_cache_hot (affinity) | 23.4 / 224 | **340** / 2 263 | **7.1** | - |
| coloc_slo_edf_hot (TTFT-only veto) | 19.1 / 230 | 347 / 3 290 | 7.8 | 96 (2.7%) |
| **coloc_slo_edf_hot2 (decode-aware veto, ITL SLO 40 ms)** | **8.3 / 44.3** | 424 / 5 161 | 8.7 | **192 (5.4%)**, 20 least-bad |
| coloc_slo_edf_hc_hot2 (decode-aware veto + L3 store) | 9.9 / 80.0 | 468 / 6 304 | 9.8 | - |
| disagg_2p2d_hot | 3.8 / 30.5 | 29 513 / 85 682 | 6.3 | - |

**Pricing the decode cost cuts chat mean ITL 65% (23.4 -> 8.3 ms) and P99 80% (224 -> 44 ms)**,
closing most of the gap to disaggregation, for a 25% chat TTFT cost (340 -> 424 ms) and slightly
worse agent ITL. Colocation keeps a ~70x TTFT advantage. This is point 1's mechanism working as
designed: route to the cache holder first, and veto it only when the load it would carry is
predicted to break the SLO -- with the SLO term that the overload actually appears in.

**The L3 store does not help once the veto spreads the load, and costs ~19% chat mean ITL / 80%
P99** (8.3 -> 9.9 ms, 44 -> 80 ms; agent ITL 8.7 -> 9.8 ms). Its write-through traffic and
prefetches compete with decode on the same GPUs, while the vetoed requests already replicate the
hot prefixes by recomputing them. Together with the earlier result that the store is idle under
affinity routing (16 pulls), the rule is: **the store pays only where routing CANNOT place a
request near its prefix** -- round-robin, or a cold cluster -- and is a cost where the router is
already doing that job.

**Point 4, honest conclusion.** The congestion argument is **not demonstrated within the 16-GPU
cap**, in either direction:
- Random choice among near-equal cache holders costs nothing and gains nothing (coloc_rand).
- Disagg's own prefill pods reached 81.9-82% prefix hit on the hot workload, because 4 hot
  prefixes replicate across 8 prefill GPUs. The bipartite graph is not a bottleneck at this size.
- The L3 store moved ~236 GB between nodes with no measurable link contention.
The prediction that colocation's complete graph scales better as nodes increase is therefore
**untested, not supported**: at 16 GPUs on 4 hosts there is no congestion to relieve. Testing it
needs either more nodes than the cap allows or an artificially narrowed interconnect.
- **Worker-side EDF does nothing here** (agent +1%, chat +16 ms, within noise). With 16 workers,
  each sees a light queue, and admission order matters only when work queues up. On a single
  saturated H200 it cut chat TTFT 62% (point 3).

**MTP on every worker (`_mtp`), 2026-10-08.** *Correction, same day: the "colocation regresses
with MTP" reading below is RETRACTED. The cause was a router tie-break hotspot, not MTP; see the
resolution after the table.*
EAGLE3 (3 steps, topk 1, 4 verify tokens) runs on every role. Colocated workers also run the
4-mode step on the FA3/FA2 split (`KERNEL=split`). Before any MTP arm, the H20 serving gate
(`h20_gate.sh`) passed both comparisons with 0 fallbacks, weakly: the split's run had only ONE
mixed step, and one short prompt agreed with stock MTP for 5 tokens (noise floor 101). Commands:
`KERNEL=split python mn_run.py disagg_2p2d_mtp coloc_cache_mtp coloc_cache_edf_mtp` (6561caf51).

| arm | completed agent / chat | agent TTFT ms mean / P99 | chat TTFT ms mean / P99 | ITL ms agent / chat | prefix hit | requests/worker |
|---|---|---|---|---|---|---|
| disagg_2p2d (no MTP) | 384 / 3 173 | 21 581 / 77 164 | 20 147 / 77 131 | 5.8 / 3.7 | 23.3% | 19-885 |
| disagg_2p2d_mtp | 384 / 3 178 | 27 423 / 77 995 | 26 470 / 78 841 | **2.9 / 4.1** | 20.5% | 22-919 |
| coloc_cache (no MTP) | 384 / 3 200 | **1 596** / 8 371 | **101** / 1 539 | 7.2 / 6.4 | 72.9% | 88-329 |
| coloc_cache_mtp | 384 / 3 200 | 18 456 / 142 986 | 42 361 / 142 796 | 14.5 / 24.3 | 72.9% | 90-**812** |

- MTP halves disagg's agent ITL, as expected, and worsens its TTFT: the prefill pods also run
  the draft.
- **Colocation with MTP + 4-mode is ~12x worse on agent TTFT and 2-4x worse on ITL than without
  MTP.** Cache hit is unchanged (72.9%), and per-worker queues stay tiny (max 9 waiting). Chat
  TTFT is 37-93 s over the first 60% of requests, then recovers. Agent TTFT stays at 12-23 s for
  the whole 30-minute run, against a flat 1.4-1.8 s without MTP. One worker takes 812 requests
  (329 max without MTP).
- **Resolution: router hotspot, not MTP.** The `cache` policy broke load ties with a bare
  `min()`, which returns the LOWEST-INDEX worker. Every session's first turn and every chat
  request ties at hit 0, so ties went to workers 0 and 1, and affinity then pinned those sessions
  there: workers 1 and 0 took 811 and 565 requests (61 and 91 agent turns, vs at most 36
  per worker otherwise). The next arm, coloc_cache_edf_mtp, ran the same code with the same bias.
  It happened to spread evenly (max 341 requests, 37 agent turns per worker) and is healthy:

  | arm | agent TTFT ms mean / P99 | chat TTFT ms mean / P99 | ITL ms agent / chat | max requests/worker |
  |---|---|---|---|---|
  | coloc_cache (no MTP) | 1 596 / 8 371 | 101 / 1 539 | 7.2 / 6.4 | 328 |
  | coloc_cache_mtp (hotspot) | 18 456 / 142 986 | 42 361 / 142 796 | 14.5 / 24.3 | 811 |
  | **coloc_cache_edf_mtp** | **1 561 / 8 067** | 114 / 1 382 | **3.8** / 5.7 | 341 |
  | disagg_2p2d_mtp | 27 423 / 77 995 | 26 470 / 78 841 | 2.9 / 4.1 | 919 |

  **Rerun on the fixed router (`RUN_TAG=_fixtie`, 7010f40fe), plus stock MTP without the 4-mode step
  (`_mtps`):**

  | arm | agent TTFT ms mean / P99 | chat TTFT ms mean / P99 | ITL ms agent / chat |
  |---|---|---|---|
  | coloc_cache (no MTP) | 1 596 / 8 371 | **101** / 1 539 | 7.2 / 6.4 |
  | coloc_cache_mtps (stock MTP) | 1 629 / 8 719 | 135 / 3 586 | 4.2 / 5.8 |
  | **coloc_cache_mtp (4-mode, fixed router)** | **1 569** / 8 516 | 122 / 2 607 | **3.9 / 5.7** |
  | coloc_cache_edf_mtp (4-mode + worker EDF) | 1 561 / **8 067** | 114 / **1 382** | 3.8 / 5.7 |

  MTP cuts agent ITL **46%** (7.2 -> 3.9 ms) at unchanged TTFT, and the 4-mode split is slightly
  ahead of stock MTP on every metric (agent TTFT -3.7%, agent ITL -7%, chat P99 TTFT -27%). Worker
  EDF adds a little more in the chat TTFT tail. With MTP + the 4-mode split, colocation keeps its
  TTFT and cuts agent ITL 46% against colocation without MTP. Against disagg with MTP on both sides: TTFT -94% (agent) and
  -99.6% (chat). Disagg's dedicated decode GPUs still give lower ITL (2.9 / 4.1 vs 3.8 / 5.7 ms).
  The tie-break is now randomized among equally loaded workers. The stock-MTP arm and a rerun of
  coloc_cache_mtp run on the fixed router.


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
      - [x] **Same comparison at a moderate load, chat 16/s, 3 rounds (2026-10-08).**
        `TAG=_itl30{,_r2,_r3} ARMS="MTP_LPT MTPMIX_SPLIT_EDFP" mtp_edf.sh 16 2000 30`. All requests
        completed, 0 fallbacks. Means:

        | metric | stock MTP | full stack | change |
        |---|---|---|---|
        | chat TTFT ms | 256 | **104** | **-59%** |
        | chat mean / P99 ITL ms | 13.95 / 82.7 | **8.95 / 44.3** | **-36% / -46%** |
        | codex TTFT ms | 824 | 667 | **-19% (below the 30% bar)** |
        | codex mean / P99 ITL ms | 8.53 / 48.1 | **5.44 / 27.1** | **-36% / -44%** |
        | chat SLO attainment | 0.740 | **0.907** | |

        At this load stock MTP is not yet queueing codex, so there is less TTFT to remove. Codex
        TTFT misses the bar here and clears it at 24/s (-51%). Every ITL metric clears it at both
        loads.
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
