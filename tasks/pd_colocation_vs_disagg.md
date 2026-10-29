# Current goal

Make the case for **PD colocation with an EDF/SLO-aware router** against **PD disaggregation**,
for a 4x8 H100/H200 cluster serving agentic traffic (TraceLab-shaped: 64-100K contexts, ~3.8K
appends, ~480-token outputs, mixed with chat and math).

Tasks, in priority order (status 2026-10-09; evidence in the sections below).

---

## [2026-10-12] SERVING EVIDENCE USED A CORRUPTED MODEL -- re-measurement in progress

Every serving run behind this doc passed the YaRN `--json-model-override-args`, which under
transformers 5.x reset Qwen2.5's `rope_theta` from 1e6 to 1e4 (sglang's override replaced the rope
dict). Fixed in sglang's config loader; details and bisection in
`debug_kernel_server_perf/resume_20261012/a2_serving_ropefix_20261012.md`.

What that does and does not touch, point by point:

1. **Cache-hit-first routing with an EDF/SLO check** -- the policy argument is unaffected; any
   serving latency used to motivate it was measured on the corrupted model and must be re-run.
2. **Best-effort migration / replication** -- unaffected in logic (no serving numbers depend on
   the model's outputs).
3. **EDF predictor accuracy** -- the step model (`heldout_mape=21.9%`) and the EDF A/Bs were fitted
   and measured on the corrupted model. Step *latency* depends weakly on outputs, but acceptance
   length and request lengths do not, so the fit and its SLO results are re-measured before use.
   The 2026-10-11 conclusion that the fused kernel is harder to model (kernel harness, no model
   outputs involved) stands.
4. **Two-way RDMA / complete graph** -- unaffected in logic.

The kernel-harness numbers in the 2026-10-11 section below (fused vs `BatchAttention`) do not
involve the served model and are unaffected. The headline serving comparison was re-run on the
fixed model (`agent_chat_rated`, chat 24/s + TraceLab codex 0.1/s, one round, no graphs): the
**unfused** 4-mode schedule beats stock MTP by TTFT -32% and throughput +10%/+13% (mean ITL -13%),
and the co-compiled fused kernel lost 11-13% throughput against it. After moving FA2 out of the
co-compiled kernel (PDL co-launch, classes 64/32/16, auto FA2 chunk) the fused 4-mode arm beats
stock MTP by TTFT -55% / -58% and throughput +18% / +21% (chat / codex), mean ITL -19% / -11%, and
the unfused 4-mode arm by +7% / +6% throughput (2 rounds, no CUDA graphs)
(`debug_kernel_server_perf/resume_20261012/colaunch_20261012.md`). The EDF step model (point 3)
must be re-fitted on this kernel before its accuracy claim is tested.

---

## [2026-10-11] CORRECTED KERNEL EVIDENCE -- supersedes the 2026-10-10 section below

The 2026-10-10 update to this doc leaned on two kernel figures that are now **RETRACTED**, because
both compared against an arm that fails the causal fp32 reference (`MIX_ALL_FA3=1`, a diagnostic
configuration, failing 6/6 at 68-116 bad rows of 384):

* ~~"the fusion is 1.46-2.01x slower than all-FA3 on realistic batches"~~ -- the baseline was broken;
* ~~"3.14x slower than `BatchAttention`"~~ -- measured under the same knob, with a fence since shown
  to be a regression.

**The correctness-gated re-measurement is worse for the fusion, not better.** Two-role path
(`MIX_ALL_FA3=0`), backfill live, bad rows beside every timing:

| dec | sm_split | verdict | fused | `BatchAttention` | ratio |
|---|---|---|---|---|---|
| 8 | 96 | 0/264 | 0.3670 ms | 0.0699 ms | **5.3x slower (best case)** |
| 32 | 96 | 0/288 | 1.2560 ms | 0.1180 ms | 10.6x |
| 128 | 112 | 0/384 | 8.1775 ms | 0.2740 ms | **30.0x** |

**What this means for each of the four points, stated against corrected evidence:**

1. **Route to highest cache-hit node, EDF/SLO check** -- **UNAFFECTED.** This is a router policy and
   does not depend on the fused kernel. The argument in section 1 stands on its own.
2. **Best-effort cache migration / replication** -- **UNAFFECTED**, same reason. Section 2 stands.
3. **"Our EDF predictor is more accurate because the fused kernel is easier to model"** --
   **THE PREMISE IS DEAD, and now for a better-evidenced reason than the 2026-10-10 entry gave.**
   That entry said the premise was inverted by a "23x `sm_split` cliff". The corrected measurement
   shows something simpler and more damaging: latency varies by **5.3x to 30x across `sm_split` and
   decode count on correct arms** (0.367 -> 8.18 ms). A kernel whose runtime moves by 22x across its
   own configuration space is *harder* to model than the unfused baseline, not easier. **Point 3
   must be dropped or re-argued without the kernel.**
4. **Two-way RDMA / complete graph** -- **UNAFFECTED.** A network-topology argument, independent of
   the attention kernel.

**So three of the four points survive intact and one is dead.** The surviving three are all
*scheduling and networking* arguments that never needed the fused kernel; the one that depended on
it is the one that fails. That is a cleaner statement of this doc's thesis than the kernel-first
framing: **the colocation case rests on the router, not on the attention kernel.**

**And the FA3-only alternative does not rescue point 3 either** -- it would run the all-FA3 path,
which fails 6/6 on the fp32 reference. Both kernel designs are currently evidence-blocked, so point
3 has no working kernel to stand on in either form.

---


**The thesis in one line:** our policy space (4-mode colocation + EDF/SLO router + tunable P/D
admission ratio) is a **strict superset** of every baseline's fixed policy, so we should match
the best baseline at any static workload and beat all of them when the workload moves.
Baselines are two-tier: **single node** = prefill-prioritized (the default) and mixed chunked
prefill; **multi node** = PD multiplex and PD disagg (1P3D, 2P2D).

**Item 5 carries that claim and is the weakest-evidenced part of the thesis:** nothing measured
so far varies the workload mixture, shifts it mid-run, or compares against any baseline on
goodput. Items 1-3 measure routing, cache replication and EDF ordering against 2P2D at a single
static mixture.

- [x] **1. Cache-hit-first routing with an SLO check** (2026-10-08, H20 16 GPUs) — cache-affinity
      colocation vs the best disagg (2P2D): agent TTFT -93% (21.6 -> 1.6 s), chat TTFT -99.5%,
      prefix hit 73% vs 23-31%. Disagg keeps lower ITL (3.7 vs 6.4 ms). The calibrated SLO check fires on 2% of
      requests and trims chat P99 TTFT 15%.
      Hot-prefix stress (4 shared 16K prompts piled on their holders): a TTFT-only SLO check barely
      fires, because a cache holder's PREFILL stays fast while its DECODE degrades. Pricing the
      decode slowdown (`--decode-grid`) cuts chat ITL **-65%** and P99 **-80%** for a 25% chat TTFT
      cost (v0.18.0). Sub-items of this point:
      - [x] route to the highest-hit node first (affinity beats rr and disagg by 8-99x on TTFT)
      - [x] SLO check on predicted SLO violation, counting only EARLIER-deadline in-flight requests
            (`router.py feasible()`: it sums the prefill ahead of this request's deadline, and
            re-checks that admitting it does not push later-deadline requests past their own)
      - [x] the SLO check must price the term the overload appears in (TPOT, not just TTFT)
      - [x] load balancing: tie-break on hit bucket, then in-flight load, then random (a bare
            `min()` created a 2-worker hotspot; retracted and fixed, daaedef6a)
- [x] **2. Best-effort cache migration / replication** (2026-10-08) — a cluster-wide mooncake L3
      store (write-through, best-effort prefetch) lifts round-robin's hit rate 12.7% -> 22.3% (chat
      TTFT -41%, P99 ITL -83%). Under affinity it sits idle (16 pulls): affinity already serves 73%
      locally, so migration pays only where routing must place a request elsewhere.
- [ ] **2b. A workload where replication + load balancing must matter (design, 2026-10-18, user; 4 nodes of 8
      H200 / H100, scaled to H20 4 x 4 within the cap).** The earlier runs were neutral because no request had to
      leave its prefix holder (affinity served 73-93% of tokens locally, the store sat idle) and recompute was cheap
      (16K hot prefixes ~1 s). The case needs three conditions at once:
      1. **Demand per prefix above one node's capacity** -- *fan-out agents*: K tasks, each a long shared context
         (100-200K tokens, a TraceLab Claude/Codex session prefix), each spawning N parallel programs (best-of-N
         rollouts, sub-agents on one repository; N = 8-32) that append their own rounds. Affinity piles all N on the
         holder; spreading without a copy recomputes 100K+ tokens (~10 s on H200), a pull ships ~5.7 GB of KV (Qwen 7B,
         ~57 KB/token; ~0.15 s at 400 Gb/s), so a replica is worth ~60x its transfer.
      2. **Working set above one pool's memory, below the cluster's** -- sum of hot contexts > the prefill pool's
         KV + host cache in 2P2D-style disagg, < all nodes' together; with tool-call gaps between rounds (ThunderAgent's
         thrashing), disagg's pool evicts and re-prefills while colocation keeps copies across every node's memory.
      3. **Popularity that shifts** -- Zipf over tasks whose ranking rotates every few minutes (a new task's fan-out
         starts while an old one drains), so any static placement or pool sizing lags and replicas must follow demand.
      Plus chat at a fixed rate on every node as the latency-sensitive tenant. Load: agent program concurrency C per
      node (never sessions/s), chat by request rate, both around measured capacity.
      **Arms:** PD disaggregation (sglang P/D roles + cache-aware router; with and without MTP on decode), colocation
      with affinity only (no migration), colocation round-robin + store, ours (SLO-checked hit-first routing +
      best-effort pull + EDF, 4-mode step with MTP). **Measured, besides TTFT / ITL / throughput:** re-prefilled tokens
      (the falsifier: ours must recompute less than disagg at equal memory), replica count per hot prefix over time,
      pulled bytes and their latency, max/mean per-node load, and the time to recover TTFT after each popularity shift.
      Needs: a fan-out loader (shared-prefix programs on top of the TraceLab replay), per-node replica / recompute
      counters from the radix and store logs.
      - [x] 2026-10-18 loader `drivers/serving/fanout_agents.py` (K tasks x L-token shared prefix, program concurrency C,
        Zipf(s) task choice over a ranking that rotates every ROTATE s, R rounds with tool gap G and A-token appends;
        per-request TTFT / ITL / cached tokens / popularity epoch, TTFT recovery after each rotation). Checked against
        a mock streaming server only (623 requests, 4 tasks, rotation and Zipf draw as specified); not yet run on GPUs.
      - [x] **First run, 8 H20s (job bb854a10b8954223, 4 pods x 2 GPUs; 2026-10-18; `bash
        tasks/pd_colocation/mn_fanout_8gpu_20261018.sh`; 16 tasks x 64K prefixes, 32 programs, 4 rounds, 5 s tool
        gaps, Zipf 1.2 rotating every 180 s, 900 s, + ShareGPT 8 req/s; every arm 0 errors / 0 tracebacks / 0
        fallbacks).** Agent out tok/s, TTFT mean / p99, ITL mean ; prefix cached ; chat TTFT / ITL:
        | arm | agent | cached | chat |
        |---|---|---|---|
        | PD disagg 2P2D, stock MTP (clean upstream tree) | 44 / 62.6 s / 240 s / 148.8 ms | 83.0% | 75.6 s / 57.7 ms |
        | colocation, prefix affinity only (no pool) | 405 / 1.18 s / 19.9 s / 85.5 ms | 97.9% | 74.5 ms / 5.4 ms |
        | **ours: SLO-checked hit-first routing + pool + transfer term + EDF + 4-mode MTP** | **505 / 0.94 s / 6.75 s / 47.1 ms** | 97.7% | 139 ms / 7.0 ms |
        Ours vs PD disagg: agent throughput 11.5x, TTFT -98.5%, ITL -68%; chat TTFT -99.8%, ITL -88%, chat throughput
        +18% (1647 vs 1395 tok/s). Disagg does not recover from the rotating hot set: its agent TTFT grows epoch over
        epoch (~37 s -> 104 s), while both colocation arms hold 0.5-1.2 s after the first epoch's cold start. Ours vs
        affinity only (isolates the routing + replication policy): agent throughput +25%, TTFT -20%, p99 -66%, ITL
        -45%; chat TTFT 74 -> 139 ms and ITL 5.4 -> 7.0 ms -- worse, still far inside a 2 s / 50 ms SLO. Caveats:
        single repeat; disagg runs clean upstream sglang + stock MTP, so ours-vs-disagg folds kernel and policy
        together (the affinity arm is the policy-only control); disagg is overloaded at this load. Our patched tree
        cannot run disagg + MTP: its prefill workers assert at warmup (`disaggregation/prefill.py`
        `batch.spec_info is result.next_draft_input`), so that baseline runs on `up_clean`.
      - [x] **Lighter load, 8 H20s (2026-10-18; 16 programs, chat 4 req/s / 3600 requests, otherwise as above; all arms
        0 errors / 0 tracebacks / 0 fallbacks; `AGENT_CONC=16 CHAT_RATE=4 CHAT_N=3600 REP=_light ARMS="disagg_2p2d_mtps
        disagg_2p2d coloc_cache_edf_mtp coloc_slo_edf_mtp_hc" bash tasks/pd_colocation/mn_fanout_8gpu_20261018.sh`).**
        | arm | agent tok/s / TTFT mean / p99 / ITL | cached | chat TTFT / ITL |
        |---|---|---|---|
        | PD disagg 2P2D, no spec | 122 / 10.85 s / 56.4 s / **6.1 ms** | 88.6% | 6.90 s / 5.4 ms |
        | PD disagg 2P2D, stock MTP | 107 / 11.31 s / 70.7 s / 23.0 ms | 86.9% | 7.35 s / 7.0 ms |
        | colocation, affinity only | 262 / 0.93 s / 3.95 s / 41.4 ms | 97.6% | 59 ms / 4.6 ms |
        | ours | **281 / 0.83 s** / 14.0 s / 31.6 ms | 97.5% | 64 ms / 4.7 ms |
        Ours vs disagg without spec (the stronger disagg here): agent throughput +130%, TTFT -92%, chat TTFT -99%,
        chat ITL -13% -- **but agent ITL is 5x worse (31.6 vs 6.1 ms)**: disagg's two decode pods are lightly loaded
        and isolated, while ours carries agent decode rows through mixed steps that prefill 64K-prefix chunks (the
        colocation weakness seen in the hot-prefix test). Disagg's prefill pods are its bottleneck (TTFT ~11 s, slowly
        recovering epoch over epoch); MTP on its decode pods makes agent ITL worse here (23 vs 6.1 ms). Ours vs
        affinity only: +7% / -11% / -24% ITL, p99 TTFT worse (14.0 vs 3.95 s). Single repeat.
      - [x] Smaller prefill chunks do not buy agent ITL here (light load, ours; clean): chunk 512 agent 259 tok/s / 1.24 s
        / 35.1 ms, deadline chunk 512/2048 259 / 1.09 s / 39.8 ms, vs chunk 2048 281 / 0.83 s / 31.6 ms. Agent ITL is
        not set by mixed-step prefill length on this workload; the likelier cause is load: ours decodes 2.3x the agent
        tokens per second of disagg (281 vs 122 tok/s; disagg's prefill pods throttle how many agents decode at once),
        so the fair comparison is ITL at matched agent throughput. `CHUNK=512` / `CHUNK=2048
        DEADLINE_CHUNK=512:2048:5019` with `REP=_light_ch512` / `_light_dl512` (`/tmp/h20/fanout_itl.sh`).
      - [x] Ours at fewer programs (light load, clean): C=8 agent 150 tok/s / 0.80 s / ITL 23.0 ms, C=4 74 tok/s /
        1.09 s / ITL 18.0 ms. At C=8 ours already exceeds disagg's agent throughput (150 vs 122) with its MTP ITL (23.0
        ms = disagg + MTP), and even at C=4 (74 tok/s) ITL stays 3x disagg's no-spec 6.1 ms -- so the gap is per-step
        MTP cost, not load: the driver's random-id prefixes and appends give near-zero draft acceptance, so MTP drafts
        and verifies for nothing (which is also why MTP made disagg worse). Next: natural-text prefixes
        (`fanout_agents.py --text-source`, ShareGPT slices; `FO_TEXT=1`), ours with and without MTP vs disagg with and
        without MTP. `for C in 8 4; do AGENT_CONC=$C REP=_light_c$C ARMS=coloc_slo_edf_mtp_hc ...` (`/tmp/h20/fanout_pareto.sh`).
      - [x] **Metric correction (2026-10-18): the fan-out driver's "ITL" is the gap between streamed chunks, and an MTP
        step streams ~2-3 tokens per chunk, so for MTP arms it is a per-step time, biased 2-3x against MTP.** Per-token
        TPOT ((e2e - TTFT) / (tokens - 1), recomputed for every arm from the saved records, `tasks/perf_model/fanout_tpot.py`) replaces
        it below and supersedes the ITL columns above for MTP arms: heavy load, ours 16.8 ms vs disagg + MTP 127 ms;
        light load ours 11.2 ms, affinity-only 14.7, disagg 6.1 (no spec) / 19.7 (MTP); ours at C=8 / 4 8.0 / 6.4 ms.
      - [x] **Natural-text fan-out, light load (2026-10-18; `FO_TEXT=1`: prefixes and appends are ShareGPT slices; 16
        programs, chat 4 req/s; all four arms 0 errors / 0 tracebacks / 0 fallbacks; `bash /tmp/h20/fanout_text.sh` =
        `FO_TEXT=1 AGENT_CONC=16 CHAT_RATE=4 CHAT_N=3600 REP=_light_text ARMS="disagg_2p2d disagg_2p2d_mtps
        coloc_slo_edf_mtp_hc coloc_slo_edf_hc" bash tasks/pd_colocation/mn_fanout_8gpu_20261018.sh`).**
        | arm | agent tok/s / TTFT / TPOT | chat TTFT / ITL |
        |---|---|---|
        | PD disagg, no spec | 128.5 / 10.05 s / **6.3 ms** | 10.20 s / 5.3 ms |
        | PD disagg, stock MTP | 104.1 / 11.60 s / 22.1 ms | 10.65 s / 7.4 ms |
        | ours + MTP | 246.4 / 0.91 s / 18.5 ms | 66 ms / 4.9 ms |
        | ours, no MTP | **279.0 / 0.74 s** / 12.3 ms | 63 ms / 6.2 ms |
        Ours + MTP vs disagg + MTP: agent +137% / -92% / TPOT -16%; chat TTFT -99%, ITL -34%. Ours without MTP vs
        disagg without: agent +117% / -93% / TPOT +97% (worse); chat TTFT -99%, ITL +17% (worse). MTP raises agent
        TPOT in both systems on H20 even with natural text: H20's low compute (vs its bandwidth) makes the 4-token
        verify over 64K+ contexts cost more than the accepted drafts save. Disagg's low TPOT comes with half the agent
        throughput (its 2 prefill pods throttle how many agents decode), so the fair TPOT comparison is at matched
        throughput: ours at C=8 (150 tok/s > disagg's 128) gave TPOT 8.0 ms with MTP on random text.
      - [x] **Ours without MTP at matched throughput, natural text (clean):** C=8 agent 145.7 tok/s / TTFT 0.65 s / TPOT
        10.8 ms; C=4 72.7 tok/s / 0.74 s / 10.0 ms (chat 56 / 53 ms, 5.8 / 5.5 ms). At C=8 ours serves more agent
        throughput than disagg (145.7 vs 128.5) with TTFT -94% but TPOT 1.7x disagg's 6.3 ms -- and TPOT barely moves
        from C=16 (12.3) through C=4 (10.0). *Corrected 2026-10-18 (decode-kernel check below): the reading "neither
        load nor prefill interference; our decode step is slower" was wrong -- these points vary agent concurrency only,
        while chat prefill stays at 4 req/s, so the interference was held constant, not removed.* `for C in 8 4; do FO_TEXT=1
        AGENT_CONC=$C ... REP=_light_text_c$C ARMS=coloc_slo_edf_hc ...` (`/tmp/h20/fanout_text2.sh`).
      - [x] **Decode-kernel check at C=4 (natural text, clean): not the kernel.** Stock kernel (`KERNEL=lpt`), persistent
        path on / off (`FI_PERSISTENT=1/0`): agent 67.2 / 69.0 tok/s, TPOT 12.9 / 12.1 ms -- no better than our split
        kernel (10.0 ms), and the persistent path makes no difference. What separates colocation from disagg's 6.3 ms
        is the steps decode shares: on colocated nodes every chat prefill (constant 4 req/s) and every 1024-token
        agent append is a mixed step the decoding agents ride through, while disagg's decode pods run decode only.
        This is the colocation cost the 4-mode step and EDF reduce (ours 10.0-12.3 ms vs the stock kernel's 12-13 ms)
        but do not remove at this load. `for FP in 1 0; do FI_PERSISTENT=$FP KERNEL=lpt FO_TEXT=1 AGENT_CONC=4 ...
        REP=_light_text_c4_lpt_fp$FP ARMS=coloc_slo_edf_hc ...` (`/tmp/h20/fanout_pers.sh`).
      - [x] **Heavy load on natural text (repeat of the first run with ShareGPT text and both disagg variants; 32
        programs, chat 8 req/s; all four arms 0 errors / 0 tracebacks / 0 fallbacks; `bash /tmp/h20/fanout_heavy_text.sh`
        = `FO_TEXT=1 REP=_text ARMS="disagg_2p2d disagg_2p2d_mtps coloc_slo_edf_mtp_hc coloc_slo_edf_hc" bash
        tasks/pd_colocation/mn_fanout_8gpu_20261018.sh`).** Agent tok/s / TTFT / per-token TPOT ; chat tok/s / TTFT / ITL:
        | arm | agent | chat |
        |---|---|---|
        | PD disagg, no spec | 177.9 / 17.21 s / **5.2 ms** | 1618 / 18.90 s / 5.0 ms |
        | PD disagg, stock MTP | 51.7 / 52.29 s / 104.8 ms | 1318 / 64.60 s / 50.0 ms |
        | ours + MTP | 439.7 / **0.86 s** / 27.0 ms | 1642 / 0.198 s / 9.3 ms |
        | ours, no MTP | **466.5** / 1.14 s / 20.4 ms | 1643 / 0.135 s / 7.5 ms |
        Ours without MTP vs disagg without: agent throughput +162%, TTFT -93%, TPOT 4x worse; chat TTFT -99%, ITL +50%.
        Ours + MTP vs disagg + MTP: +750% / -98% / TPOT -74%, chat +25% / -99.7% / -81%. The first run's "11.5x" was
        against disagg + MTP, the weak baseline on H20; against disagg without spec ours serves 2.6x the agent tokens.
      - [x] **Agent SLO goodput (requests meeting both a TTFT and a per-token TPOT SLO; `tasks/perf_model/fanout_slo_goodput.py` on the saved
        records).** Attainment / goodput tok/s at TTFT 10 s + TPOT 80 ms (the eval matrix's agent SLO), and at 20 s + 30
        ms (a tight TPOT): heavy -- ours no MTP 98.0% / 457 and 81.2% / 379, ours + MTP 97.6% / 429 and 65.4% / 288,
        disagg no spec 48.2% / 86 and 62.1% / 111, disagg + MTP 29.1% / 15; light -- ours no MTP 98.0% / 273 and
        96.8% / 270, disagg no spec 65.8% / 85 and 79.7% / 102. **Ours delivers 3.2-5.3x disagg's agent goodput**
        (3.4x even at a 30 ms TPOT SLO): disagg's per-token decode is faster, but its two prefill pods make half its
        requests miss the TTFT deadline. Per-token TPOT is the one metric where disagg leads (decode isolation).
      - [ ] **2026-10-20 multi-node SLO control (decode-first workers + node-reported load + load-aware router), 8 H20s,
        text fan-out, full fabric, 180 s rotation; `mn_slo_control_8gpu_20261020.sh`.** Pieces: decode-first worker rule
        (`apply_edf_itl_slo_chunk.py`, 20 ms ITL SLO, no exception), LoadStat with decode rows / KV / acceptance / prefill
        queue by deadline (`apply_load_stat_4mode.py`, direct PUB on the bootstrap port), router `--slo-chunk` +
        `--load-endpoints` (TTFT walked chunk by chunk under each node's decode load; infinite on a decode-full node).
        Two bugs found and fixed on the way: (1) a 1-token floor rounded to a zero-token extend, (2) under mixed chunk the
        scheduler charges one token per decode row against the chunk, so a ~100-token cap beside 50+ rows left a zero
        prefill share -- both crashed every worker in flashinfer's planner (`kv_len[0]-1 should be non-negative`); the
        two collapsed runs (35-41K errors) are void, as are three overlapping runs from a malformed queue script.
        Clean results, agent tput / TTFT / ITL ; chat TTFT / ITL (p99 ITL):
        | arm | rep 1 | rep 2 |
        |---|---|---|
        | ours, no SLO control | 549.1 / 0.75 s / 12.4 ms ; 294 ms / 10.7 ms (253) | 561.2 / 0.71 s / 12.3 ms ; 269 ms / 10.4 ms (118) |
        | ours, SLO control, grid-fit decode model | void | 244.6 / 8.19 s / 26.5 ms ; 4.54 s / 16.1 ms (61) |
        | PD disagg 2P2D, no spec | 181.6 / 16.62 s / 5.0 ms ; 18.59 s / 5.0 ms (14) | 182.8 / 16.37 s / 5.0 ms ; 18.05 s / 5.1 ms (14) |
        **With the grid-fit decode model, SLO control loses to our own uncontrolled arm on everything but chat p99 ITL**:
        the model (fit on distinct-context calibration rows, 0.114 ms per KV-ktok) prices 4 rows at 68K depth at the whole
        20 ms budget while the same load measures ~12 ms ITL, so the rule capped prefill to ~10 tokens / step. It still
        beats disagg on throughput (+34%) and TTFT (-50%, chat -75%).
        Measured decode-step base (`ITL_SLO=...:measured`, LoadStat `decode_step_ms`, rep 1, clean): 305.7 / 5.38 s /
        20.8 ms ; 4.14 s / 13.9 ms (p99 62) -- better than the grid model, still far behind no control. The measured
        decode step is genuinely ~20 ms here (16 rows x ~68K KV ~ 63 GB read per step on H20), so a 20 ms per-step budget
        leaves ~no prefill. **Per tenant (`fanout_per_tenant.py`, rep 2 / measured rep 1): chat attainment at 0.5 s + 20 ms
        -- no control 91.4%, SLO control measured 65.1%, grid 50.3%, disagg 26.2%; agent goodput at 10 s + 80 ms -- 551 /
        262 / 176 / 92 tok/s.** The per-step rule lowers attainment: chat TPOT p90 is already 13.6 ms without control, so
        the ITL SLO was not binding, and capping every step to it costs TTFT (and raises TPOT p90 to 28 ms as requests
        pile up). The SLO, and how attainment is scored, is a request's mean TPOT, so the next arm budgets per running
        row: x <= s*(o + acc) - E (no row ever past its SLO; fast rows lend slack), chat 20 ms / agents 80 ms
        (`ITL_SLO=...:measured:slack:cls24000_80`, router per class; `mn_slo_slack_8gpu_20261020.sh`).
        Measured rep 2: 381.7 / 3.04 s / 17.4 ms ; 1.01 s / 12.0 ms (p99 53). **Per-row slack rep 1 is worse still:**
        142.9 / 16.25 s / 26.1 ms ; chat 23.53 s (p99 214 s) / 18.0 ms (p99 58), 4 errors, 0 tracebacks. Caps were live
        (108-203 tokens at 66-69K depth where rows had slack, 2-19 where they had none); rep 2 (146.8 / 17.58 s /
        27.7 ms ; chat 12.92 s, p99 104 s / 18.0 ms, 1 error) reproduces it. The measured decode step
        (~20 ms) sits at the chat SLO, so chat rows never bank slack and any row slightly behind drives the budget
        negative -- no prefill on that node at all. **Conclusion (H20, this fan-out load): no strict decode-first policy
        can hold chat at 20 ms TPOT beside long-context agent decodes without starving prefill; every variant lowers
        attainment below the uncontrolled arm (chat 91.4%, agent goodput 551 tok/s), which already beats disagg (26.2%,
        92 tok/s) because disagg reaches its 5 ms decode only by keeping few requests in flight.** The SLO-checked router
        without worker-side decode-first stays the multi-node configuration.
        sglang `079a8a6dc`.
      - [x] **32 H20s, disagg at every P:D ratio (2026-10-21; job 2f608c1b1a67cf09, 8 pods x 4, user cap 32 GPUs;
        Qwen2.5-7B TP1 per GPU = 32 workers, no speculation; text fan-out x4 the 8-GPU load: 64 tasks x 64K prefixes, 128
        agent programs, 4 rounds + ShareGPT chat 32 req/s, 900 s; router calibrated on one worker -- warmed with a
        4096-token prompt after the first calibration's 4096 point read 7.1 s; `bash
        tasks/pd_colocation/mn_scale32_20261021.sh`; every arm 0 errors / 0 tracebacks; sglang 5e5efa3cf).** One repeat each:
        | arm | agent tput / TTFT mean / p50 / ITL | prefix cached | chat TTFT mean / p50, TPOT |
        |---|---|---|---|
        | **ours** (SLO-checked hit-first routing + pool + EDF) | **2153 / 0.93 s / 0.44 s / 12.9 ms** | 0.975 | **0.08 s / 0.05 s, 7.2 ms** |
        | disagg, 2 prefill + 6 decode pods | 125 / 110 s / 105 s / 2.8 ms | 0.62 | 95.0 s / 90.2 s, 83.5 ms |
        | disagg, 4 + 4 | 244 / 57.0 s / 45.4 s / 3.0 ms | 0.63 | 44.7 s / 31.7 s, 3.8 ms |
        | disagg, 5 + 3 | 303 / 44.3 s / 38.2 s / 3.4 ms | 0.64 | 34.1 s / 26.4 s, 4.0 ms |
        | disagg, 6 + 2 | 264 / 41.9 s / 30.8 s / 4.7 ms | 0.60 | 29.9 s / 18.1 s, 30.4 ms |
        Disagg improves with prefill share up to 5 of 8 pods, then decode becomes the bottleneck (chat TPOT 30 ms at 6 + 2).
        Even at its best split it serves agents at 1/7 of our throughput with ~45x the TTFT. Its prefix-cached fraction
        stays ~0.6 vs 0.975 for ours, and ours had the cluster KV pool while disagg had none. **Attribution (same load,
        one repeat each; `mn_scale32_fair_20261021.sh`, `/tmp/run_p5hc.sh` equivalent: `ARMS=disagg_p5_hc`):**
        | arm | agent tput / TTFT | prefix cached | chat TTFT / TPOT |
        |---|---|---|---|
        | ours without the pool (router told there is none) | 2092 / 0.99 s | 0.971 | 0.07 s / 7.7 ms |
        | disagg 5 + 3 with the pool on the prefill pods | 353 / 36.2 s | 0.722 | 29.6 s / 4.6 ms |
        **The pool is not where the gap comes from**: ours loses 3% without it (hit-first routing alone keeps 97% of
        each prefix local), and giving disagg the pool lifts it 16% (0.64 -> 0.72 cached). The gap is structural to
        stock sglang PD on agent sessions: a session's history is decoded on a decode pod, so the next round's prefill
        on a prefill pod misses it (cached 0.6-0.72) and re-prefills it, and the prefill pods queue every tenant. (PD
        decode servers run a chunk cache, so under disagg the pool can only serve the prefill pods.) Systems that keep
        a session on one node across rounds (Dynamo's conditional disaggregation) would narrow this; see point 5.
      - [x] **Per-tenant SLO attainment of the 32-GPU sweep, and MTP at 16 / 32 GPUs (2026-10-21; Qwen2.5-7B, trained
        EAGLE3 head at real acceptance ~2.0, chat prompts through the chat template; 16 GPUs = pods 0-3 at half the load:
        32 tasks, 64 programs, chat 16 req/s; `bash tasks/pd_colocation/mn_mtp_scale_20261021.sh`, scoring
        `fanout_per_tenant.py`; sglang 8f5230e60).** Chat attainment at 0.5 s + 20 ms / 2 s + 50 ms; agent goodput at
        10 s + 80 ms:
        | GPUs, spec | arm | chat att. | agent goodput | agent tput / TTFT | errors |
        |---|---|---|---|---|---|
        | 32, none | **ours** | **97.0 / 99.6%** | **2106 tok/s** | 2153 / 0.93 s | 0 |
        | 32, none | ours without the pool | 97.0 / 99.4% | 2010 | 2092 / 0.99 s | 0 |
        | 32, none | disagg 2 + 6 / 4 + 4 / 5 + 3 / 6 + 2 | 7.6 / 17.3 / 19.4 / 24.1% (2 s + 50 ms: 7.8-27.0%) | 1 / 30 / 46 / 46 | 125-303 / 42-110 s | 0 |
        | 32, none | disagg 5 + 3 with the pool | 24.7 / 27.1% | 82 | 353 / 36.2 s | 0 |
        | 32, MTP | **ours** | **91.1 / 99.3%** | **1727** | 1777 / 1.17 s | 0 |
        | 32, MTP | disagg 5 + 3 | 4.2 / 4.7% | 2 | 235 / 57.2 s | 0 |
        | 32, MTP | disagg 6 + 2 | 7.2 / 7.3% (20,323 of 28,800 chats completed) | 0 | 38 / 121 s | 79 |
        | 16, MTP | **ours** | **90.3 / 96.1%** | **883** | 907 / 0.91 s | 0 |
        | 16, MTP | disagg 2 + 2 | 0.5 / 0.9% | 3 | 119 / 57.5 s | 0 |
        | 16, MTP | disagg 3 + 1 | 41.2 / 42.5% (7,589 of 14,400 completed) | 1 | 12 / 157 s | 48 |
        **At these loads disaggregation is overloaded at every P:D ratio, and ours is not**: the prefill side queues
        every tenant (TTFT 30-110 s) and re-prefills agent history it does not hold (cached 0.46-0.72), and giving
        decode fewer pods to add prefill (6 + 2, 3 + 1) collapses decode instead. With MTP on H20 at ~2.0 accepted tokens
        ours loses ~18% agent throughput vs no spec (1777 vs 2153 at 32 GPUs). The comparison is at one load per scale; a
        load sweep at which disagg is not overloaded (where ours' advantage should shrink) is the next run.
      - [ ] **Load sweep at 32 H20s with both systems on FA3 (2026-10-21; `bash tasks/pd_colocation/
        mn_load_sweep32_fa3_20261021.sh`; sglang 6d3f344af, pods' flashinfer_backend.py md5 ed98c8e1fd51; Qwen2.5-7B no
        spec, 900 s; 1x and 2x the 8-GPU load).** H20 is Hopper, where upstream's default backend is FA3; every earlier
        disagg arm forced flashinfer, so disagg now runs FA3 and ours runs its split rows on FA3 (`SPLIT_ROWS_FA3=1`).
        1x load (agent programs 32 / tasks 16, chat 8 req/s), all errors=0, 0 tracebacks:
        | arm | agent tok/s | agent TTFT mean / p99 | chat TTFT mean / p99 | ITL p99 agent / chat | cached |
        |---|---|---|---|---|---|
        | ours (EDF + SLO router + pool) | 596.3 | 0.74 / 2.72 s | 0.05 / 0.31 s | 19.3 / 18.7 ms | 0.981 |
        | disagg 5 + 3, FA3 | 382.5 | 4.59 / 28.60 s | 2.98 / 28.25 s | 14.3 / 10.2 ms | 0.851 |
        | disagg 4 + 4, FA3 | 324.5 | 5.79 / 35.31 s | 3.55 / 34.06 s | 12.9 / 9.7 ms | 0.805 |
        Even at 1x load, where disagg is not collapsed, ours leads on agent throughput (+56%), agent TTFT (-84%) and chat
        TTFT (-98%); disagg holds the lower ITL tail (its decode pods never see a prefill).
        2x load (agent programs 64 / tasks 32, chat 16 req/s), all errors=0, 0 tracebacks:
        | arm | agent tok/s | agent TTFT mean / p99 | chat TTFT mean / p99 | ITL p99 agent / chat | cached |
        |---|---|---|---|---|---|
        | ours | 1172.2 | 0.72 / 3.11 s | 0.06 / 0.37 s | 23.6 / 25.4 ms | 0.979 |
        | disagg 5 + 3, FA3 | 443.7 | 12.25 / 56.52 s | 9.68 / 55.22 s | 15.8 / 12.1 ms | 0.757 |
        | disagg 4 + 4, FA3 | 393.2 | 14.66 / 60.46 s | 11.92 / 59.44 s | 13.0 / 10.1 ms | 0.763 |
        Ours doubles its agent throughput from 1x to 2x (596 -> 1172) at flat TTFT; disagg grows 16-21% and its TTFT
        triples -- its prefill pods saturate re-prefilling agent history (cached fraction 0.85 -> 0.76). The gap widens
        with load: agent throughput +56% at 1x, +164% at 2x. Ours' ITL tail stays 1.5-2.5x disagg's (19-25 vs 10-16
        ms). Per-tenant attainment and error bars still to add.
      - [ ] **Point 2 measured as ThunderAgent poses it: cache THRASHING (2026-10-22, 16 H20s = 4 pods x 4, both
        systems on FA3, `bash tasks/pd_colocation/mn_thrash_16g_20261022.sh`; agent program concurrency fixed at 32,
        the WORKING SET swept by distinct agent tasks at 64K prefix each: 16 / 64 / 128 tasks = ~1M / 4M / 8M live KV
        tokens against ~6.7M of cluster capacity, so only the last point must evict and re-prefill).** The claim is
        that colocation plus the shared pool keeps prefixes resident where disagg's prefill pods must recompute them,
        so the cached fraction is the direct re-prefill proxy. First point (16 tasks, no thrashing -- the baseline):
        | arm | agent tok/s | agent TTFT mean / p99 | chat TTFT mean / p99 | ITL p99 agent / chat | cached |
        |---|---|---|---|---|---|
        | ours | 597.5 | 0.60 / 2.40 s | 0.06 / 0.34 s | 21.9 / 19.9 ms | 0.980 |
        | disagg 2 + 2, FA3 | 313.1 | 7.03 / 44.72 s | 6.50 / 48.58 s | 17.2 / 12.1 ms | 0.890 |
        Ours +91% agent throughput and 12x lower agent TTFT before any thrashing, with disagg already 9 points behind
        on cached fraction. **Second point (64 tasks, ~4M live KV against ~6.7M capacity) is the first thrashing
        evidence and it supports point 2 as posed:**
        | metric | 16 tasks | 64 tasks | change |
        |---|---|---|---|
        | ours agent tok/s | 597.5 | 570.7 | -4.5% |
        | ours cached fraction | 0.980 | 0.966 | -1.4 pts |
        | ours agent TTFT mean / p99 | 0.60 / 2.40 s | 0.62 / 3.66 s | flat |
        | disagg 2+2 agent tok/s | 313.1 | 155.9 | **-50%** |
        | disagg cached fraction | 0.890 | 0.681 | **-21 pts** |
        | disagg agent TTFT mean / p99 | 7.03 / 44.7 s | 20.26 / 98.9 s | **3x** |
        | disagg chat TTFT mean | 6.50 s | 15.66 s | 2.4x |
        Ours' lead widens from +91% to **+266%** agent throughput as the working set grows 4x, and the mechanism is
        legible in the cached fraction: disagg's prefill pods lose prefixes between an agent session's rounds and
        recompute them, where colocation plus the pool keeps them resident. **COMPLETE CURVE (all 6 arms, errors=0,
        0 tracebacks):**
        | tasks (live KV) | ours tok/s | ours cached | ours agent TTFT | disagg tok/s | disagg cached | disagg agent TTFT | ratio |
        |---|---|---|---|---|---|---|---|
        | 16 (~1M) | 597.5 | 0.980 | 0.60 s | 313.1 | 0.890 | 7.03 s | +91% |
        | 64 (~4M) | 570.7 | 0.966 | 0.62 s | 155.9 | 0.681 | 20.26 s | **+266%** |
        | 128 (~8M, past the ~6.7M pool) | 534.5 | 0.949 | 0.68 s | 141.8 | 0.605 | 22.58 s | **+277%** |
        Across an 8x working-set growth ours loses **10.6% throughput and 3.1 points of cached fraction**; disagg loses
        **55% and 28.5 points**, with agent TTFT 3.2x and chat TTFT 2.7x worse (76 ms vs 17.8 s at the last point, a
        234x gap). The degradation is monotone in both systems and the cached fraction tracks it, so the mechanism is
        not inferred: disagg must recompute an agent session's prefix on whichever prefill pod the round lands on,
        while colocation plus the cluster pool keeps it resident or pulls it over RDMA. **This is point 2's claim,
        measured as ThunderAgent poses it, and it holds.** Disagg keeps the better ITL tail throughout (11.9-17.2 vs
        21.9-113.2 ms p99), which is the standing trade.

      - [ ] **Capacity at SLO, 16 GPUs (8 pods x 2), phase A: disagg at every P:D split, KV pool on, FA3 (2026-10-23,
        `bash tasks/pd_colocation/mn_capacity_16g_20261023.sh`; 32 agent programs + chat 8 req/s, Qwen2.5 no spec):**
        | prefill pods of 8 | chat attainment | agent SLO goodput | agent tok/s | agent TTFT | chat TPOT p90 | cached |
        |---|---:|---:|---:|---:|---:|---:|
        | 1 | 16.2% | 86 | 194.9 | 15.18 s | 6.2 ms | 0.940 |
        | 2 | 37.3% | 201 | 293.6 | 7.93 s | 7.2 ms | 0.933 |
        | 3 | 51.5% | 230 | 316.1 | 6.78 s | 7.9 ms | 0.924 |
        | 4 | 60.4% | 296 | 366.3 | 4.84 s | 8.5 ms | 0.926 |
        | 5 | 66.1% | 320 | 386.6 | 4.31 s | 9.8 ms | 0.922 |
        | **6** | **67.8%** | **359** | **413.1** | 3.47 s | 13.0 ms | 0.922 |
        | 7 | 50.0% | 326 | 369.7 | 3.23 s | 31.1 ms | 0.918 |
        **A properly tuned disagg is far stronger than the baseline every earlier multi-node table used**: its best
        split (6 of 8) attains 67.8% chat, against 23.3% at the pod-granular best of the old 4 x 4 sweep, and with the
        KV pool its cached fraction is 0.92-0.94, not 0.76. The old 7-44x ratios came largely from the weak baseline.
        Ours at this load is being rerun: its phase-A arm aborted before starting because the cn-seed WebShell SSO
        session expired mid-sweep ("mooncake master did not start: webshell relay redirected to SSO"), not because
        of the arm (`mn_cap16_ours_L32_rerun_20261023.sh`).
      - [x] **32-GPU capacity (2026-10-24, `mn_capacity_32g_20261023.sh`, 8 pods x 4 H20, all on FA3, kernel off,
        Qwen2.5-7B no spec).** Chat attainment (0.5 s + 20 ms) / agent SLO goodput (tok/s):
        | agent programs + chat rate | ours | disagg 6 + 2 pods | disagg 7 + 1 pods |
        |---|---|---|---|
        | 32 + 8 req/s | **99.6%** / 569 | 83.7% / 430 | 82.4% / 409 |
        | 64 + 16 req/s | **99.1%** / 1096 | 55.9% / 342 | 48.7% / 352 |
        | 96 + 24 req/s | **98.2%** / 1546 | 37.2% / 227 | 22.6% / 205 |
        | 128 + 32 req/s | **96.3%** / 1910 | 26.7% / 146 | 15.9% / 111 |
        Phase A at 64 programs: disagg with 1..7 prefill pods reached 10.5 / 20.6 / 31.5 / 44.2 / 48.9 / 55.9 / 48.7%.
        Best-tuned disagg reaches 90% at none of the tested loads, while ours holds >= 96% through 128 programs; ours'
        agent SLO goodput grows with load (569 -> 1910) while disagg's falls after 32 programs.
        **Caveat before quoting any ratio.** The disagg prefill pods ran clean upstream FCFS, so chat prefills queue
        behind 64K-token agent prefills. Part of this gap may therefore be scheduling policy rather than
        disaggregation. That was tested at 16 GPUs (next item): it is not the cause.
      - [x] **Disagg + EDF on the prefill pods (2026-10-24, `mn_cap16_disagg_edf_20261024.sh`, 16 GPUs, 6 + 2 pods).**
        Our tree on every disagg worker with `apply_edf_prefill_role_20261024.py` (deadline order on the prefill
        queue; `EDF_PREFILL_ROLE_QUEUE` logged on every arm; errors=0), otherwise identical to phase B:
        | agent programs + chat rate | chat attainment, FCFS -> EDF | agent SLO goodput, FCFS -> EDF | cached prefix fraction (EDF) |
        |---|---|---|---|
        | 16 + 4 | 92.8% -> 92.5% | 282 -> 283 | 0.966 |
        | 32 + 8 | 67.8% -> 70.7% | 359 -> 298 | 0.836 |
        | 48 + 12 | 45.8% -> 46.7% | 231 -> 226 | 0.788 |
        | 64 + 16 | 31.2% -> 30.5% | 165 -> 163 | 0.814 |
        **Deadline order does not move disagg.** Its limit is the prefill -> decode KV transfer, not the prefill queue.
        At 64 programs the decode workers hold 52-55 requests in `#transfer-req` (mean; max 108) while running only
        ~7 each at 63-68% KV usage, and the prefill workers' queues stay short (mean 3.5-6). Every agent round
        re-ships its whole ~64K-token prefix (~3.7 GB of KV for Qwen2.5-7B) from a prefill pod to a decode pod. So
        the multi-node gap against STOCK sglang PD is the per-round re-ship (point 5). NVIDIA Dynamo's conditional
        disaggregation avoids most of it by prefilling short appends locally on the decode worker. **That is the
        baseline the capacity claim has to beat**:
      - [ ] **Conditional disaggregation (Dynamo's policy) at 16 GPUs (2026-10-24, `mn_cap16_cond_disagg_20261024.sh`).**
        `router.py --policy cond_disagg` on the same colocated workers as ours (EDF, chunk preemption, FA3, pool),
        so only placement differs:
        - workers on the first P pods take remote prefills of prompts whose effective uncached length on the chosen
          decode worker exceeds 4096 tokens;
        - the KV reaches the decode worker through the cluster pool (write-through);
        - short appends prefill locally.
        Validity per arm: `cond_disagg_check.py` reports the remote share and the decode-side cached fraction.
        - [ ] 2026-10-24 **Re-run at Dynamo's defaults (user, 2026-10-24):** remote prefill when the effective uncached
          length exceeds 2048 tokens AND 0.7 of the prompt, not 4096 alone. The 4096 numbers below stay as the earlier
          setting.
          - [x] 2026-10-25 `router.py --cond-disagg-min-frac` (`COND_FRAC` in mn_run.py). Unit test: a 3000-token
                append to a cached 20K session goes remote at 2048 alone and stays local at 2048 + 0.7
                (`test_router_cond_disagg.py`, 3 cases pass).
          - [ ] Run inside stage 3 on the A6 workers (`mn_stage3_a6_16g_20261025.sh`, running).
        Router unit test: `test_router_cond_disagg.py`. Phase A sweeps P = 2/4/6; phase B runs the best P at
        16/32/48 programs.
        **First arm, P = 2 at 64 programs + chat 16 (errors=0).** Chat attainment 94.6% (ours 94.2%, stock PD best
        31.2%). Agent SLO goodput 845 tok/s (ours 964, stock PD 165). Agent tok/s 863 (ours 991).
        Only 0.5% of requests (110 / 20546) were prefilled remotely; 6050 long prompts were served locally from the
        decode worker's own cache or the pool. Those 110 got 79% of their prefix from the pool (22 under 50%).
        **Reading: conditional disagg captures nearly all of ours' gain over stock PD.** In this workload it is
        effectively colocation on the decode pods. Ours' remaining edge at this load is agent SLO goodput (+14%,
        using all 16 workers rather than 12 that take decode). The claim against it has to come from the knee
        (96 / 128 programs, `mn_cap16_cond_high_20261024.sh`, queued) and from the axes that do not depend on
        transfer: co-executing prefill with spec decode (4-mode + fused kernel), cache-hit routing and
        replication across symmetric nodes, and SLO admission.
        **Full sweep (2026-10-24, all errors=0).** Phase A at 64 programs: P = 2 / 4 / 6 prefill pods reached 94.6 /
        81.7 / 12.7% chat attainment, so P = 2 is best. Chat attainment / agent SLO goodput (tok/s):
        | agent programs + chat rate | ours | conditional disagg, 2 + 6 pods | stock PD, 6 + 2 pods |
        |---|---|---|---|
        | 16 + 4 | 99.5% / 293 | 99.6% / 275 | 92.8% / 282 |
        | 32 + 8 | 99.0% / 568 | 98.9% / 518 | 67.8% / 359 |
        | 48 + 12 | 96.6% / 772 | 96.0% / 635 | 45.8% / 231 |
        | 64 + 16 | 94.2% / 964 | 94.6% / 845 | 31.2% / 165 |
        | 96 + 24 | 89.2% / 1187 | 84.4% / 906 | -- |
        | 128 + 32 | 72.6% / 1221 | 60.9% / 890 | -- |
        Validity: remote prefills were 0.4-0.7% of requests at every load. Their decode-side cached fraction fell from
        0.92 to 0.49 as load grew (pool pulls time out), which is too few requests to move the result.
        **Against the strongest PD baseline:**
        - Chat attainment is equal through 64 programs. Ours leads at the knee: +4.8 points at 96, +11.7 at 128.
        - Ours' agent SLO goodput is +6% at 16, +10% at 32, +22% at 48, +14% at 64, +31% at 96 and +37% at 128.
        - Interpolated load at 90% chat attainment: ours ~91 programs vs ~78 for conditional disagg, i.e. **~17% more
          load at SLO**, with +31% agent SLO goodput at the knee.
        - Mechanism: conditional disagg dedicates 4 of 16 workers to remote prefills that are 0.5% of the work.
        - Open: a 1-prefill-pod split (`mn_cap16_cond_p1_20261024.sh`, running) is needed before this counts as
          best-tuned.
      - [x] **Phase B: CAPACITY AT SLO, ours vs the two best-tuned disagg splits (2026-10-24,
        `mn_capacity_16g_phaseB_20261023.sh`, splits 6 and 7 picked automatically from phase A; all errors=0).**
        chat attainment (0.5 s + 20 ms) / agent SLO goodput (tok/s) / agent tok/s / agent TTFT:
        | agent programs + chat rate | ours | disagg 6 + 2 | disagg 7 + 1 |
        |---|---|---|---|
        | 16 + 4 req/s | **99.5%** / 293 / 295.8 / 0.57 s | 92.8% / 282 / 285.7 / 1.00 s | 88.3% / 266 / 269.5 / 0.96 s |
        | 32 + 8 req/s | rerun pending | 67.8% / 359 / 413.1 / 3.47 s | 50.0% / 326 / 369.7 / 3.23 s |
        | 48 + 12 req/s | **96.6%** / 772 / 787.2 / 0.65 s | 45.8% / 231 / 371.1 / 9.91 s | 19.1% / 234 / 361.3 / 8.96 s |
        | 64 + 16 req/s | **94.2%** / 964 / 991.2 / 0.74 s | 31.2% / 165 / 361.5 / 15.69 s | 4.4% / 63 / 253.9 / 23.90 s |
        | 96 + 24 req/s | 89.2% / 1187 | -- | -- |
        | 128 + 32 req/s | 72.6% / 1221 | -- | -- |
        **The fair multi-node result.** The best-tuned disagg holds >= 90% chat attainment only at the lightest load
        (16 programs); ours still holds 94.2% at 64. So **ours sustains at least 2-4x the load of the best-tuned PD
        disaggregation at SLO**. Ours' own capacity point (2026-10-24, `mn_cap16_ours_high_20261024.sh`): 89.2% at 96
        programs and 72.6% at 128, so ours' capacity lies between 64 and 96 programs and disagg's between 16 and 48:
        **4x at the tested grid (64 vs 16)**. The grid is coarse on disagg's side; a 24 / 32-program disagg point would
        tighten the lower end. At equal light load the systems are close (99.5% vs 92.8%, agent
        SLO goodput 293 vs 282), as a fair comparison should be; the gap is a saturation effect -- disagg's agent
        throughput plateaus at ~360-410 tok/s from 32 programs on while ours scales with load to 991.
        Standing trade: ours' agent ITL p99 is 109-134 ms at 48-64 programs vs disagg's 29-31 ms (its decode pods
        never see a prefill); chat TPOT p90 ours 9.7-12.9 ms vs disagg 13.5-14.3 ms.

      - [ ] **Qwen3-8B + trained EAGLE3 head (AngelSlim) with MTP on 16 H20s, both on FA3 (2026-10-21;
        `bash tasks/pd_colocation/mn_qwen3_16g_mtp_20261021.sh`; sglang fc9847d2a, pods' flashinfer_backend.py md5
        ed98c8e1fd51; ours rows + verify on FA3, disagg `--attention-backend fa3`; router perf model calibrated on
        ours' worker config, `/tmp/h20/mn_q3_calibrate.log`; chat template; 16 tasks / 32 agent programs, chat 8 req/s,
        900 s -- half the Qwen2.5-7B 16-GPU load). All errors=0, 0 tracebacks:**
        | arm | agent tok/s | agent TTFT mean / p99 | chat TTFT mean / p99 | agent ITL p99 | chat TPOT | cached |
        |---|---|---|---|---|---|---|
        | ours | 447.8 | 1.67 / 21.50 s | 0.21 / 2.46 s | 339.6 ms | 11.1 ms | 0.930 |
        | disagg 2 + 2, FA3 | 105.0 | 30.74 / 109.70 s | 24.79 / 102.16 s | 30.3 ms | 3.6 ms | 0.641 |
        | disagg 3 + 1, FA3 | 128.4 | 23.56 / 85.94 s | 16.97 / 76.71 s | 58.3 ms | 7.7 ms | 0.652 |
        Ours 3.5x disagg's better ratio in agent throughput, TTFT -93% / -99%; disagg's prefill pods re-prefill agent
        history (cached 0.64-0.65 vs 0.93) and queue. Ours' agent ITL tail is 6-11x disagg's (340 ms p99) -- the cost of
        colocating deep chunks with decode on H20; to examine. One repeat; per-tenant attainment to add.
      - [x] **Llama-3.3-70B-Instruct at TP4, 4 pods x 4 H20 = 4 replicas (2026-10-21, job f3b4627e0864b06f, provisioned by
        `mn_provision_70b_20261021.sh` from the archived serving tree -- manifest verified on all 4 pods -- plus
        `mn_provision_70b_post_20261021.sh`).**
        - [x] **Output gate (`gate_70b_20261021.sh`, one TP4 pod, trained EAGLE3 head; 96 ShareGPT + 8 x 32K TraceLab
          prompts at once, greedy, 128 tokens; all arms 0 tracebacks; ours 13K split calls, 151 4-mode steps): PASS.**
          vs stock MTP, short / long prompts exact and mean common prefix: stock rerun (noise floor) 42/96, 355 of 564
          chars / 1/8, 111 of 446; ours 41/96, 357 / 2/8, 177. The trained head accepts only 1.29-1.39 tokens per verify
          on these raw-text prompts (the Qwen-7B head: ~2.1), so the serving A/B fixes acceptance at 2.1 for every arm.
        - [x] **Ours vs PD disagg 2P2D (2 prefill + 2 decode TP4 pods), MTP everywhere at acceptance 2.1, text fan-out
          (16 x 64K prefixes) at AGENT_CONC 8 + chat 2 req/s, 900 s, two repeats (`bash
          tasks/pd_colocation/mn_70b_16gpu_20261021.sh`; router calibrated on one 70B TP4 worker -- cold prefill ~2.2K
          tok/s, 49K tokens in 25 s; ours 20-30K split calls, 0 tracebacks; disagg's 3 tracebacks are aborted health-check
          requests; sglang b28082f80).**
          | arm | agent tput / TTFT / ITL | prefix cached | chat TTFT mean / p50, TPOT | chat att. 0.5 s+20 ms / 2 s+50 ms | agent goodput 10 s+80 ms |
          |---|---|---|---|---|---|
          | ours rep 1 | 68.6 / 2.54 s / 114 ms | 0.91 | 3.40 s / 0.19 s, 40.7 ms | 55.5 / 76.7% | 59 tok/s |
          | ours rep 2 | 76.4 / 2.47 s / 96 ms | 0.92 | 1.18 s / 0.15 s, 28.7 ms | 69.5 / 89.1% | 69 tok/s |
          | PD disagg rep 1 | 20.0 / 43.0 s / 27 ms | 0.76 | 40.5 s / 37.2 s, 11.9 ms | 10.7 / 16.0% | 4 tok/s |
          | PD disagg rep 2 | 21.3 / 41.2 s / 27 ms | 0.78 | 39.0 s / 36.4 s, 10.8 ms | 12.8 / 18.2% | 5 tok/s |
          **At 70B the result holds: chat attainment 5x (2 s + 50 ms: 77-89% vs 16-18%), agent goodput ~13x, agent
          throughput 3.5x, agent TTFT -94%, chat TTFT -94..-97%.** Disaggregation again keeps the faster per-token decode
          (11 vs 29-41 ms) but its two prefill pods queue every request (TTFT ~40 s). The workers needed `libssl-dev`
          (HiCache's hash extension) and a bf16 draft config (`mn_provision_70b_post_20261021.sh`).
      - [ ] 2026-10-24 **Fault tolerance, both failure roles, full stack, 16 GPUs (user, 2026-10-24).** Kill one
        and then two workers mid-run, (a) PREFILL-role and (b) DECODE-role in the disagg baseline, with the same
        workers killed for ours (whose nodes are symmetric). Same workload and load as the stage-3 capacity runs.
        Report per-tenant TTFT / ITL SLO attainment and throughput before, during and after, in 30 s windows, plus
        lost requests and time to recover.
        Hypotheses to test, not to assume:
        - **prefill down:** ours wins, because every node can prefill, and the SLO router moves new prefills to the
          survivors while disagg's remaining prefill pods queue;
        - **decode down:** disagg keeps admitting at the same rate and loses ITL on its remaining decode pods, while
          ours holds ITL by admitting fewer prefills (the SLO check plus EDF), trading some TTFT attainment.
        Baselines: stock sglang PD (health-check ejection on, `DISAGG_HEALTH=1`) at its best P:D split.
      - [x] **Fault tolerance: lose 2 of 8 GPUs mid-run (2026-10-20; same text fan-out C=32 + chat 8 req/s; at t = 300 s
        every worker on pod 1 is killed, `FAULT_AT` / `FAULT_POD` in mn_run.py -- under PD disagg pod 1 is one of the
        two prefill pods, under ours two colocated workers; our router marks a worker dead on a refused or broken
        connection and reroutes (`test_router_dead_worker.py` PASS), the stock PD router ejects it by health check
        (`DISAGG_HEALTH=1`: every 10 s, 3 failures); `bash tasks/pd_colocation/mn_fault_8gpu_20261020.sh`, windows by
        `spike_windows.py --spike 300:600`; sglang 58e562574).** Two repeats:
        | arm | agent att. before / after the fault | agent goodput before / after | chat att. (0.5 s + 20 ms) | lost requests |
        |---|---|---|---|---|
        | ours rep 1 | 96.3 / 97.9% | 506 / 480 tok/s | 84.8% | 1 agent + 4 chats (in flight on pod 1) |
        | ours rep 2 | 96.4 / 98.3% | 500 / 506 tok/s | 85.1% | 6 chats |
        | PD disagg rep 1 | 12.9 / 34.9% | 14 / 50 tok/s | 9.0% | 30 chats |
        | PD disagg rep 2 | 23.1 / 33.0% | 24 / 49 tok/s | 14.9% | 1 agent + 2 chats |
        Without a fault (same load): ours chat 91.4% / agent goodput 551 tok/s, disagg 26.2% / 92 tok/s. **Losing a
        quarter of the GPUs costs ours ~6 chat-attainment points and nothing on agents** (both workers marked dead
        within 10 ms of each other on their broken streams; their in-flight requests are lost, every later one is
        rerouted to the six survivors, which already hold replicas of the hot prefixes); **disagg loses half its
        prefill capacity and its chat attainment falls from 26% to 9-15%.** Its agent attainment rises after the fault
        only because its pre-fault window holds the start-up burst it is already overloaded by -- at this load it has
        no SLO headroom to lose, so the cleaner contrast is the chat number.
      - [x] **Second model: Qwen2.5-14B-Instruct (2026-10-20; 40 q / 8 kv heads, TP1 per H20, no MTP; same text fan-out
        C=32 + chat 8 req/s on 8 H20s; router SLO model recalibrated on 14B -- cold prefill ~2.3K tok/s at long context,
        own decode grid `decode_grid_qwen14b.json`; `mn_14b_8gpu_20261020.sh`; ours 0 tracebacks / 0 fallbacks).**
        | arm | agent tput / TTFT / ITL | chat TTFT / ITL | chat attainment 0.5 s+20 ms / 2 s+50 ms | agent goodput 10 s+80 ms |
        |---|---|---|---|---|
        | ours rep 1 | 216.0 / 2.88 s / 85.8 ms | 2.95 s / 44.2 ms | 45.3 / 70.3% | 131 tok/s |
        | ours rep 2 | 185.7 / 3.70 s / 100.3 ms | 2.11 s / 44.9 ms | 51.4 / 73.0% | 102 tok/s |
        | PD disagg rep 1 | 6.5 / 289.5 s / 16.9 ms | 199.2 s / 12.5 ms | 11.3 / 11.5% | 0 |
        | PD disagg rep 2 | -- | -- (6857 / 7200 chats completed) | 13.2 / 13.2% | 0 |
        **The 14B result widens the gap: disagg's two prefill nodes cannot keep up with 14B prefill on 64K prefixes
        (agent TTFT ~290 s, chat ~200 s, chat output below the offered load, run took 2414 s of wall time for a 900 s
        workload), so no agent request meets its SLO.** Ours serves the load but 14B on H20 is overloaded at a 7B-sized
        offered load: chat TPOT p50 18-21 ms, so tight-SLO chat attainment falls to 45-51% (from ~90% on 7B). A load
        scaled to 14B's capacity is the fair next point. sglang 642e476b7.
        - [x] **Half load, scaled to 14B's capacity (2026-10-20; agent C=16, chat 4 req/s, 3600 chats; same calibration;
          `bash tasks/pd_colocation/mn_14b_half_8gpu_20261020.sh`, scored by `fanout_per_tenant.py`; sglang 7e7b390b7;
          ours 0 errors).**
          | arm | agent tput / TTFT mean / p50 / ITL | chat TTFT mean / p50, TPOT | chat attainment 0.5 s+20 ms / 2 s+50 ms | agent goodput 10 s+80 ms |
          |---|---|---|---|---|
          | ours rep 1 | 157.3 / 2.16 s / 0.95 s / 45.2 ms | 0.93 s / 75 ms, 24.1 ms | 80.1 / 92.9% | 143 tok/s |
          | ours rep 2 | 152.8 / 2.29 s / 0.96 s / 47.5 ms | 0.80 s / 75 ms, 24.5 ms | 80.4 / 92.8% | 133 tok/s |
          | PD disagg rep 1 | 16.1 / 110 s / 105 s / 12.4 ms | 83.0 s / 75.0 s, 10.3 ms | 9.4 / 10.6% | 1 |
          | PD disagg rep 2 | 15.2 / 118 s / 119 s / 12.7 ms | 75.9 s / 70.1 s, 11.9 ms | 16.9 / 18.7% | 0 |
          **At a load 14B can carry, ours meets chat SLOs for 80% of requests (vs 9-17%) and serves ~10x the agent
          throughput with ~140 tok/s agent goodput vs ~0. Halving the load does not rescue disagg: its two prefill nodes
          still queue 64K prefixes (agent TTFT 110-118 s, prefix cached fraction 0.44-0.50 vs ours 0.94).**
        - [x] **With MTP, simulated acceptance (2026-10-20; same half load; draft = random-weight EAGLE3 head of 14B's shape,
          `make_sim_eagle3_head.py`, acceptance fixed at 2.1 tokens per verify via `SGLANG_SIMULATE_ACC_LEN` -- logged
          accept len 2.05-2.13 -- so draft and verify work per step are real; `bash
          tasks/pd_colocation/mn_14b_mtp_8gpu_20261020.sh`; sglang 0e54fc671; ours 0 errors).**
          | arm | agent tput / TTFT / ITL | chat TTFT / TPOT | chat attainment 0.5 s+20 ms / 2 s+50 ms | agent goodput 10 s+80 ms |
          |---|---|---|---|---|
          | ours + MTP rep 1 | 116.5 / 3.45 s / 144 ms | 0.56 s / 29.4 ms | 76.8 / 88.6% | 88 tok/s |
          | ours + MTP rep 2 | 121.3 / 3.18 s / 141 ms | 0.80 s / 42.3 ms | 73.1 / 84.1% | 94 tok/s |
          | PD disagg + MTP rep 1 | 15.0 / 108 s / 84 ms | 90.6 s / 21.5 ms | 0.1 / 0.2% | 0 |
          | PD disagg + MTP rep 2 | 15.5 / 106 s / 162 ms | 91.4 s / 45.7 ms | 1.1 / 1.4% | 0 |
          **With MTP disaggregation still collapses (TTFT ~107 s, no agent request within SLO, chat attainment ~1%);
          ours keeps chat attainment at 73-77% and serves ~8x the agent throughput.** At 2.1 accepted tokens per verify
          MTP does not pay for 14B on H20 for either system (ours 117-121 vs 153-157 tok/s without it, agent ITL 141-144
          vs 45-48 ms): the verify forward at 14B is too costly for that acceptance.
      - [x] **2026-10-19 per-tenant scoring: chat on SLO attainment (TTFT + per-request TPOT), agents on SLO goodput.**
        Same saved records, re-scored by `tasks/perf_model/fanout_per_tenant.py` (per-request TPOT = sum(itls) /
        (output_len - 1), so MTP arms count tokens, not stream chunks). Chat attainment at 0.5 s + 20 ms / 1 s + 30 ms /
        2 s + 50 ms, agent goodput at 10 s + 80 ms / 20 s + 30 ms:
        | arm (8 H20, text fan-out) | chat attainment | chat TPOT p50 / p90 ms | agent goodput tok/s |
        |---|---|---|---|
        | heavy, ours no MTP (ITL SLO auto 2.5) | **91.3 / 94.3 / 95.9%** | 7.0 / 13.4 | **546 / 544** |
        | heavy, ours no MTP (ITL SLO 20 ms) | 91.4 / 93.9 / 95.8% | 6.9 / 13.1 | 553 / 552 |
        | heavy, ours + MTP | 91.0 / 94.5 / 96.3% | 4.9 / 15.9 | 429 / 288 |
        | heavy, disagg 2P2D no spec (2 reps) | 24.5-25.2 / 27.5-28.6 / 31.9-32.7% | 5.3-5.5 / 7.5-9.0 | 79-86 / 100-111 |
        | heavy, disagg 2P2D + MTP | 27.6 / 29.2 / 30.6% | 8.9 / 176 | 15 / 15 |
        | light, ours no MTP (auto 2.5) | **96.8 / 97.8 / 98.6%** | 5.2 / 7.9 | **291 / 291** |
        | light, ours + MTP | 98.3 / 99.1 / 99.5% | 4.1 / 6.9 | 240 / 202 |
        | light, disagg 2P2D no spec (3 reps) | 43.1-44.6 / 47.0-48.6 / 51.3-53.7% | 5.4-5.5 / 7.2-7.5 | 81-85 / 96-102 |
        | light, disagg 2P2D + MTP | 39.7 / 44.7 / 50.2% | 5.5 / 17.8 | 63 / 60 |
        Chat's mean ITL is 1.3-1.9x disagg's, but at the per-request level ours keeps chat TPOT under 20 ms for >= 90% of
        requests; disagg's chat requests miss on TTFT (they queue behind long fan-out prefills on its two prefill pods).
        Per tenant, ours meets chat SLOs **3.0-3.7x as often (heavy) and 1.8-2.2x (light)** and delivers 3.0-7.1x the
        agent goodput. Whether the bar is stated per tenant is the user's decision; this records the numbers. sglang
        `e4cfed359`; runs as in the rows above.
      - [x] **Config correction (2026-10-19): every "ours, no MTP" arm above (`coloc_slo_edf_hc`) ran without its router
        arguments.** `mn_fanout_8gpu_20261018.sh` attached the calibrated SLO model, pool size and transfer term only to
        the exact arm name `coloc_slo_edf_mtp_hc`, so the no-MTP arms routed with `router.py`'s default calibration and
        no transfer term (pool still on at the workers). Their numbers stand as measured for that configuration, not for
        the full policy; the MTP arms are unaffected. Fixed (`coloc_slo_*` match); the no-MTP rows are re-run below.
      - [x] **Router ITL SLO with the full router arguments, heavy natural-text load, ours without MTP (2026-10-19;
        clean; `bash /tmp/h20/fanout_itlslo.sh` = `FO_TEXT=1 EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms {15,10}"
        REP=_text_itl{15,10} ARMS=coloc_slo_edf_hc bash tasks/pd_colocation/mn_fanout_8gpu_20261018.sh`).** The default
        ITL SLO (100 ms) never moved a request at ~20 ms TPOT. ITL SLO 15 / 10 ms: agent 505 / 501 tok/s, TTFT 0.92 /
        0.88 s, TPOT 16.0 / 17.2 ms; chat TTFT 319 / 332 ms, ITL 10.3 / 10.2 ms. Agent SLO goodput at 10 s + 80 ms:
        495 / 485 tok/s (97.9 / 96.8%); at 20 s + 30 ms: 489 / 480 tok/s (96.7 / 95.8%) -- **5.8x and 4.4x disagg's**
        (86 / 111 tok/s). Vs the earlier no-MTP arm (default router arguments): agent +8% / -19% / TPOT -22%, chat
        TTFT and ITL worse (135 -> 319 ms, 7.5 -> 10.3 ms). Agent TPOT stays 3x disagg's 5.2 ms (decode isolation);
        chat ITL 2x disagg's 5.0 ms.
      - [x] **Repeat 2, heavy natural-text load (2026-10-19; clean; `bash /tmp/h20/fanout_rep2.sh`).** Disagg (no spec):
        agent 170 tok/s / 17.90 s / TPOT 5.7 ms, chat 19.5 s / 5.4 ms; ours (ITL SLO 15 ms, full router arguments): 503
        tok/s / 0.89 s / 16.9 ms, chat 312 ms / 10.3 ms. Agent SLO goodput at 10 s + 80 ms / 20 s + 30 ms: ours 490 /
        483 tok/s (97.3 / 95.9%), disagg 79 / 100 (46.3 / 58.9%). **Two repeats: ours delivers 4.4-6.2x disagg's agent
        goodput**, agent throughput 2.8-3.0x, agent TTFT -95%, chat TTFT -98%; agent TPOT 3x and chat ITL 2x disagg's.
      - [x] **Re-prefill, the falsifier named above (`tasks/perf_model/fanout_reprefill.py`: sum of input - cached tokens
        per request, from the saved records).** Heavy natural-text load: ours (no MTP) recomputes 8.6-8.7M of 246-249M
        input tokens (3.4-3.5%; 2.3% with default router arguments), ours + MTP 5.3M of 213M (2.5%); disagg (no spec)
        8.7-8.8M of 84-87M (10.2-10.4%), disagg + MTP 5.2M of 31M (16.7%). Disagg recomputes ~3x the share of what it
        serves: about the same absolute prefill work as ours, turned into a third of the served tokens. "Colocation
        replicates better" passes this test at equal memory.
      - [x] **Light-load repeat with request tagging (2026-10-19; driver tags rid `fo-t<task>-...`, router logs
        `client_rid`; clean; `bash /tmp/h20/fanout_lightrep.sh`).** Disagg (no spec): agent 124 tok/s / 10.62 s / TPOT
        6.10 ms (p50 6.61), chat 9.59 s / 5.4 ms; ours (ITL SLO 15 ms, full router arguments, no MTP): 294 tok/s / 0.69
        s / 9.18 ms (p50 **6.49**), chat 82 ms / 7.5 ms. Agent SLO goodput 10 s + 80 ms / 20 s + 30 ms: ours 288 / 288
        tok/s (97.7 / 97.9%), disagg 81 / 96 (65.6 / 77.6%) -- **3.0-3.5x**. With the ITL SLO the median agent TPOT
        drops below disagg's; the mean stays 1.5x (a tail of slow tokens).
      - [x] **Replica count per hot prefix (`tasks/perf_model/fanout_replicas.py`, that run).** Over 900 s each task's
        64K prefix was served by 3-7 of the 8 workers; per 180 s popularity epoch the currently hot tasks sit on the
        most workers (epoch 0: task 0 on 7; epoch 4: tasks 4 and 7 on 6 each) while cold ones fall to 1-2 -- replicas
        follow demand. Only 76 of 2106 routings put a task on a worker that had not served it (each leaves a new
        copy, pulled from the pool or recomputed). Disagg holds prefixes on its 2 prefill pods only.
      - [x] **Heavy load, repeat 3 with tagging (2026-10-19; clean; `bash /tmp/h20/fanout_heavyrep3.sh`):** ours 496 tok/s /
        0.98 s / TPOT 17.8 ms, chat 342 ms / 10.9 ms; goodput 484 / 474 tok/s (97.5 / 95.5%) -- 4.3-6.1x the two disagg
        repeats (79-111 tok/s). Three repeats of ours: 496-505 tok/s, TTFT 0.89-0.98 s, TPOT 16.0-17.8 ms. Replicas:
        each task's prefix on 3-8 of 8 workers over the run; per epoch the hot tasks on 6-8 (task 0 on all 8 in epoch
        0); 78 of 3509 routings made a new copy -- the same demand-following replication as at the light load.
      - [x] **Light-load repeat 2 and router ITL SLO 20 ms (2026-10-19; clean; `bash /tmp/h20/fanout_more.sh`).** Light:
        disagg 127.7 tok/s / 10.23 s / TPOT 5.8 ms (p50 6.60), goodput 84 / 101 tok/s; ours (ITL SLO 15) 299.6 /
        0.68 s / 8.85 ms (p50 6.48), chat 89 ms / 7.4 ms -- reproduces repeat 1 (294 / 0.69 / 9.18, p50 6.49). Heavy,
        ITL SLO 20 ms: agent **562 tok/s / 0.73 s / TPOT 12.0 ms (p50 8.5)**, chat 305 ms / 10.3 ms; goodput 553 / 552
        tok/s (98.3 / 98.2%) -- **5.0-7.0x disagg's** and better than ITL SLO 15 on every agent metric (496-505 tok/s,
        16.0-17.8 ms): a looser SLO moves fewer requests off their cached prefix. Agent TPOT gap to disagg narrows from
        ~3x to ~2x (12.0 vs 5.2-5.7 ms).
      - [x] **Router ITL SLO 25 / 30 ms, heavy load (2026-10-19; clean; `bash /tmp/h20/fanout_itl2530.sh`):** agent 562 /
        558 tok/s, TTFT 0.72 / 0.66 s, TPOT 12.0 / 12.0 ms; chat TTFT 268 / 254 ms, ITL 10.3 / 9.5 ms (p99 90 / 42 ms);
        goodput 552 / 549 tok/s. 20-30 ms is a plateau (10 and 15 ms move too many requests off their cached prefix);
        30 ms is the chat-friendliest point and the default from here. Settled numbers vs PD disagg (no spec, two
        repeats), heavy load: agent throughput 3.1-3.3x, TTFT -96%, goodput 5.0-7.0x, TPOT 2.1-2.3x disagg's; chat TTFT
        -99%, chat ITL ~1.8x disagg's.
      - [x] **30 ms repeats (2026-10-19; clean; `bash /tmp/h20/fanout_itl30rep.sh`).** Heavy rep 2: agent 569 tok/s /
        0.64 s / TPOT 12.0 ms (p50 9.5), chat 249 ms / 9.8 ms, goodput 562 tok/s (98.7%) -- reproduces rep 1 (558 /
        0.66 / 12.0, 549). Light: 299 tok/s / 0.56 s / 10.0 ms (p50 7.35), chat 60 ms / 6.4 ms, goodput 293 tok/s -- vs
        the 15 ms SLO at light load (294-300 / 0.68-0.69 s / 8.9-9.2 ms, p50 6.5): the looser SLO costs TPOT at light
        load and wins it at heavy load, so the ITL SLO wants to scale with load (the router could set it from the
        observed decode interval). **Settled vs PD disagg without spec (natural text):** heavy -- agent throughput
        3.2-3.3x, TTFT -96%, goodput 5.0-7.1x, TPOT 2.1-2.3x disagg's, chat TTFT -99%, chat ITL ~1.9x; light -- agent
        2.3x, TTFT -94..-95%, goodput 3.0-3.5x, TPOT p50 equal (15 ms SLO) / mean 1.5x, chat TTFT -99%.
      - [x] **Load-adaptive router ITL SLO, K = 2 (2026-10-19; `router.py --itl-slo-auto K --itl-slo-floor-ms 10`: SLO = K
        x EWMA of finished requests' per-token decode time, logged per decision; clean; `bash /tmp/h20/fanout_auto.sh`).**
        Effective SLO p10 / p50 / p90 (`tasks/perf_model/router_itl_slo_values.py`): light 12.0 / 12.6 / 21.9 ms, heavy
        16.6 / 19.6 / 30.6 ms. Light: agent 302 tok/s / 0.65 s / TPOT 8.71 ms (p50 6.49), goodput 296 tok/s -- the best
        light-load point (fixed 15 ms: 8.9-9.2 ms). Heavy: 521 tok/s / 0.88 s / 14.6 ms, goodput 509 tok/s -- behind
        fixed 20-30 ms (~560 / 12.0 ms / ~552): the SLO's low tail (~17 ms) moves too many requests off their cached
        prefix.
      - [x] **K = 2.5 (2026-10-19; clean; `bash /tmp/h20/fanout_auto25.sh`):** effective SLO p10 / p50 / p90 heavy 19.8 /
        22.1 / 33.9 ms, light 15.3 / 16.0 / 26.5 ms. Heavy: agent 555 tok/s / 0.75 s / TPOT 12.1 ms (p50 8.9), goodput
        545 tok/s; light: 298 tok/s / 0.64 s / 9.6 ms (p50 6.51), chat 88 ms / 7.1 ms. **One setting tracks the
        hand-tuned optimum at both loads** (fixed 20-30 ms at heavy: ~560 / 12.0 / ~552; fixed 15 ms at light: 8.9-9.2
        ms), within 1-8%; K = 2.0 is slightly better at light load (8.7 ms) and falls behind at heavy load. Default for
        the router from here: `--itl-slo-auto 2.5 --itl-slo-floor-ms 10`.
      - [x] **Per-class ITL SLO for chat-length requests (negative; 2026-10-19; `router.py --itl-slo-short-ms`; heavy
        natural-text load, adaptive K = 2.5 for the rest; clean; `bash /tmp/h20/fanout_short.sh`).** Chat SLO 12 / 9 ms:
        agent 505 / 461 tok/s, TTFT 0.92 / 1.08 s, ITL 16.9 / 21.7 ms; chat ITL 10.5 / 10.5 ms (vs 10.8 ms without).
        Chat ITL does not move: agent prefill runs on every node at this load, so the SLO check finds no prefill-free
        node to keep chat decode on, and moving agents only costs their cached prefixes. Off by default (0).
      - [x] **Soft node roles (negative; 2026-10-19; `router.py --long-prefill-pool N`: cold long prefills prefer workers
        [0, N), cache-less short requests the rest, hit-holding requests keep hit-first order; heavy natural-text load,
        adaptive K = 2.5; clean; `bash /tmp/h20/fanout_pool.sh`).** Pool 6 / 4: agent 497 / 507 tok/s, TTFT 1.07 / 1.02
        s, TPOT 16.2 / 15.3 ms, goodput 477 / 490 tok/s; chat ITL 9.8 / 9.9 ms (p99 76 / 68). vs no pool (555 / 0.75 s /
        12.1 ms / 545; chat 10.8 ms): chat ITL -1 ms, agents -9..-13%. Agent prefill still reaches the chat-side workers
        (SLO overflow and agents' own appends), so chat decode is never prefill-free; disagg's 5 ms chat ITL comes from
        decode pods that by construction never prefill. Off by default (0). With this the routing levers for the
        decode-latency gap (fixed / adaptive / per-class ITL SLO, soft roles) are exhausted at this scale.
      - [x] **Time-based draining (negative; 2026-10-19; `router.py --drain-frac 0.25 --drain-period-s {20,8}`: a rotating
        quarter of the workers takes no new requests per period; heavy natural-text load, adaptive K = 2.5; clean;
        `bash /tmp/h20/fanout_drain.sh`).** Period 20 / 8 s: agent 521 / 503 tok/s, TTFT 0.99 / 1.03 s, ITL 14.4 / 16.4
        ms; chat ITL 12.1 / 12.1 ms (p99 315 / 317 ms) -- worse than no draining (555 / 0.75 s / 12.1 ms; chat 10.8 ms,
        p99 ~90). Taking a quarter of the cluster out of admission concentrates new prefills on the rest, so the open
        workers' decode sees more prefill, not less, and requests moved off their cached node re-prefill. Off by
        default (0).
- [x] **3. EDF with an EWMA acceptance length** (2026-10-08, H200) — measured 2.10 tokens/step. a=1
      collapses EDF; a per-request EWMA is biased low by the min over requests; a global EWMA is
      right, but every per-step chunk cap loses. **Deadline ORDER wins**: chat TTFT -62%, codex
      P99 ITL -27%. Full stack vs stock MTP: TTFT -95%/-51%, ITL -42%/-38% (chat/codex, 24/s, 3 rounds).
- [x] **The goal's required schedule on 8 H20s -- EAGLE3 MTP in the 4-mode fused step (2026-10-18,
  `mn_fourmode_unc_8gpu_20261018.sh`):** KERNEL=mtpsplit SPEC=1 (prefill chunk + verify rows in one target forward on
  the FA3 / FA2 split), worker EDF + chunk preemption, TTFT deadlines on uncached tokens, cache-affinity routing; chat
  16 req/s, agent C=16, 8 workers, 0 tracebacks / 0 fallbacks. vs stock colocation + MTP (same cohort): chunk 2048
  agent +20% / -33% / -5%, chat +23% / -84% / -19%; chunk 1024 agent +15% / -31% / -13%, chat +22% / -87% / -28%.
  **TTFT and throughput clear; ITL does not.** Each mixed step carries a prefill chunk over 32-64K contexts plus the
  verify rows (~100 ms on H20, chat TPOT 108-117 ms), and every decoding request waits for it; smaller chunks buy ITL
  with agent TTFT, already at -31% at 1024. Spatial isolation (pdmux) is what removes that wait, and MTP on top of pdmux
  was negative on H200 -- so on this hardware the 4-mode MTP schedule and the ITL bar pull against each other.
  **Chunk sweep (`CHUNKS="512 256"`, 8 workers, 0 tracebacks / 0 fallbacks):** chunk 512 agent +37% / -21% / -45%, chat
  +47% / -87% / -61%; chunk 256 agent +45% / +11% / -71%, chat +48% / -76% / -76%. With 2048 / 1024 above, the agent
  pair moves along a frontier -- (-33%, -5%), (-31%, -13%), (-21%, -45%), (+11%, -71%) -- and between 1024 and 512 it
  trades linearly, so no single chunk size clears both agent metrics; at chunk 512 agent TTFT (-21%) is the only miss.
  **Per-step chunk from EDF deadlines (`apply_edf_deadline_chunk.py` + `_long.py`, `DEADLINE_CHUNK=512:2048:<tok/s>`,
  8 workers, 0 tracebacks / 0 fallbacks):** small chunk unless a late request (deadline order, cumulative uncached
  tokens at the small chunk's pace) needs it. Pace 3000 / 2000 tok/s: agent +13% / -35% / -5%, +21% / -33% / -11%; with
  only requests longer than one small chunk counting as late: agent +13% / -36% / -2%, chat +21% / -92% / -23%. Under
  this load the agent backlog is always behind its deadlines, so the TTFT-driven choice degenerates to the large chunk;
  an ITL budget would degenerate to the small one (chunk 512: -21% / -45%). **On H20 the 4-mode MTP schedule's agent
  TTFT / ITL trade is physical -- prefill work shares each mixed step with decode -- and per-step chunk choice moves
  along the same frontier; it does not clear both by 30%.** Spatial isolation (pdmux, non-speculative) clears all 12
  vs stock + MTP; the 4-mode MTP schedule clears 10 of 12 at chunk 512 (agent TTFT the miss) or at chunk >= 1024
  (agent and chat ITL the misses).
- [x] **Single-GPU winner on 8 H20s (2026-10-17):** PD multiplexing + worker-side EDF with uncached-token
  deadlines (divisor 30) cleared all 12 bar means against stock MTP on one H200 (main doc, Direct baselines). Here it
  runs on every colocated worker under cache-affinity routing (`mn_run.py` suffix `_pdmux`; `mn_node.sh` PDMUX /
  PDMUX_DIV / EDF_UNCACHED; uncached-deadline patch applied on all four pods), vs stock colocation with MTP and stock
  colocation without speculation, same session, agent C=16 + chat 16 / 32 req/s. On H20 pdmux forms 5 SM groups
  (78 SMs: 56/22, 48/30, 40/38 prefill/decode) instead of 8. `bash tasks/pd_colocation/mn_pdmux_8gpu_20261017.sh`.
  *Run 1 (19:33-20:3x): our arm invalid -- 3 of 8 workers OOMed on their first unchunked 32-64K agent prefills (6.3 GB
  left after the KV pool at mem fraction 0.85; H20 has 95 GB vs H200's 140 GB), so the router lost them, the agent
  bench never completed and chat served 424 / 2400. The baselines in that session are clean (8 workers, 0 tracebacks /
  0 fallbacks): stock colocation + MTP at chat 16 / 32 req/s: agent TTFT 15.7 / 22.8 s, ITL 50.9 / 42.1 ms, tput 162 /
  146; chat TTFT 16.8 / 29.8 s, ITL 129.6 / 74.7 ms, tput 2214 / 3467. Stock colocation without speculation: agent
  7.2 / 7.4 s, 27.9 / 43.9 ms, 266 / 217; chat 2.7 / 2.8 s, 27.3 / 69.2 ms, 3243 / 5432.* Our arm reruns at
  `MEM_FRAC=0.70` (`OURS_ONLY=1 bash mn_pdmux_8gpu_20261017.sh`).
  **Run 2 (our arm at 0.70, 20:03-20:28): 8 workers, 0 tracebacks / 0 fallbacks on every pod (`mn_audit.py`), 20.7 GB
  activation headroom; prefill cache hit 0.395 / 0.368 (baselines 0.394 / 0.288 MTP, 0.394 / 0.367 no-spec) -- routing
  is not the difference.** Agent C=16, tput / mean TTFT / mean ITL:
  | chat req/s | arm | agent | chat |
  |---|---|---|---|
  | 16 | stock coloc + MTP | 162 / 15.68 s / 50.9 ms | 2214 / 16.84 s / 129.6 ms |
  | 16 | stock coloc, no spec | 266 / 7.18 s / 27.9 ms | 3243 / 2.74 s / 27.3 ms |
  | 16 | **ours: pdmux + EDF (uncached deadlines, divisor 30)** | 231 / 13.97 s / 15.1 ms | 3216 / 5.19 s / 16.0 ms |
  | 32 | stock coloc + MTP | 146 / 22.77 s / 42.1 ms | 3467 / 29.84 s / 74.7 ms |
  | 32 | stock coloc, no spec | 217 / 7.36 s / 43.9 ms | 5432 / 2.80 s / 69.2 ms |
  | 32 | **ours** | 195 / 16.71 s / 19.1 ms | 5348 / 8.98 s / 26.9 ms |
  vs stock coloc + MTP: agent +42% / -11% / -70% (16), +33% / -27% / -55% (32); chat +45% / -69% / -88%, +54% / -70% /
  -64% -- all clear except agent TTFT. **vs stock coloc without speculation (the stronger H20 baseline): ITL -41..-61%
  but TTFT 1.9-3.2x worse and throughput -1..-13%.** The single-GPU H200 result does not carry over: on H20 (148 dense
  TFLOP/s, 78 SMs, 5 pdmux groups) prefill is the bottleneck, and pdmux runs it unchunked on 40-56 of 78 SMs, so
  TTFT pays for the decode SMs. On H20 the bar is not met against stock non-speculative colocation; the lever there
  is prefill capacity, not decode isolation. Command: `bash tasks/pd_colocation/mn_pdmux_8gpu_20261017.sh` (run 1:
  baselines), `OURS_ONLY=1 bash ...` (run 2); audit `SID=bb854a10b8954223 NPODS=4 python mn_audit.py <arm>_8g_pdmux_c16_r<R>`.
  - [x] **Prefill-heavy SM groups on H20 (2026-10-17 20:31, negative):** `divide_sm` gives decode >= 16 SMs and prefill
    >= half, so prefill holds at most 56 of 78 SMs whenever decode runs. pdmux's `manual_divisions` (`mn_node.sh`
    PDMUX_MANUAL) with 64/14 below 24 decoding requests: every worker died at start-up -- `CUBLAS_STATUS_EXECUTION_FAILED`
    in a GEMM during decode CUDA-graph capture on the 14-SM green context (twice per GPU, then killed). With partitions
    in multiples of 8 and a working decode group needing >= 16 SMs, prefill cannot exceed 56 of H20's 78 SMs under
    pdmux: the TTFT loss on H20 is structural to SM-split colocation on this GPU, not a tuning gap.
    `mn_pdmux_manual_8gpu_20261017.sh`.
  - [x] **Backlog-triggered SM sharing (2026-10-17, `apply_pdmux_backlog_share.py`, `PDMUX_SHARE=N`):** pdmux's group 0
    is unpartitioned (prefill and decode on normal streams over the whole GPU, as stock colocation); a split-prefill
    batch formed beside running decode now runs there when >= N uncached prompt tokens wait, else in pdmux's
    decode-isolating split. Chat 16 req/s, agent C=16, 8 workers healthy, 0 tracebacks / 0 OOMs, sharing fires on
    large backlogs only (e.g. 45K / 57K / 112K tokens), agent tput / TTFT / ITL ; chat TTFT / ITL:
    N=32K: 303 / 5.91 s / 22.0 ms ; 1.23 s / 20.8 ms. N=128K: 242 / 13.57 s / 15.1 ms ; 3.91 s / 14.3 ms.
    **N=32K vs stock colocation without speculation: agent +14% / -18% / -21%, chat TTFT -55% / ITL -24%** (chat
    throughput = offered load in every arm); vs stock colocation + MTP: agent +87% / -62% / -57%, chat -93% / -84%.
    P99 ITL rises to ~340 ms (shared batches stall decode, as in stock). N trades TTFT for ITL; 16K / 64K at 16
    req/s and 32K at 32 req/s (`POINTS="32:32768 16:65536 16:16384" bash mn_pdmux_share_8gpu_20261017.sh`, all 8
    workers, 0 tracebacks / 0 fallbacks). vs stock colocation without speculation (same cohort, run-1 baselines):
    | chat req/s | N | agent tput / TTFT / ITL | chat tput / TTFT / ITL |
    |---|---|---|---|
    | 16 | 16K | +12.8% / -15.7% / -21.9% | -0.6% / -61.2% / -25.6% |
    | 16 | 32K | +13.9% / -17.7% / -21.1% | -0.6% / -55.0% / -23.8% |
    | 16 | 64K | -4.9% / +69.0% / -43.0% | -0.5% / +63.2% / -44.3% |
    | 16 | 128K | -9.0% / +88.9% / -45.9% | -0.6% / +42.8% / -47.6% |
    | 32 | 32K | +9.1% / -10.5% / -21.9% | +14.5% / -38.9% / -35.7% |
    vs stock colocation + MTP at 32 req/s, N=32K: agent +61.5% / -71.1% / -18.5%, chat +79.4% / -94.3% / -40.4%.
    **Every metric now beats the strongest H20 baseline at N <= 32K, but the agent pair sits at about (-18%, -21%)
    against the (-30%, -30%) bar**; raising N buys ITL with TTFT, lowering it below 32K stops helping. On H20 the
    bar is met against stock colocation + MTP except agent ITL at 32 req/s (-18.5%), and not met against stock
    non-speculative colocation.
    - [x] **N=48K, rep 1 (2026-10-17 22:3x-22:5x, 8 workers, 0 tracebacks / 0 fallbacks):** chat 16 req/s agent 259 /
      8.49 s / 19.0 ms, chat 2.20 s / 17.5 ms; 32 req/s agent 228 / 10.08 s / 28.3 ms, chat 5766 / 4.05 s / 35.1 ms.
      **vs stock colocation + MTP (same cohort): all 12 clear -- 16 req/s agent +59% / -46% / -63%, chat +46% / -87% /
      -87%; 32 req/s agent +56% / -56% / -33%, chat +66% / -86% / -53%.** vs stock colocation without speculation:
      16 req/s agent -3% / +18% / -32%, chat TTFT -20% / ITL -36%. Smallest margin vs MTP: agent ITL at 32 req/s
      (-32.8%). **Repeat 2 (8 workers, 0 tracebacks / 0 fallbacks): 16 req/s agent 271 / 8.39 s / 18.6 ms, chat 2.40 s /
      18.7 ms; 32 req/s agent 227 / 9.94 s / 29.9 ms, chat 5838 / 4.43 s / 38.1 ms. Two-repeat means vs stock
      colocation + MTP: 16 req/s agent +63% / -46% / -63%, chat +46% / -86% / -86%; 32 req/s agent +56% / -56% /
      -30.9%, chat +67% / -86% / -51% -- all 12 clear in the mean;** per repeat the 32 req/s agent ITL is -32.8 /
      -29.0%, so that one metric is borderline. **Repeat 3 at 32 req/s (8 workers, 0 tracebacks / 0 fallbacks): agent
      226 / 10.19 s / 28.2 ms, chat 5765 / 4.04 s / 38.5 ms -- agent ITL -33.0%; three-repeat agent ITL mean -31.6%
      (-32.8 / -29.0 / -33.0), TTFT -56%.** Command: `POINTS="16:49152 32:49152" [REP_SUFFIX=_rep2] bash
      tasks/pd_colocation/mn_pdmux_share_8gpu_20261017.sh`.
  - [x] **SLO-check routing over these workers (2026-10-17, negative):** router `cache_slo` (calibrated H20 prefill
    curve, 10 s TTFT SLO, resident check off, no-feasible -> top-hit) over pdmux + EDF + 32K backlog sharing, chat 16
    req/s, 8 workers healthy, 0 tracebacks / 0 fallbacks: agent 302 / 6.18 s / 20.6 ms, chat 1.47 s / 19.3 ms -- vs
    cache-affinity routing on the same workers (303 / 5.91 s / 22.0 ms; 1.23 s / 20.8 ms) no gain: moving a cold agent
    prompt off its backed-up cache node costs the hit about what it saves in queueing. `bash
    tasks/pd_colocation/mn_pdmux_slo_8gpu_20261017.sh`.
  - [x] **Agent-TTFT floor on H20 (2026-10-17, bound from the calibrated prefill curve):** a cold 48K prompt (the
    cohort's median context) costs ~11.1 s of prefill on one H20, a cold 32K prompt 6.1 s (`calibrate_h20.py` curve);
    about half of the 96 agent turns are cold first turns and the rest hit the cache (measured token hit ~0.39) at
    < 1 s. With zero queueing the mean agent TTFT is therefore ~(48 x 11 + 48 x 0.8) / 96 = **~5.9 s**. Ours is 5.91 s;
    stock colocation without speculation 7.18 s; the bar's -30% would be 5.03 s, **below the compute floor**. Against
    that baseline the agent-TTFT target is unreachable on H20 by any scheduling, routing or SM-split change; only
    less prefill work (more cache reuse or less compute per token) could move it.
  - [x] **TP=2 per pod (2026-10-17, `TP=2` in `mn_node.sh` / `mn_run.py`, `mn_tp2_8gpu_20261017.sh`):** one worker per
    pod over both GPUs halves the per-request prefill floor. *pdmux + TP=2 hangs:* 4 workers came up, served until
    21:33, then every GPU sat at 100% with no log progress for 30 min (green-context streams vs TP collectives); killed
    by PID. Instead (22:05-22:30, 4 workers, 0 tracebacks / 0 fallbacks): EDF with uncached deadlines on chunked
    colocation + cache-affinity routing vs stock colocation, both TP=2, chat 16 req/s, agent tput / TTFT / ITL ; chat
    TTFT / ITL: ours 251 / 5.30 s / 41.7 ms ; 1.63 s / 59.4 ms; stock 239 / 4.76 s / 45.1 ms ; 2.50 s / 61.3 ms.
    TP=2 lowers agent TTFT for any scheduler (stock 7.18 -> 4.76 s) but raises ITL (27.9 -> 45.1 ms agent, 27.3 ->
    61.3 ms chat). **The best stock setting differs per metric (TP=2 for TTFT, TP=1 for ITL) and our best H20 config
    (TP=1 pdmux + EDF + 32K sharing: 5.91 s / 22.0 ms agent) sits between them**: vs stock TP=2 agent TTFT +24%, ITL
    -51%; vs stock TP=1 -18% / -21%. No tested H20 configuration clears both TTFT and ITL by 30% against the best
    stock setting for each.

- [ ] **7. Inter-node routing: predict whether a request meets its TTFT and ITL SLO on node i, given
      i's estimated hit length x_i and current load** (added 2026-10-17 on the user's request; needs
      more design thinking before more code). This is the decision behind point 1's SLO check: when the
      highest-hit node is too hot, route to the highest-hit node predicted to still meet the SLO.
      **Current predictor** (`router.py` `predict()` / `feasible()`, policy `cache_slo`; router-side
      bookkeeping only, no worker query):
      `TTFT_i = (Σ prefill_left of in-flight requests on i with deadline <= this one + n - x_i) / prefill_tok_s`;
      `TPOT_i = grid(running_i + 1, mean ctx incl. this prompt) / acceptance EWMA_i`; feasible iff both
      meet the SLO and no later-deadline request queued on i is pushed past its own deadline.
      **Known errors, by sign:**
      - pessimistic: the global KV pool (HiCache + mooncake L3) pulls the missing prefix
        best-effort while the request waits in the queue, so the effective hit at start is
        `x_i + min(x_pool - x_i, W_i * B_src / bytes_per_token)`, not `x_i`. Scale on H20 / Qwen-7B:
        a 32K prefix is ~1.8 GB of KV, ~0.1 s at ~20 GB/s RDMA, vs ~6 s to recompute at 5K tok/s.
        With multi-second queue waits the pull almost always finishes, so ignoring it overstates a
        non-holder's TTFT by up to the whole recompute time. (Exact only with the pool off, as in
        every H20 run so far.)
      - optimistic: one blended prefill rate (5 012 tok/s); calibrated 8.3K tok/s at 4K falling to
        4.4K at 48K (`/tmp/h20/mn_slo_calibrate.log`), so the longest agent prefills -- the ones that
        decide agent TTFT -- are under-predicted.
      - optimistic: decode interference -- colocated steps mix <= 2048-token chunks with decode rows
        (4-mode), so prefill drains slower when many rows decode; ignored.
      - optimistic: future earlier-deadline arrivals preempt the queue under EDF (and chunk parking),
        so the current queue is a lower bound on the work ahead.
      - optimistic: x_i is the router's hash estimate; the worker may have evicted those blocks.
      **Open design questions:**
      - target: a calibrated (unbiased) prediction plus an explicit margin, or a quantile (e.g. P90)
        prediction? The cost is asymmetric -- a false SLO check loses the hit (recompute elsewhere,
        cache replication as a side effect), a false accept misses an SLO.
      - queue drain as a sum of fitted step times (`fit_mtp_step.py`: chunk tokens, prefix ctx,
        decode rows, decode KV, acceptance) over ceil(ahead / chunk) steps -- point 3's claim that the
        fused/split kernel is easy to model is tested here.
      - arrival-aware wait: add the expected earlier-deadline work arriving during the wait (rate x
        work of tight-deadline classes, e.g. chat).
      - transfer term: source bandwidth from an EWMA of observed pulls per source (no router query);
        sglang's prefetch policies (best_effort / wait_complete / timeout) change what "best effort"
        guarantees; source eviction during the wait; host-to-device copy contention.
      - ITL over the request's lifetime, not one step: resident rows finish and others arrive; the
        ITL SLO applies to every resident row, and all share one step time.
      - staleness: router-side bookkeeping only sees first-token and finish events; piggybacking
        worker queue state on response headers is cheaper than a per-decision query.
      - interaction with point 2 (routing to a non-holder IS replication) and point 4 (which holder to
        pull from).
      **Sub-items:**
      - [x] **Current SLO check (no pool, no transfer term) vs cache affinity, same session, final stack,
            8 H20 (2026-10-17, `bash tasks/pd_colocation/mn_slo_check_8gpu_20261017.sh`; calibrated on one
            final-stack worker: prefill 8.3K tok/s at 4K -> 4.4K at 48K, blended `--prefill-tok-s 5012`,
            decode grid conc x ctx). All 4 arms 8 workers 0 tracebacks / 0 fallbacks. Out tok/s / mean
            TTFT / mean ITL:
            | point | arm | agent | chat | moved / no feasible node | prefill hit |
            |---|---|---|---|---|---|
            | C=16, chat 8/s | cache | 377 / 6.80 s / 17.0 | 1659 / 304 ms / 16.2 | - | 65.2% |
            | C=16, chat 8/s | SLO check | 366 / 7.14 s / 17.6 | 1658 / 474 ms / 14.7 | 345 / 8 of 2 500 | 60.6% |
            | C=32, chat 8/s | cache | 257 / 11.4 s / 75.5 | 1657 / 2.41 s / 135 | - | 64.4% |
            | C=32, chat 8/s | **SLO check** | **319 / 12.6 s / 58.0** | 1657 / 2.65 s / **91.7** | 1 667 / 314 of 2 596 | 54.9% |
            At C=16 the SLO check is neutral-to-worse (agent TTFT +5%, chat TTFT +56%): without a pool, a
            moved request recomputes its prefix. At C=32 it wins: agent throughput +24%, ITL -23%,
            chat ITL -32%, at +10% TTFT on both tenants -- spreading hot holders' decode load. The
            control reproduces the 2026-10-16 C=32 fixed-router run within 3%. **SLO check stack vs stock
            colocation at C=32:** agent throughput +44%, ITL -40%; chat throughput +13%, TTFT -64%, ITL
            -50% -- every bar metric except agent TTFT (+4%). The SLO check moved a request on 64% of C=32 decisions,
            and 314 had no feasible worker: the predictor is coarse (item 7's error list).
      - [x] **Pool-on A/B (2026-10-17, `bash tasks/pd_colocation/mn_pool_slo_check_8gpu_20261017.sh`; agent
            C=16, chat 8 req/s; workers at sglang's default `best_effort` prefetch).** All 3 arms 8 workers
            0 tracebacks / 0 fallbacks. Out tok/s / mean TTFT / mean ITL (chat P99 TTFT):
            | arm | agent | chat | requests moved off their top-hit node | prefill hit |
            |---|---|---|---|---|
            | cache | 358 / 7.59 s / 17.5 | 1659 / 259 ms / 15.3 (5.5 s) | - | 65.0% |
            | SLO check, no transfer term | 333 / 8.04 s / 19.4 | 1659 / 176 ms / 14.8 (1.35 s) | 379 / 2 500 | 59.0% |
            | SLO check + transfer term | 349 / 7.40 s / 20.1 | 1658 / 458 ms / 17.6 (15.9 s) | 406 / 2 500 | 59.7% |
            **The transfer term never fired:** it predicted 0 pulled tokens for every agent request.
            Under `best_effort` a pull lands only while a request waits in the queue, and the SLO check sends
            requests to idle non-holders, where the predicted wait is 0 -- so 0 is the right prediction,
            and the workers did pull almost nothing (60-412 tokens per agent request on average); the
            bandwidth EWMA never moved off its 10 GB/s prior. **The two SLO-check arms therefore ran the same
            routing logic, and their difference is run-to-run noise:** 8% on agent TTFT and 1.35 vs 15.9 s
            on chat P99 TTFT. Single runs do not resolve these metrics; the earlier single-run SLO check A/Bs
            carry the same caveat. Design implication: `best_effort` is the wrong prefetch policy for
            deliberate non-holder routing -- recomputing a 32K prefix takes ~6 s, pulling it ~0.2 s -- so
            the workers should wait for the pull (`timeout` / `wait_complete`) and the router should
            predict the same. Without a pool the prefix costs a full recompute, so the SLO check is neutral
            at C=16 (above) and wins at C=32 only by spreading decode load.
      - [x] **Timeout-prefetch A/B (2026-10-17, `bash tasks/pd_colocation/mn_pool_timeout_8gpu_20261017.sh`,
            interleaved x2).** Agent tok/s / mean TTFT, chat mean TTFT (P99):
            | arm | rep 1 | rep 2 |
            |---|---|---|
            | cache | 359 / 7.36 s, 368 ms (9.8 s) | 366 / 7.37 s, 376 ms (8.2 s) |
            | SLO check + transfer | 343 / 7.84 s, 177 ms (1.75 s) | 354 / 8.01 s, 400 ms (12.8 s) |
            Cache affinity repeats within ~2%; the SLO check's chat tail does not (P99 1.75 vs 12.8 s). Two-rep
            means: SLO check agent throughput -4%, agent TTFT +8%, chat TTFT -22% -- **at C=16 on this mix the
            SLO check's chat gain is inside the noise and it costs the agent ~8% TTFT.** **The transfer term still cannot fire on this workload, for a
            reason the policy does not change:** all 27 agent requests the SLO check moved were first rounds with
            local hit 0 and pool hit 0 (cold prompts, nothing to pull); a second round's prefix sits on its
            holder and is rarely moved off it; 1 of 99 agent decisions predicted any pull. TraceLab sessions share
            no prefix across sessions, so with 2 rounds there is no request that is both moved and
            transferable. Testing it needs hot shared prefixes on overloaded holders: queued
            `mn_hot_transfer_8gpu_20261017.sh` (HOT=1, 4 x 16K shared system prompts at 16 req/s; cache |
            SLO check ignoring the pool | SLO check + transfer, x2). The cold moved requests also show the
            predictor's spread: 8.9 s predicted vs 0.7 s measured, 10.7 s vs 37.9 s.
      - [x] **Hot-prefix transfer A/B (`bash tasks/pd_colocation/mn_hot_transfer_8gpu_20261017.sh`; HOT=1, 4 groups x 16K shared
            prefixes at 16 req/s, agent C=16, pool on, timeout prefetch; cache | SLO check ignoring the pool |
            SLO check + transfer, x2).** First run (02:0x): cache rep 1 agent 239 tok/s / 10.4 s / 30.0 ms,
            chat 1768 / 9.8 s / 75.4 ms. The SLO check-ignoring-pool rep 1 **crashed a worker** on
            `assert self.chunked_req is None` (`scheduler.py:4314`): EDF's reservation for earlier-deadline
            requests (`SGLANG_EDF_RESERVE_EARLIER`, on with `PREEMPT=1`) caps the in-progress chunk below
            the step budget, and the admission guard that refuses to chunk a second request only applied
            under `shortest-prefill-first`, so a waiting request was chunked beside the in-progress one.
            Fixed by applying the guard whenever a chunked request is in flight
            (`tasks/pd_colocation/edf_one_chunked_req.patch`, applied to all 4 pods); every EDF + PREEMPT
            arm before this carried the bug, but only the hot-prefix mix (many concurrent 16K-cached
            requests) triggered it. **Rerun on the patched tree (2026-10-17 02:5x-03:4x), all 6 arms 8
            workers 0 tracebacks / 0 fallbacks.** Two-rep means, out tok/s / mean TTFT / mean ITL:
            | arm | agent | chat | requests / worker | prefill hit |
            |---|---|---|---|---|
            | cache | 236 / 9.96 s / 30.5 | 1664 / 13.9 s / 78.6 | 19-807 | 92.9% |
            | SLO check ignoring the pool | 224 / 10.1 s / 35.5 | **2017 / 1.41 s / 51.6** | 328-506 | 93.3% |
            | SLO check + transfer | 216 / 10.2 s / 35.5 | **2015 / 1.43 s / 50.1** | 221-549 | 93.3% |
            **SLO check vs cache: chat throughput +21%, TTFT -90%, ITL -34%; agent throughput -5..-8%, TTFT
            +2..+3%, ITL +16%.** Cache affinity piles each hot group on its holder (19 vs 807 requests per
            worker); the SLO check spreads them at the same hit rate, because the 4 hot prefixes replicate after
            a few recomputes. Reps agree within ~2% for the SLO-check arms (chat TTFT 1.61 / 1.22 s); the cache
            arm varies more (agent 271 / 200 tok/s). **The transfer term makes no difference here, as
            expected:** only 37 of 1 623 moved requests had a pool hit beyond their local one (4 x 16K
            replicates almost immediately). On those 37 its prediction was close: 26.3K tokens predicted vs
            21.4K pulled (81%), TTFT predicted / measured median 0.94 (p10 0.09, p90 1.93), and the learned
            pull bandwidth settled at 0.75 GB/s (prior 10). The predictor still rates most nodes
            infeasible: 1 215 of the 1 623 moves found no feasible worker (least-bad).
            A transferable workload is queued: `mn_manyprefix_transfer_8gpu_20261017.sh` (192 x 16K
            shared prefixes ~3.1M tokens, fits the pool but not one node; same 3 arms x 2).
      - [x] **Predictor error (same runs, `ttft_obs` vs `ttft_pred`):** predicted / measured median 0.64
            (chat) and 0.72-0.73 (agent) -- about 30% optimistic, as the error list expected (blended
            prefill rate, decode interference, future arrivals). Agent error p10 / p90 -6.4..-7.7 s /
            +9.2..+9.6 s: far too wide for a hard SLO check at a ~8-15 s deadline.
      - [ ] step-model queue drain (with decode interference) replacing the constant prefill rate.
            **Step 1 implemented 2026-10-17 (`router.py --prefill-curve`, not yet measured):** prefill time
            from the calibrated cold-TTFT curve, `T(ctx) - T(ctx - uncached)`, so long contexts cost more
            per token and a cached prefix still costs attention: a 4K append on a 60K prompt predicts
            1.27 s vs 0.82 s with the blended rate (+55%), 48K cold 11.2 s vs 9.8 s. Decode interference
            and future arrivals are still unmodelled. `calibrate_h20.py` now prints the curve.
      - [ ] transfer-aware effective hit; requires an arm with the L3 pool on. **Implemented
            2026-10-17 (`router.py`, not yet measured):** the router mirrors the pool's contents
            (write-through on each finished request) with the same chained block keys as HiCache's
            page keys, takes `missing = pool hit - local hit`, and predicts the pulled tokens from
            the predicted queue wait and the workers' prefetch policy (best_effort: what arrives
            during the wait; timeout: wait up to 1 s + 0.25 s/Ktok; wait_complete: all). Pull
            bandwidth is an EWMA learned from finished requests (`cached_tokens` includes
            prefetched pages, minus the router's local-hit estimate, over the observed queue
            wait); a complete pull only raises it. The router now logs `ttft_obs` next to
            `ttft_pred`. Unit check: a 32K-prefix request whose holder has a 12 s queue -- without
            the pool the SLO check picks a non-holder at a predicted 9.5 s (3 s queue + 6.5 s recompute);
            with it, 3.0 s (the 1.9 GB pull fits inside the wait). A/B queued:
            `mn_pool_slo_check_8gpu_20261017.sh` (pool on: affinity | SLO check without | SLO check with the
            transfer term, agent C=16, chat 8 req/s).
            The index answers "radix tree?": it IS one, flattened at 256-token blocks (each key
            chains its parent's), so matching is the radix walk; it loses up to 255 tokens per
            match and evicts per block rather than per leaf.
      - [x] **[CORRECTION 2026-10-17 04:5x] Every SLO-check arm above mostly measured its no-feasible-node
            fallback, not the check.** Per-check rejection counts (first logged in the resident-ITL A/B, hot
            mix): ~14-15K node rejections per run came from this request's own predicted ITL, against 0.7-0.8K
            for TTFT, 0.2K for queued requests' TTFT and 0.8K for residents' ITL; 56-63% of decisions found no
            feasible node. Cause: `predict()` counted every in-flight request on a node as a decoding row,
            later-deadline requests still waiting for prefill included, so the calibrated decode grid was read
            at 40-60 rows (~210-250 ms) where measured ITL was 35-59 ms. The measured numbers above stand, but
            the mechanism behind them was mostly "most-slack node among infeasible ones", and the earlier
            explanations should be read that way. Fixed: TPOT now counts the rows decoding beside this request
            once it decodes -- those already decoding plus those queued ahead of it (earlier deadline) -- e.g.
            10 decoding + 40 queued behind now predicts 44.8 ms instead of ~210 ms. The resident-ITL A/B was
            stopped after rep 1 and rerun on the fixed router with a cache-affinity control.
            **Second cause, in the calibration itself (04:3x):** with the row count fixed, own-ITL still
            rejected ~11K nodes and 45% of decisions had no feasible node. `calibrate_h20.py` timed decode
            on the FIRST pass of N concurrent requests, so the other requests' prefill chunks ran inside the
            timed steps (32 x 16K = 512K prefill tokens at ~5K tok/s): the grid read 210 ms at 32 x 16K
            where serving measured 35-84 ms at similar rows and context. This is the "15x knee" the script's
            comment left unexplained -- not a capacity effect. Fixed: prefill every prompt first (1 output
            token), then time a second, fully cached pass. **New grid (05:0x):** 32 x 16K 75 ms (was 210),
            8 x 16K 24 ms (61), 32 x 4K 31 ms (88), 64 x 16K 146 ms (251) -- in line with serving. Prefill
            curve unchanged (8.3K tok/s at 4K -> 4.4K at 48K). Router flags now:
            `--prefill-tok-s 5021 --prefill-curve 4096:491.8,8192:1014.1,16384:2358.4,32768:6072.7,49152:11148.2
            --decode-grid /dev/shm/w/mn/decode_grid.json`. Both A/Bs requeued on it
            (`mn_resident_itl_8gpu_20261017.sh`, then `mn_overload_fallback_8gpu_20261017.sh`). Old grid kept as `/dev/shm/w/mn/decode_grid_prefillmixed_20261017.json` (pod 0).
      - [x] **Resident ITL check, measured (2026-10-17 06:1x-07:3x; `bash tasks/pd_colocation/mn_resident_itl_8gpu_20261017.sh`;
            hot mix 4 x 16K shared-prefix chat at 16 req/s + agent C=16, pool on, timeout prefetch,
            `HC_WRITE=write_through`, clean grid, `--chunk-interference 1.124`, `--no-feasible-fallback cache`
            in both SLO-check arms; interleaved x2; all 6 arms 8 workers 0 tracebacks / 0 fallbacks, prefill
            hit 92.6-93.2%).** Two-rep means, out tok/s / mean TTFT / mean ITL (chat P99 ITL):
            | arm | agent | chat | requests / worker |
            |---|---|---|---|
            | cache affinity | 236 / 10.1 s / 29.4 | 1729 / 15.2 s / 83.5 (457) | 19-807 |
            | SLO check | 208 / 10.6 s / 36.7 | 2015 / 1.69 s / 51.9 (406) | 23-572 |
            | **SLO check + resident ITL check** | 182 / 13.3 s / 36.6 | **2016 / 1.78 s / 35.3 (308)** | 19-632 |
            **Spread** (figure [`h20x8_hot_resident_itl.png`](pd_colocation/figures_20261017/h20x8_hot_resident_itl.png),
            data [`.csv`](pd_colocation/figures_20261017/h20x8_hot_resident_itl.csv): per arm and rep, per-request
            tok/s/user median with p10-p90 bar and p1-p99 whisker, and each rep's mean TTFT / TPOT). Rep 1 / rep 2:
            | arm | chat tok/s/user p1 / p10 / median / p90 / p99 | agent tok/s/user median (p10-p90) | chat TPOT ms | agent TTFT s | agent TPOT ms |
            |---|---|---|---|---|---|
            | cache | 4.2 / 7.1 / 14.0 / 16.5 / 42 ; 4.3 / 7.6 / 14.3 / 24.9 / 126 | 51 (10-242) ; 33 (8-209) | 85.8 ; 81.1 | 10.0 ; 10.2 | 47.2 ; 53.0 |
            | SLO check | 4.4 / 8.5 / 27.7 / 268 / 387 ; 4.4 / 8.5 / 24.5 / 260 / 382 | 20 (7-92) ; 23 (7-94) | 50.1 ; 53.6 | 10.9 ; 10.3 | 65.7 ; 62.5 |
            | + resident check | 5.9 / 11.5 / 63.5 / 266 / 388 ; 5.0 / 11.2 / 68.6 / 196 / 338 | 39 (7-139) ; 28 (7-165) | 33.3 ; 37.3 | 13.3 ; 13.4 | 52.6 ; 69.5 |
            The resident check moves the median chat user from 24-28 to 63-69 tok/s; chat's slow tail (p1 4-6
            tok/s/user) is the same in every arm, and so is the agent's (p10 ~7-10). The noisiest rep-to-rep
            metric is agent TPOT under the resident check (52.6 vs 69.5 ms). Command:
            `PYTHONPATH=<matplotlib> python tasks/pd_colocation/mn_ab_figure.py tasks/pd_colocation/figures_20261017/h20x8_hot_resident_itl
            cache=coloc_cache_edf_mtp_hc_8g_hot_c16_cleangrid2_ctl,slo_check=coloc_slo_edf_mtp_hc_8g_hot_c16_cleangrid2_residents0,slo_check+resident=coloc_slo_edf_mtp_hc_8g_hot_c16_cleangrid2_residents1`. **The resident check
            cuts chat ITL a further -32% (P99 -24%) over the plain SLO check -- -58% (P99 -33%) vs cache
            affinity -- at chat TTFT -88% and throughput +17% vs cache affinity.** It costs the agent: TTFT
            +26% over the plain check (+32% vs cache affinity), throughput -13% (-23%); agent ITL is unchanged
            (36.6 vs 36.7 ms). It rejected 1 544 nodes in rep 1 and doubled the moved requests (893 -> 1 773),
            with no feasible node on 19-23% of decisions (sent to the top-hit node). Reading: the check
            protects the requests that are decoding when a prefill arrives -- mostly chat here -- by delaying
            or relocating the agent's long prefills; it cannot help the agent's own ITL, which is set by the
            chat prefills the agent decodes beside. The ITL SLO is one value (`--itl-slo-ms 100`) for both
            tenants; per-tenant SLOs are the next lever.
      - [x] **Protect requests already decoding on the target node (user, 2026-10-17).** The check
            covered this request's TTFT and ITL and the TTFT deadlines of requests still queued on the
            node, but not the ITL of requests already decoding there -- and admission costs them twice:
            the new request's prefill chunks ride in their decode steps until its prefill ends, then it
            decodes beside them. The hot-prefix A/B shows the cost: moving chat raised agent ITL +16%.
            **Implemented (`router.py --check-residents`; measured above):** reject the node if any
            resident's predicted mean ITL -- decode time spent so far + its remaining steps at the step
            time with the new request + one prefill chunk (`--chunk-tokens`, prefill-curve cost) per
            step while the new request's prefill runs -- exceeds its ITL SLO. Each decision logs which
            check rejected each node (`own_ttft` / `own_itl` / `queued_ttft` / `resident_itl`). Unit
            check: a 24K-uncached request (12 chunks) is admitted beside residents early in their output
            (predicted mean ITL ~44 ms vs a 50 ms SLO) and moved to an idle node when the residents are
            near their end (20 of 300 tokens left). A/B queued on the hot mix:
            `mn_resident_itl_8gpu_20261017.sh` (`--check-residents 0` vs `1`, same build, x2).
      - [ ] **What to do when no node can meet the SLO (2026-10-17).** On the 192 x 16K shared-prefix mix
            (`mn_manyprefix_transfer_8gpu_20261017.sh`, rep 1) the SLO check ignoring the pool lost badly
            to cache affinity -- agent TTFT 48.7 vs 30.4 s, chat TTFT 73.4 vs 28.1 s, chat throughput -19%,
            chat ITL +48% -- because 3 896 of 3 940 decisions (99%) found no feasible node and fell back to
            the most-slack node, scattering requests off their prefixes (779 went to a node with no hit)
            and adding recompute to a saturated cluster. Its predictions there were also pessimistic
            (median 107 s predicted vs 73 s measured). When every node misses, the request misses wherever
            it goes, so the router should minimise added work: `--no-feasible-fallback cache` takes the
            top-hit node (implemented, unit-checked; default stays `slack`).
            **The transfer term made it worse still (rep 1, SLO check + transfer):** chat TTFT 234 s and
            845 tok/s, agent TTFT 85 s. Two faults in the router's pool model, both measured: (1) the pool
            does not hold what the router's mirror says -- 1 395 decisions predicted a pull of 19.5K tokens
            on average, but workers pulled 1.6K per request, and the worker-reported hit fell 80% -> 64%:
            requests were moved to non-holders for pulls that never landed and recomputed. sglang's
            default `write_through_selective` writes a prefix to the pool only after its second hit on a
            worker, while the mirror inserted it after the first. (2) The bandwidth learner fell to 0
            GB/s: it read every incomplete pull as bandwidth-bound and divided by queue waits of 100+ s,
            but an incomplete pull usually means missing pages. Fixed (router.py, 2026-10-17 04:2x): the
            router learns `pull_realized`, an EWMA of pulled / predicted tokens, and scales predicted pulls
            by it; only complete pulls update bandwidth, and only upward; workers can run
            `HC_WRITE=write_through` (pool write after the first compute). Both queued A/Bs below now run
            with these. A/B queued:
            `mn_overload_fallback_8gpu_20261017.sh` (cache affinity vs SLO check + transfer + resident
            ITL + cache fallback, x2).
      - [~] **Clean-grid A/B, hot mix (2026-10-17 05:0x-06:0x, rep 1 only; `mn_resident_itl_8gpu_20261017.sh`).**
            Out tok/s / mean TTFT / mean ITL:
            | arm | agent | chat | moved / no feasible node | rejections by check |
            |---|---|---|---|---|
            | cache | 253 / 9.99 s / 30.2 | 1676 / 15.4 s / 83.1 | - | - |
            | SLO check | 224 / 9.59 s / 36.5 | 2017 / 1.67 s / 56.6 | 797 / 468 of 3 300 | own ITL 3 483, own TTFT 411, queued TTFT 272 |
            | + resident ITL check | 209 / 10.4 s / 36.9 | 2018 / 1.74 s / 56.4 | 2 267 / 1 711 | resident ITL 7 680, own ITL 5 864, own TTFT 1 209, queued TTFT 374 |
            With the clean grid the SLO check itself decides (no-feasible fell from 45-63% to 14%), and the
            outcome holds: chat throughput +20%, TTFT -89%, ITL -32% vs cache affinity, at agent throughput
            -11% and ITL +21%. Predicted / measured TTFT median 0.80 for chat, 2.99 for agents (p90 16.6):
            pessimistic for agents, likely because an in-progress chunked prefill keeps its full uncached
            length in the router's table until its first token. **The resident ITL check fires a lot but
            does not help:** it rejected 7 680 nodes, pushed no-feasible back to 52%, and left agent ITL
            unchanged (36.9 vs 36.5 ms). ~~Its charge -- one full cold chunk time per shared step -- assumes
            prefill and decode serialise inside a step; the 4-mode step overlaps them.~~ **Retracted
            (06:1x, measured):** they do serialise. Calibration now times decoders beside a cold 32K
            prefill: ITL 12.6 -> 380 ms (4K ctx x 8), 34.3 -> 426 ms (4K x 32), 23.2 -> 385 ms (16K x 8),
            against a 334-349 ms cold chunk: added time per step / chunk time = 1.07-1.12
            (`--chunk-interference 1.124`). So the charge was right: on H20 a step carrying a 2048-token
            chunk costs ~10-30x a decode-only step, and serving's ~56 ms mean chat ITL averages over
            stretches without prefill. The check did not help because, with every node rejected on 52% of
            decisions, the most-slack fallback -- which ignores ITL -- chose the node. Rerunning with
            `--no-feasible-fallback cache` in both SLO-check arms, so the resident check changes only which
            feasible node is chosen.
      - [x] **Spread of every routing A/B above (backfilled 2026-10-17 08:xx; user: "you should always record
            that").** Per arm and stream, rep 1 ; rep 2: tok/s/user p1 / p10 / median / p90 / p99, then mean TTFT s and
            mean TPOT ms. Figures show the same per rep (median marker, p10-p90 bar, p1-p99 whisker). Command:
            `python tasks/pd_colocation/mn_ab_figure.py <out_prefix> <label>=<arm dir prefix>,...`.
            *SLO check vs cache affinity, no pool, agent C=16 + chat 8 req/s (`mn_slo_check_8gpu_20261017.sh`, 1 run)* -- [figure](pd_colocation/figures_20261017/h20x8_slocheck_nopool_c16.png), [data](pd_colocation/figures_20261017/h20x8_slocheck_nopool_c16.csv)
            | arm | stream | tok/s/user p1 / p10 / med / p90 / p99 | mean TTFT s | mean TPOT ms |
            |---|---|---|---|---|
            | cache | chat | 4.3/17.2/241/302/393 | 0.30 | 20.3 |
            | cache | agent | 8.2/18.6/94.1/197/260 | 6.80 | 22.9 |
            | slo_check | chat | 4.4/18.0/240/302/394 | 0.47 | 18.9 |
            | slo_check | agent | 6.5/14.7/82.7/204/347 | 7.14 | 28.4 |
            *same at agent C=32 (1 run)* -- [figure](pd_colocation/figures_20261017/h20x8_slocheck_nopool_c32.png), [data](pd_colocation/figures_20261017/h20x8_slocheck_nopool_c32.csv)
            | arm | stream | tok/s/user p1 / p10 / med / p90 / p99 | mean TTFT s | mean TPOT ms |
            |---|---|---|---|---|
            | cache | chat | 2.5/4.0/6.2/137/299 | 2.41 | 155.2 |
            | cache | agent | 3.9/6.3/9.4/17.0/281 | 11.45 | 104.0 |
            | slo_check | chat | 2.9/4.7/9.0/264/343 | 2.65 | 105.2 |
            | slo_check | agent | 3.4/7.2/11.9/41.4/256 | 12.56 | 83.6 |
            *pool on, best_effort prefetch, C=16 (`mn_pool_slo_check_8gpu_20261017.sh`, 1 run)* -- [figure](pd_colocation/figures_20261017/h20x8_pool_besteffort_c16.png), [data](pd_colocation/figures_20261017/h20x8_pool_besteffort_c16.csv)
            | arm | stream | tok/s/user p1 / p10 / med / p90 / p99 | mean TTFT s | mean TPOT ms |
            |---|---|---|---|---|
            | cache | chat | 4.0/16.7/233/294/365 | 0.26 | 20.1 |
            | cache | agent | 7.1/18.4/96.7/179/304 | 7.59 | 22.8 |
            | slo_check+transfer | chat | 4.4/13.1/230/294/365 | 0.46 | 22.7 |
            | slo_check+transfer | agent | 6.8/12.0/83.0/190/324 | 7.40 | 30.9 |
            | slo_check_no_transfer | chat | 4.3/15.2/234/294/365 | 0.18 | 21.6 |
            | slo_check_no_transfer | agent | 7.1/11.4/75.3/164/319 | 8.04 | 27.2 |
            *pool on, timeout prefetch, C=16 (`mn_pool_timeout_8gpu_20261017.sh`, x2)* -- [figure](pd_colocation/figures_20261017/h20x8_pool_timeout_c16.png), [data](pd_colocation/figures_20261017/h20x8_pool_timeout_c16.csv)
            | arm | stream | tok/s/user p1 / p10 / med / p90 / p99 | mean TTFT s | mean TPOT ms |
            |---|---|---|---|---|
            | cache | chat | 4.2/14.8/233/294/364 ; 4.1/16.0/232/293/362 | 0.37 ; 0.38 | 21.8 ; 21.7 |
            | cache | agent | 7.5/17.5/93.3/182/308 ; 6.7/15.8/89.2/211/305 | 7.36 ; 7.37 | 24.3 ; 26.3 |
            | slo_check+transfer | chat | 3.9/12.1/230/293/370 ; 4.0/14.0/231/294/368 | 0.18 ; 0.40 | 25.2 ; 22.7 |
            | slo_check+transfer | agent | 6.3/10.5/82.5/201/333 ; 6.3/15.0/77.9/178/312 | 7.84 ; 8.01 | 30.9 ; 29.0 |
            *hot 4 x 16K prefixes, patched scheduler (`mn_hot_transfer_8gpu_20261017.sh`, x2)* -- [figure](pd_colocation/figures_20261017/h20x8_hot_transfer.png), [data](pd_colocation/figures_20261017/h20x8_hot_transfer.csv)
            | arm | stream | tok/s/user p1 / p10 / med / p90 / p99 | mean TTFT s | mean TPOT ms |
            |---|---|---|---|---|
            | cache | chat | 4.3/7.6/14.5/42.2/115 ; 3.9/7.4/14.8/23.4/84.8 | 13.21 ; 14.56 | 76.9 ; 80.2 |
            | cache | agent | 5.0/8.7/27.4/206/268 ; 4.4/8.0/30.3/194/317 | 9.16 ; 10.76 | 52.6 ; 53.6 |
            | slo_check+transfer | chat | 4.1/8.4/36.5/304/402 ; 4.3/8.4/27.2/313/408 | 1.41 ; 1.46 | 47.8 ; 52.2 |
            | slo_check+transfer | agent | 5.0/8.6/24.6/88.7/169 ; 5.4/8.3/22.2/79.9/192 | 10.10 ; 10.32 | 58.2 ; 57.9 |
            | slo_check_no_transfer | chat | 4.0/8.4/29.6/310/402 ; 4.5/8.6/26.6/312/405 | 1.61 ; 1.22 | 51.9 ; 51.2 |
            | slo_check_no_transfer | agent | 6.3/8.1/20.4/77.3/201 ; 6.4/11.3/21.7/88.8/216 | 10.14 ; 10.13 | 56.9 ; 50.0 |
            *192 x 16K prefixes, pre-fix router (`mn_manyprefix_transfer_8gpu_20261017.sh`; SLO-check arms rep 1 only)* -- [figure](pd_colocation/figures_20261017/h20x8_manyprefix_oldrouter.png), [data](pd_colocation/figures_20261017/h20x8_manyprefix_oldrouter.csv)
            | arm | stream | tok/s/user p1 / p10 / med / p90 / p99 | mean TTFT s | mean TPOT ms |
            |---|---|---|---|---|
            | cache | chat | 3.7/7.8/12.6/45.7/270 ; 3.5/7.5/12.2/28.8/221 | 28.08 ; 27.92 | 84.3 ; 88.1 |
            | cache | agent | 4.8/8.2/11.7/222/353 ; 4.3/8.1/11.1/252/346 | 30.42 ; 28.54 | 67.7 ; 73.9 |
            | slo_check+transfer | chat | 3.2/5.7/10.3/11.9/14.5 | 233.54 | 115.0 |
            | slo_check+transfer | agent | 5.3/7.4/12.4/258/340 | 84.79 | 67.9 |
            | slo_check_no_transfer | chat | 2.9/5.3/9.4/10.9/12.5 | 73.42 | 125.1 |
            | slo_check_no_transfer | agent | 4.7/8.2/10.7/276/343 | 48.67 | 66.9 |
      - [x] **Overload A/B with the cache fallback (2026-10-17 07:4x-08:4x; `bash tasks/pd_colocation/mn_overload_fallback_8gpu_20261017.sh`;
            192 x 16K shared-prefix chat at 16 req/s + agent C=16, pool on, timeout prefetch, `HC_WRITE=write_through`,
            clean grid; SLO check + transfer term + resident ITL check + `--no-feasible-fallback cache`; x2; all 4
            arms 8 workers 0 tracebacks / 0 fallbacks, prefill hit 87.7-89.5%).** Two-rep means: cache affinity
            agent 128 tok/s / 28.3 s / 40.6 ms, chat 1 737 / 27.5 s / 88.1 ms; SLO check agent 129 / 29.5 s / 40.5,
            chat 1 773 / 27.6 s / 92.7. **Parity:** the cache fallback removes the collapse (chat TTFT 73-234 s with
            the most-slack fallback) but gains nothing -- 95% of decisions (3 749 / 3 692 of 3 940) had no feasible
            node, almost all on this request's own TTFT (22-24K rejections), and went to the top-hit node, i.e.
            cache affinity. Under full overload no placement meets the SLO; the lever there is admission (what to
            delay), not placement. **Why pool pulls do not land (found 08:5x):** the learned pull realisation fell
            to 0.0 in both reps because the pool is full. The mooncake master holds 178-181 of 192 GB (92.7-94.4%)
            and evicts only above its 0.95 watermark, so workers' BatchPut fails "due to insufficient space" (one
            worker: 14 failed writes, 1 789 pages). With `write_through` every computed prefix is written --
            including ~96 agent contexts of ~50K tokens -- so the working set (192 x 16K shared prefixes ~3.1M
            tokens plus agent contexts) overflows the ~3.35M-token pool and the shared prefixes are evicted or
            never stored. The router's mirror assumed every computed prefix was in the pool; its realisation EWMA
            correctly learned that pulls do not land and switched its pull predictions off. Next: keep sglang's
            default `write_through_selective` (a prefix is written after its second hit -- a popularity filter
            that keeps shared prefixes and skips one-off agent contexts), mirror that rule in the router, and lower
            the master's eviction watermark (0.95 -> 0.85) so puts find space.
            **Selective-pool A/B, rep 1 (`mn_selective_pool_8gpu_20261017.sh`, 09:xx):** the pool fix works -- 0
            failed page writes and 0 "insufficient space" puts on the workers checked, pool at 81% of 192 GB (was
            93-94%) -- and the three arms sit within noise (agent 124 / 129 / 130 tok/s, chat TTFT 26.6 / 26.2 /
            27.5 s for cache / SLO check ignoring the pool / SLO check + transfer). Pulls still barely land: 48 of
            3 940 decisions predicted one, 7 989 tokens predicted vs 2 157 pulled on average (27%). A second flaw,
            in my learner: the realisation EWMA was learned from the SCALED prediction, so once it fell, scaled
            predictions dropped below the 256-token prefetch threshold, no sample was taken, and it stayed at 0.0.
            Fixed: decisions log the unscaled `pull_raw` and the EWMA learns pulled / raw (unit check: from 0.0, one
            full pull lifts it to 0.20). Rep 2 runs on the unfixed router, consistent with rep 1.
            **Both reps (`bash tasks/pd_colocation/mn_selective_pool_8gpu_20261017.sh`; all 6 arms 8 workers 0
            tracebacks / 0 fallbacks, prefill hit 88.2-89.6%):** the three arms tie within noise -- agent 124 / 134
            (cache), 129 / 129 (SLO check ignoring the pool), 130 / 129 tok/s (+ transfer); chat TTFT 26.6 / 28.1,
            26.2 / 27.4, 27.5 / 27.8 s. At 16 chat req/s on this mix the cluster is past capacity (95% of decisions
            have no feasible node, see the overload A/B), so placement -- and with it the transfer term -- has
            nothing to choose between. The transfer test needs a load where feasible non-holders exist; next run
            at a lower chat rate with the fixed learner (`mn_selective_pool_r8_8gpu_20261017.sh`, chat 8 req/s,
            running). Its rep 1 shows the learner no longer locks (pull realisation 1.0 -> 0.0 -> 1.0 range, last
            0.03), but pulls still mostly fail: 111 decisions predicted ~21.2K tokens, ~3.8K landed (18%), with the
            pool 84% full and no failed writes. The workers log "HiCache storage prefetch reissue cap reached ...
            attempts=8; the request is admitted without further L3 lookups" for 107 requests on one pod -- nearly
            every request routed for a pull retried its L3 prefetch 8 times and was admitted without it. sglang
            re-arms a prefetch when it is rate-limited, when the host pool has no room to stage the pages
            (`host_capacity`; the host pool is capped at 48 GB per worker), or when the L3 hit is below the
            256-token threshold; those reasons reach only HiCache's storage metrics. Diagnostic queued:
            `mn_prefetch_diag_8gpu_20261017.sh` (the transfer arm once with `--enable-metrics`, every worker's
            `/metrics` saved before teardown; new knobs `WORKER_EXTRA`, `SCRAPE_METRICS`).
            **Chat 8 req/s A/B, both reps (`bash tasks/pd_colocation/mn_selective_pool_r8_8gpu_20261017.sh`):** cache
            affinity is best on every agent metric -- agent 169 / 180 tok/s and TTFT 19.2 / 18.5 s vs the SLO check
            ignoring the pool 146 / 160, 20.4 / 19.1 s, and with the transfer term 154 / 148, 19.3 / 19.6 s; chat TTFT
            6.6-7.3 s in every arm. Below capacity the SLO check moves requests off holders and each move costs a
            16K recompute (or a pull that does not land), so it does not pay here; the transfer term recovers part of
            the loss in rep 1 only. Spread ([figure](pd_colocation/figures_20261017/h20x8_manyprefix_r8_selective_pool.png),
            [data](pd_colocation/figures_20261017/h20x8_manyprefix_r8_selective_pool.csv)):
            | arm | stream | tok/s/user p1 / p10 / med / p90 / p99 (rep 1 ; rep 2) | out tok/s | mean TTFT s | mean TPOT ms |
            |---|---|---|---|---|---|
            | cache | chat | 4.7/9.7/190/394/432 ; 4.4/9.6/190/396/433 | 1022 ; 1022 | 6.9 ; 6.6 | 33.4 ; 33.6 |
            | cache | agent | 5.4/7.9/19.9/205/280 ; 5.6/8.2/27.5/207/302 | 169 ; 180 | 19.2 ; 18.5 | 58.5 ; 56.2 |
            | slo_check_no_transfer | chat | 4.8/9.5/130/386/428 ; 4.9/9.7/162/391/429 | 1022 ; 1022 | 7.2 ; 7.2 | 37.6 ; 36.9 |
            | slo_check_no_transfer | agent | 4.4/7.3/12.5/101/223 ; 5.1/8.2/14.6/74.7/165 | 146 ; 160 | 20.4 ; 19.1 | 72.7 ; 64.4 |
            | slo_check+transfer | chat | 4.5/9.5/167/392/429 ; 4.5/9.2/143/384/426 | 1022 ; 1021 | 7.0 ; 7.3 | 37.4 ; 40.3 |
            | slo_check+transfer | agent | 5.1/8.0/16.4/83.5/154 ; 5.0/8.1/13.3/78.4/184 | 154 ; 148 | 19.3 ; 19.6 | 64.8 ; 69.6 |
            **Prefetch diagnostic (`bash tasks/pd_colocation/mn_prefetch_diag_8gpu_20261017.sh`, `--enable-metrics`, 6
            of 8 workers' `/metrics` saved):** the pool HAS the pages -- `storage_prefetch_hit_tokens_total` 796K --
            but only 349K were prefetched (44%). Every deferred and unfulfilled token is `host_capacity`: deferred
            553K, unfulfilled 608K, against 399 `below_threshold` and 0 for every other reason (rate limit, anchor
            lost, device capacity, storage transfer). The host pool is full -- `hicache_host_used_tokens` 5 020 161 of
            5 022 324 -- because write-through backs every hot node up to host, and sglang defers L3 staging for lack
            of host pages rather than evicting L2 to make room. So the pool pull fails in the worker's host tier,
            not in the pool, the network or the router. Fix under test: `HC_HOST_MODE=buffer_only`
            (`--hicache-host-memory-mode buffer_only`: host memory is only a staging buffer for L3, no L2 cache),
            queued as `mn_bufferonly_8gpu_20261017.sh` (192-prefix at chat 8 req/s, cache affinity vs SLO check +
            transfer, x2, spread figure) after the margin A/B.
            **Buffer-only A/B (`bash tasks/pd_colocation/mn_bufferonly_8gpu_20261017.sh`, x2; all 4 arms 8 workers 0
            tracebacks / 0 fallbacks; workers log `hicache_host_memory_mode: buffer_only`):** pulls land more often --
            7 885 of 20 661 predicted tokens (38%, was 18%), realisation EWMA ending 0.4 -- but the workers still log
            58-80 prefetches hitting the 8-reissue cap each, and the transfer arm still loses to cache affinity: agent
            144 / 151 vs 162 / 169 tok/s, TTFT 21.1 / 20.8 vs 19.5 / 18.3 s, ITL 42.7 / 42.5 vs 35.0 / 38.1 ms; chat
            TTFT 8.1 / 8.0 vs 7.9 / 7.8 s. Buffer-only also costs cache affinity itself against the default host
            cache (agent 162-169 vs 169-180 tok/s, chat TTFT 7.8-7.9 vs 6.6-6.9 s): it gives up the L2 hits. **Reading
            for point 2 on H20:** moving a request to a non-holder and pulling its prefix through the pool does not
            beat keeping it on the holder at this load, with either host mode. Spread
            ([figure](pd_colocation/figures_20261017/h20x8_manyprefix_r8_bufferonly.png),
            [data](pd_colocation/figures_20261017/h20x8_manyprefix_r8_bufferonly.csv)):
            | arm | stream | tok/s/user p1 / p10 / med / p90 / p99 (rep 1 ; rep 2) | out tok/s | mean TTFT s | mean TPOT ms |
            |---|---|---|---|---|---|
            | cache | chat | 4.2/8.8/157/380/420 ; 3.7/8.4/161/379/420 | 1022 ; 1022 | 7.9 ; 7.8 | 38.9 ; 42.0 |
            | cache | agent | 5.0/7.2/13.2/164/250 ; 4.2/7.4/15.3/129/285 | 162 ; 169 | 19.5 ; 18.3 | 65.7 ; 66.8 |
            | slo_check+transfer | chat | 4.2/8.7/107/363/411 ; 3.9/8.5/117/373/418 | 1021 ; 1020 | 8.1 ; 8.0 | 44.7 ; 44.5 |
            | slo_check+transfer | agent | 4.2/7.4/14.5/75.7/171 ; 4.4/7.5/14.4/89.3/212 | 144 ; 151 | 21.1 ; 20.8 | 74.6 ; 72.5 |
            Spread of the 16 req/s run
            ([figure](pd_colocation/figures_20261017/h20x8_manyprefix_selective_pool.png),
            [data](pd_colocation/figures_20261017/h20x8_manyprefix_selective_pool.csv)):
            | arm | stream | tok/s/user p1 / p10 / med / p90 / p99 (rep 1 ; rep 2) | out tok/s | mean TTFT s | mean TPOT ms |
            |---|---|---|---|---|---|
            | cache | chat | 3.5/7.6/12.7/45.3/319 ; 3.6/7.4/12.8/35.7/311 | 1673 ; 1823 | 26.6 ; 28.1 | 84.5 ; 85.7 |
            | cache | agent | 5.8/8.2/11.3/278/345 ; 5.0/8.1/12.2/134/332 | 124 ; 134 | 29.8 ; 28.2 | 67.8 ; 67.5 |
            | slo_check_no_transfer | chat | 3.6/7.3/12.0/17.7/43.9 ; 3.4/7.1/11.8/18.8/46.8 | 1892 ; 1885 | 26.2 ; 27.4 | 93.8 ; 94.8 |
            | slo_check_no_transfer | agent | 5.1/8.0/11.2/236/339 ; 4.5/8.2/11.0/271/353 | 129 ; 129 | 28.0 ; 27.5 | 72.8 ; 76.8 |
            | slo_check+transfer | chat | 3.6/7.2/11.8/18.0/52.8 ; 3.5/7.0/12.0/19.0/40.2 | 1857 ; 1802 | 27.5 ; 27.8 | 94.2 ; 94.8 |
            | slo_check+transfer | agent | 3.9/7.8/11.6/255/342 ; 4.9/8.3/11.3/291/352 | 130 ; 129 | 28.6 ; 29.0 | 74.5 ; 71.0 |
            Spread ([figure](pd_colocation/figures_20261017/h20x8_manyprefix_overload_fallback.png),
            [data](pd_colocation/figures_20261017/h20x8_manyprefix_overload_fallback.csv)):
            | arm | stream | tok/s/user p1 / p10 / med / p90 / p99 (rep 1 ; rep 2) | out tok/s | mean TTFT s | mean TPOT ms |
            |---|---|---|---|---|---|
            | cache | chat | 3.4/7.6/13.1/59.0/326 ; 3.5/7.0/11.9/20.1/200 | 1620 ; 1855 | 26.9 ; 28.1 | 82.7 ; 93.2 |
            | cache | agent | 4.8/7.9/11.8/269/352 ; 4.7/7.7/11.7/156/329 | 122 ; 135 | 29.8 ; 26.7 | 67.9 ; 73.1 |
            | slo_check+cache_fallback | chat | 3.6/7.2/12.2/19.9/44.3 ; 3.5/7.2/11.9/21.5/56.0 | 1729 ; 1817 | 27.2 ; 28.0 | 92.4 ; 92.8 |
            | slo_check+cache_fallback | agent | 4.7/8.4/11.7/245/350 ; 5.1/7.8/11.3/299/353 | 128 ; 130 | 28.8 ; 30.1 | 71.7 ; 71.2 |
      - [x] **Margin / quantile choice from the measured error distribution (2026-10-17).** Measured / predicted
            TTFT over the 4 clean-grid hot-mix SLO-check runs (13 200 requests; router logs `ttft_pred`, `ttft_obs`):
            | class | node | n | p10 | p50 | p80 | p90 | p99 |
            |---|---|---|---|---|---|---|---|
            | chat (< 24K tokens) | feasible | 9 956 | 0.60 | 1.09 | 1.57 | 2.25 | 6.61 |
            | chat | no feasible node | 2 852 | 0.46 | 1.12 | 2.29 | 3.66 | 10.98 |
            | agent 24-48K | feasible | 119 | 0.07 | 1.04 | 1.34 | 2.36 | 4.69 |
            | agent >= 48K | feasible | 126 | 0.06 | 1.05 | 1.58 | 2.34 | 12.86 |
            The clean-grid predictor is unbiased in the median (1.04-1.12) but wide, and the spread differs by
            class, so with no margin a request judged feasible misses its deadline about half the time.
            Implemented (`router.py --ttft-quantile q`): a conformal-style margin -- feasibility uses predicted TTFT
            times the q-quantile of measured / predicted TTFT over a sliding window (500 finished requests) per
            class (`--long-prompt-tokens 24000` splits chat- from agent-length prompts); 1.0 until 50 samples.
            Unit check: q = 0.8 over lognormal(0.08, 0.5) ratios gives 1.67.
            **Measured (`bash tasks/pd_colocation/mn_ttft_margin_8gpu_20261017.sh`, hot mix, x2; all 4 arms 8 workers
            0 tracebacks / 0 fallbacks, prefill hit 92.8-93.0%): no effect beyond noise.** Two-rep means q = 0 vs
            0.8: agent 186 vs 180 tok/s, TTFT 16.2 vs 16.4 s, ITL 31.4 vs 31.6 ms; chat 2 016 vs 1 971 tok/s, TTFT
            1.80 vs 1.81 s, ITL 36.2 vs 35.6 ms. The margin binds -- own-TTFT rejections 2 417 / 2 688 -> 2 932 /
            2 941 -- but reps differ more than arms (q = 0 moved 1 215 vs 2 096 requests; agent TTFT 17.9 vs 14.4 s),
            so a quantile margin does not change outcomes on this mix: the rejected nodes were mostly ones the
            other checks already rejected, and the fallback (top-hit node) decides the infeasible 23-29%. Kept off
            by default. Spread ([figure](pd_colocation/figures_20261017/h20x8_hot_ttft_margin.png),
            [data](pd_colocation/figures_20261017/h20x8_hot_ttft_margin.csv)):
            | arm | stream | tok/s/user p1 / p10 / med / p90 / p99 (rep 1 ; rep 2) | out tok/s | mean TTFT s | mean TPOT ms |
            |---|---|---|---|---|---|
            | q0 | chat | 5.2/11.2/77.2/268/387 ; 5.3/11.1/46.0/231/368 | 2017 ; 2016 | 2.0 ; 1.6 | 34.6 ; 37.7 |
            | q0 | agent | 6.1/8.1/48.5/164/342 ; 5.0/7.4/27.9/134/192 | 168 ; 204 | 17.9 ; 14.4 | 47.1 ; 59.6 |
            | q0.8 | chat | 4.8/10.3/66.7/265/380 ; 5.5/12.0/70.9/213/361 | 1925 ; 2017 | 2.1 ; 1.5 | 39.3 ; 31.7 |
            | q0.8 | agent | 6.6/7.8/57.9/190/332 ; 6.1/7.3/35.4/135/206 | 170 ; 189 | 17.5 ; 15.4 | 45.8 ; 52.8 |
      - [ ] A/B: cache affinity | current SLO check | + step model | + transfer term | + margin, on
            goodput / SLO attainment at the 8-GPU H20 points (agent C=16/32, chat 8 req/s)
- [ ] **5. THE SUPERSET CLAIM: adapt to any workload the baselines are each tuned for.**
      Added 2026-10-09 on the user's correction; **this is the capacity/throughput axis and
      nothing measured so far tests it.**
      - [x] **Partial evidence on 8 H20s (2026-10-17): one knob spans stock colocation and PD multiplexing.** With
        backlog-triggered sharing (`PDMUX_SHARE=N`), N=0 shares the whole GPU for every prefill batch (stock-colocation
        behaviour) and N=infinity is pure pdmux. Measured at chat 16 req/s, agent C=16, agent TTFT / ITL: N=16K 6.05 s
        / 21.8 ms, 32K 5.91 / 22.0, 48K 8.44 / 18.8 (2-rep mean), 64K 12.14 / 15.9, 128K 13.57 / 15.1, pure pdmux 13.97 /
        15.1; stock colocation (TP=1) 7.18 / 27.9. Every N <= 48K beats stock colocation on agent ITL, and N <= 32K on
        TTFT as well; N is set per deployment here, not yet per step by the EDF predictor -- the per-step choice (and
        the dominance the claim needs on the capacity axis) is still to be shown.
      - [x] **Per-step choice from EDF deadlines (2026-10-17, `apply_pdmux_deadline_share.py`,
        `PDMUX_SHARE_DEADLINE=1`):** no threshold; a split-prefill batch shares the whole GPU only when the waiting queue
        in deadline order would miss a TTFT deadline at the split's prefill rate (calibrated 5019 tok/s x prefill SM
        share). 8 H20s, agent C=16, 8 workers, 0 tracebacks / 0 fallbacks, sharing on ~6-11% of batches. Agent tput /
        TTFT / ITL ; chat TTFT / ITL: 16 req/s 304 / 5.96 s / 21.3 ms ; 0.91 s / 20.2 ms; 32 req/s 232 / 6.44 s / 37.0 ms
        ; 1.40 s / 50.4 ms. vs stock colocation without speculation: 16 req/s agent +14% / -17% / -24%, chat TTFT -67%
        / ITL -26%; 32 req/s agent +7% / -13% / -16%, chat +12% / -50% / -27% -- **better on every metric at both
        rates with no tuned knob, and the best chat TTFT of any H20 arm**, but not by 30%. vs stock + MTP: all clear
        except agent ITL at 32 req/s (-12%). The deadline rule only weighs TTFT, so it lands on the TTFT side of the
        frontier; an ITL term (resident rows' step budget) is the missing half for the per-step operating point.
        `POINTS="16:0 32:0" PDMUX_SHARE_DEADLINE=1 bash tasks/pd_colocation/mn_pdmux_share_8gpu_20261017.sh`.
      - [x] **With the ITL term (2026-10-17, `apply_pdmux_itl_budget.py`, `PDMUX_ITL_BUDGET=24`):** the decode loop keeps
        an EWMA of the decode-step interval; a TTFT-late batch shares only while that EWMA is under 24 ms. 8 workers, 0
        tracebacks / 0 fallbacks, sharing held back on a few batches per worker. Agent tput / TTFT / ITL ; chat:
        16 req/s 281 / 9.28 s / 17.8 ms ; 3223 / 1.99 s / 16.1 ms; 32 req/s 212 / 13.87 s / 22.2 ms ; 5670 / 6.29 s /
        30.0 ms. **vs stock colocation + MTP all 12 clear with wide margins: 16 req/s agent +73% / -41% / -65%, chat +46%
        / -88% / -88%; 32 req/s agent +45% / -39% / -47%, chat +64% / -79% / -60% (smallest margin -39%; single rep).**
        vs stock colocation without speculation: ITL -36..-57% but agent TTFT +29% / +88% -- the ITL term moves the
        operating point to the ITL side, and against that baseline TTFT is bounded by the ~5.9 s compute floor.
        **Repeat 2 (8 workers, 0 tracebacks / 0 fallbacks): 16 req/s agent 281 / 7.99 s / 19.1 ms, chat 1.88 s / 18.4
        ms; 32 req/s agent 221 / 13.56 s / 23.2 ms, chat 5599 / 6.92 s / 30.5 ms. Two-repeat means vs stock + MTP: 16
        req/s agent +73% / -45% / -64%, chat +46% / -89% / -87%; 32 req/s agent +48% / -40% / -46%, chat +62% / -78% /
        -60% -- every per-repeat value clears, smallest -39.1%.** **Output gate on one H20 (`gate_h20_pdmux_share.sh`):
        PASS** -- LPT_PDMUX_EDF (this config: pdmux + EDF uncached deadlines + deadline / ITL-budget sharing, divisor 30)
        vs LPT short 1 / long 11, identical to the pdmux null case; stock repeat floor short 118 / long 52; 0 tracebacks
        / 0 fallbacks, sharing, EDF and pdmux active.
        `POINTS="16:0 32:0" PDMUX_SHARE_DEADLINE=1 PDMUX_ITL_BUDGET=24 [REP_SUFFIX=_rep2] bash tasks/pd_colocation/mn_pdmux_share_8gpu_20261017.sh`.

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
            **User 2026-10-09: must be better over BOTH**, not merely over whichever of the two
            is weaker at a given mixture. Each is optimal for some mixture, so "better over both"
            is the whole content of the superset claim at single-node scale.
      - [ ] **5b. MULTI-NODE baselines** — vs **PD multiplex** and **PD disagg (1P3D and 2P2D)**
            at equal total GPU count. Per the standing authorization: **H20**, trial named
            **vibecuda**, <= 16 GPUs (4 nodes x 4, or <= 8 nodes x 2).
            **User 2026-10-09: we must beat BOTH, and beating PD DISAGG is the main argument.**
            PD multiplex is not designed for multi-node, so a win there should come easily and
            carries correspondingly less weight -- do not present it as the headline.
            **User 2026-10-09: H20 is gated on ALL kernel items being closed first** (K1-K5 in
            `debug_kernel_server_perf_4_mode.md`). Do not allocate multi-node capacity before
            then.
      - [ ] **5c. Metric is GOODPUT under per-class SLOs, not raw tok/s.** A TTFT target for
            interactive chat and a TPOT/ITL target for agentic streams; a request counts only if
            it meets its OWN class's SLO. Raw throughput is what made the colocated-vs-colocated
            kernel A/B look flat and is the wrong number for a capacity claim.
            **User 2026-10-09: DERIVE the targets from the common settings of the papers already
            given, do not invent them.** Primary source is `arxiv 2504.08784`; note its arXiv PDF
            does not text-extract (compressed streams only), so use the ar5iv/HTML rendering.
            Cross-check against `arxiv 2504.14489v2` (PD multiplex) and ThunderAgent, and record
            which paper each number came from next to the number.
      - [ ] **5d. STATIC sweep: mixture ratio x request rate.** Prefill-heavy -> decode-heavy at
            several rates; report the **capacity frontier** (max sustainable rate per mixture),
            not a single operating point. This is the test of prediction (a): the best baseline
            changes across the sweep, and we must track it everywhere. Mixtures blended from the
            datasets already in use (TraceLab agentic replay + chat + math), with the blend
            recorded so it is reproducible.
      - [x] **5e. DYNAMIC: load spike (2026-10-20, 8 H20s, text fan-out C=32 + ShareGPT chat 8 req/s; at t = 300 s a
            second chat stream adds 16 req/s for 240 s -- chat load x3; `mn_spike_8gpu_20261020.sh`, scored by
            `tasks/perf_model/spike_windows.py`; all arms 0 tracebacks / 0 fallbacks).** Agent attainment (TTFT <= 10 s
            and TPOT <= 80 ms) per window pre / spike / first 60 s of spike / post; chat attainment (0.5 s + 20 ms)
            base stream / spike stream; rep 1 | rep 2:
            | arm | agent attainment | agent goodput in spike | chat base / spike |
            |---|---|---|---|
            | ours (SLO-checked hit-first router, pool, EDF) | 96.4 / 99.4 / 100 / 99.1% \| 96.4 / 99.8 / 100 / 98.1% | 554 \| 546 tok/s | 90.4 / 91.0% \| 89.8 / 87.9% |
            | PD disagg 2P2D | 24.5 / 36.0 / 50.9 / 44.9% \| 32.0 / 39.3 / 57.3 / 52.8% | 60 \| 59 tok/s | 21.5 / 20.2% \| 22.3 / 19.1% |
            **Ours absorbs a 3x chat spike with no attainment loss** (agent 99-100% through the spike and its first
            minute; the spike stream's own chat attainment 88-91%, TTFT 0.13-0.14 s); disagg's two prefill nodes queue
            both tenants for 14-31 s TTFT in every window. The pre-spike window is lowest for both arms (start-up and
            cold prefixes).
            **Mixture shift (2026-10-20, same cluster and base load; ~300 s in, a second fan-out driver adds 32 agent
            programs on the same task prefixes for 240 s -- agent load x2, chat unchanged; `FO_SPIKE=300:32:240`,
            `mn_shift_8gpu_20261020.sh`; all arms clean).** Base agents pre / shift / first 60 s / post, the added
            programs over their run, chat over the run; rep 1 | rep 2:
            | arm | base agent attainment | added agents | agent goodput in shift (base + added) | chat attainment |
            |---|---|---|---|---|
            | ours | 96.4 / 99.3 / 100 / 98.7% \| 96.1 / 98.7 / 97.3 / 97.4% | 99.6% \| 99.0% | 373 + 345 \| 393 + 393 tok/s | 83.5% \| 83.7% |
            | PD disagg 2P2D | 26.9 / 31.0 / 26.5 / 61.0% \| 27.5 / 21.5 / 20.6 / 54.6% | 28.1% \| 28.4% | 38 + 29 \| 27 + 34 tok/s | 22.2% \| 18.2% |
            **Ours takes doubled agent load at 97-100% attainment and roughly doubles agent goodput during the shift
            (718-786 tok/s); agent TTFT rises from ~0.9 s to ~1.5-1.9 s.** Chat attainment drops from ~90% (no shift) to
            ~84%: chats decode beside twice the agent prefill. Disagg serves 61-67 tok/s of agent goodput during the shift
            at 21-31% attainment, TTFT 26-49 s. sglang 6e4bcbd77; the load-spike rows above are sglang `ff6a90041`.
      - [ ] **5e (original item). DYNAMIC: mid-run mixture shift and load spike.** The core of the argument and
            the test of prediction (b): step-change the mixture (prefill share down, decode share
            up) and inject a rate spike; measure SLO attainment **through** the transition and
            time-to-recover, with the transition visible in the time series rather than averaged
            away. A fixed split cannot re-provision; we re-balance per step. This is also where a
            null result would most damage the thesis, so it gets run most carefully.
      - [ ] **5f. Read `arxiv 2504.08784`** for its SLO definitions, goodput metric and how it
            constructs and shifts mixtures. The arxiv PDF does not text-extract (compressed
            streams only); use the ar5iv/HTML rendering.

- [ ] **4. Two-way RDMA / congestion, randomized source selection** — **CORRECTION (2026-10-19, user): the
      randomized-pull rule has not been tested.** `cache_rand` / `cache_slo_rand` (`router.py`) randomize which
      equal-hit holder *serves* the request (prefill + decode land there), not which holder a non-holder *pulls* the
      KV from. Pulled pages come from the mooncake L3 pool, whose master places them; the router never chooses a pull
      source. So the "holder load decides" conclusion below measures serving-node placement: a random equal-hit holder
      can be the busier one, and its decode batch absorbs the new request. An RDMA read of an idle source's memory does
      not compete with that source's decode compute (it costs NIC egress and memory bandwidth only), so the source's
      decode load should not matter for the pull itself. A real test needs (a) more than one copy of a hot prefix in
      the pool (replica count > 1 or per-node copies exported as sources) and (b) a reader that picks among copies
      (first / least-recently-used NIC / random) while the serving node stays fixed by the SLO check.
      - [x] **Point-4 test design (2026-10-19; measured below).** Source of a pull today: sglang's mooncake store writes every page with
        the default `ReplicateConfig` (`replica_num = 1`), and mooncake's client reads the first complete replica in
        the master's order (`FindFirstCompleteReplica`, `client_service.cpp`; no random or locality ranking). So
        there is exactly one source per page, and the "which holder" choice does not exist. Installed mooncake
        0.3.13.post1 has what a real test needs: `batch_get_replica_desc`, `create_copy_task`,
        `ReplicateConfig.preferred_segments`, `TransferEngine.batch_transfer_sync_read`. Plan, serving node fixed by
        the SLO check in every arm:
        (a) copies: after a node pulls a prefix it re-exports it under a per-node key suffix on its own segment
            (`preferred_segments = [local]`), so a hot prefix has one copy per node that has served it --
            best-effort replication of the pool itself, at the memory cost replication implies;
        (b) reader: for each page, `batch_is_exist` over the suffixes, then pick a source by policy -- `first`
            (stock order), `random` (point 4's rule), `least_inflight` (fewest outstanding reads from that source,
            counted locally, no router query);
        (c) workload: the fan-out with fast rotation on the narrowed fabric (`ONE_NIC=1`), where a new hot prefix is
            pulled by every node at once from a single segment -- the case where one source NIC's egress is shared;
        (d) metrics: per-pull bandwidth and its tail, TTFT in the first 30 s after each rotation, source spread.
      - [x] **Point 4 measured as proposed: pull-source choice, serving node fixed (2026-10-19, 8 H20s, `ONE_NIC=1`, hot
        set rotating every 45 s, natural-text fan-out, adaptive ITL SLO K = 2.5; all arms 0 tracebacks / 0 fallbacks;
        single repeats).** Copies were made and read: e.g. `random`, per worker, 25-45% of reads from other workers' copies;
        `first`, almost none (the original key covers the range, so first = original). Per-page read latency is the same on every
        source, 0.011-0.019 ms (one 0.057 outlier), so no source is congested even on one NIC.
        | KV_COPY | agent goodput 10 s + 80 ms | chat attainment 0.5 s + 20 ms | agent tput / TTFT / ITL | chat TTFT / ITL |
        |---|---|---|---|---|
        | off (one source per page) | **531 tok/s** | **91.1%** | 543.7 / 0.75 s / 13.4 ms | 346 ms / 10.4 ms |
        | first | 492 | 88.8% | 514.3 / 0.89 s / 15.9 ms | 311 ms / 11.7 ms |
        | random (point 4's rule) | 477 | 87.7% | 502.5 / 0.85 s / 16.9 ms | 409 ms / 12.6 ms |
        | least_latency | 487 | 89.0% | 512.0 / 0.85 s / 16.4 ms | 321 ms / 11.6 ms |
        (`off` reproduces the unpatched 10-19 run, 533 tok/s / 91.8%.) **Among sources, the choice is neutral** (first /
        random / least_latency within 3%, random lowest). **Making copies costs 7-10% agent goodput**: re-exporting
        every pulled page (25K-204K pages per worker) runs on the prefetch path. The rule's premise, a congested source,
        does not exist at <= 16 H20s even with one NIC, consistent with the serving-node result. Point 4's congestion
        argument therefore rests on scale beyond the GPU cap; within it, one source per page is best.
        `bash tasks/pd_colocation/mn_kv_copy_source_20261019.sh` (patch `apply_kv_copy_source.py` on every pod's tree,
        `KV_COPY` knob); scoring `tasks/perf_model/fanout_per_tenant.py`. sglang `881bbc492`.
      Original record: No congestion exists to relieve at 16 GPUs on 4
      hosts: random tie-break among near-equal hits is neutral, the L3 store moved ~236 GB without
      measurable contention, and disagg's 8 prefill GPUs replicate 4 hot prefixes to 82% hit. The
      store also COSTS ~19% chat ITL once the SLO check spreads load. Sub-items:
      - [ ] randomized pull among near-equal-hit nodes (`cache_rand`): neutral at 16 GPUs -- tested serving-node
            choice, not pull-source choice (correction above)
      - [ ] the same on a NARROWED fabric (`ONE_NIC=1`; serving-node choice, see correction) -- **randomized selection LOSES even under
            deliberate contention.** Measured below; this is a negative result for the proposed
            algorithm, in the regime built to favour it.
      - [ ] **Randomized holder choice (serving node, not pull source; see correction) on the fan-out workload, narrowed fabric + fast rotation (2026-10-19; 8 H20s,
            `ONE_NIC=1`, hot set rotating every 45 s instead of 180 s, natural text, adaptive ITL SLO K = 2.5; both arms
            clean; `bash /tmp/h20/fanout_rand.sh`).** Load-ordered within a hit bucket (`coloc_slo_edf_hc`) vs random
            within it (`coloc_slo_rand_edf_hc`): agent 545 vs 526 tok/s, TTFT 0.77 vs 0.81 s, TPOT 13.3 vs 15.3 ms;
            goodput 531 vs 506 tok/s; chat TTFT 157 vs 364 ms, ITL 10.0 vs 11.3 ms; new-copy routings 79 vs 91.
            **Random loses again (-4.7% goodput, +15% TPOT, +132% chat TTFT), now on the workload built for
            replication.** The load-ordered arm loses only ~2% against the full fabric with 180 s rotation (555 tok/s,
            0.75 s), so at <= 16 H20s the links are not the constraint even at 4x less bandwidth and 4x faster churn;
            holder load is -- for choosing where a request runs. It does not test randomizing the pull source.
      - [ ] congestion modelled from router-observed past link latency (the alternative in the
            goal). NOT built, and the measurements argue against building it here: with no
            contention to observe there is nothing to fit, and a per-request router query would add
            latency to every decision. Revisit only if the narrowed-fabric run shows an effect.
      - [x] **Scaling with node count, inside the cap: 2 vs 4 pods (4 vs 8 H20s) at fixed per-GPU load (2026-10-18,
            `mn_scaling_20261018.sh`; agent C = 2 / GPU, chat 2 req/s / GPU, 6 sessions / GPU; every arm 0 tracebacks /
            0 fallbacks on every pod, `mn_audit.py`).** Agent tput / TTFT / ITL ; chat tput / TTFT / ITL ; prefill hit:
            | arm | pods | agent | chat | hit |
            |---|---|---|---|---|
            | ours (pdmux + EDF + deadline / ITL-budget sharing) | 2 | 123 / 6.73 s / 25.2 ms | 1645 / 0.64 s / 14.0 ms | 0.342 |
            | ours | 4 | 281 / 8.63 s / 18.5 ms | 3223 / 1.94 s / 17.3 ms | 0.395 |
            | stock colocation + MTP | 2 | 48 / 16.77 s / 77.0 ms | 1656 / 10.27 s / 77.8 ms | 0.339 |
            | stock colocation + MTP | 4 | 162 / 15.68 s / 50.9 ms | 2214 / 16.84 s / 129.6 ms | 0.394 |
            | stock PD disagg + MTP (1P1D / 2P2D) | 2 | 24 / 69.9 s / 55.8 ms | 1048 / 93.3 s / 48.2 ms | 0.125 |
            | stock PD disagg + MTP | 4 | 108 / 43.6 s / 11.1 ms | 2000 / 45.3 s / 5.5 ms | 0.055 |
            (4-pod ours / stock-colocation rows: the 2026-10-17 runs at the same per-GPU load, ours the two-repeat mean.)
            **Cache locality is the scaling signal: from 2 to 4 pods colocation's prefill hit rises (0.34 -> 0.39) while
            disaggregation's halves (0.125 -> 0.055)** -- more prefill nodes dilute each prefix, while cache-affinity
            colocation keeps it where its session lives. Ours stays ahead of disagg at both sizes (agent throughput 5.1x
            at 2 pods, 2.6x at 4; TTFT -90% / -80%); disagg gains the most from the extra pods (its 1P1D layout is badly
            unbalanced) and has the lowest decode-only ITL (5-11 ms) at 43-93 s TTFT. Two sizes inside the 16-GPU cap
            show the direction; more sizes need more nodes.
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

## [!] WHAT TODAY'S KERNEL MEASUREMENTS TAKE AWAY FROM THE SCHEDULING ARGUMENT — 2026-10-10

Four results from `debug_kernel_server_perf_4_mode.md` bear directly on the sections below, and
**two of them cut against the argument as written**. Recorded here rather than left in the kernel
doc, because the scheduling case leans on them.

**1. The two-role fusion is NET NEGATIVE on realistic 4-mode batches.** One prefill chunk plus N
decode rows, against routing everything to FA3 (same rows, no fusion):

| decode rows | co-resident (best `sm_split`) | all-FA3 | fusion |
|---|---|---|---|
| 2 | 0.2426 | 0.2557 | +5.4% |
| 8 | 0.3665 | **0.2512** | **1.46x SLOWER** |
| 32 | 0.7371 | **0.3670** | **2.01x SLOWER** |

The +5.4% was a 2-decode-token artifact. So **"one colocated batch through one fused attention
kernel" is currently slower than one colocated batch through FA3 alone** -- the scheduling story
does not need the two-role fusion, and is actively hurt by it at realistic decode counts.

**2. Point 3's premise is INVERTED: the fused kernel makes cost modelling HARDER, not easier.**
Section 3 below argues a colocated step is predictable because it is "one batch through one
attention implementation whose cost is linear in total KV". Measured today: the **optimal
`sm_split` moves with batch composition** (128 at 2 decode rows, 96 at 8, 64 at 32) and the
penalty for getting it wrong is **not** smooth -- at dec=32 with `sm_split=128` the kernel takes
**8.68 ms against 0.367 ms at that shape's own optimum, a 23x cliff**. A predictor over
`(#prefill, #decode, sum KV)` must therefore also model the split, and the target has a 23x
discontinuity in it. That is the opposite of "a smooth low-variance function".
**Point 3 should be rewritten to claim predictability for the UNFUSED colocated step** (FA3-only,
one kernel, linear in KV -- which still holds and is still better than disagg's two-stage
pipeline), not for the fused one.

**3. The baseline is stronger than this doc assumes.** flashinfer's own persistent
`BatchAttention` does the whole mixed batch in ONE kernel and is **validated correct** here
(0 bad rows vs the fp32 reference). Our fused kernel is **1.11x-4.9x slower than it** on every
fixture. So "colocation gives us one kernel instead of two" is not a differentiator -- the
baseline already has one kernel. The differentiator has to be the *schedule*.

**4. The split-KV partial slab is O(packed_qo x num_kv_chunks)** -- ~1 GB for a single 2048-token
chunk at 8K context. At production context lengths that workspace competes with the KV cache
budget, which is a **scheduling** constraint (it reduces the cache a colocated node can hold, and
cache capacity is exactly what points 1 and 2 trade on). Not previously accounted for anywhere
below.

**Gating, unchanged in kind but worse in degree than the 2026-10-06 note below:** the kernel is
incorrect on multi-item-per-CTA batches -- i.e. a prefill chunk plus tens of decode rows, which is
every realistic 4-mode step -- in *every* configuration tested. So no serving arm, single- or
multi-node, is measurable today. The scheduling arguments below stand or fall on their own logic
and on the H20 measurements already recorded; **none of them has fresh kernel support as of
2026-10-10.**

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

## Multi-node, program concurrency, MTP everywhere (H20, 8 pods x 2 GPUs) — 2026-10-16

The 2026-10-08 runs below drove agents by a session-start rate (an axis since dropped: agent load is
program concurrency only) and ran stock FA2 without speculation. Re-run on job `1c90eac0a6848b48`
(trial 432235269, vibecuda, 8x2 H20) after A and B closed: every worker serves EAGLE3 MTP; agents are
the TraceLab 32-64K cohort (192 sessions x 2 rounds) at program concurrency AGENT_CONC, plus ShareGPT
chat at CHAT_RATE. Provisioning `tasks/pd_colocation/mn_provision_20261016.sh` (upstream 3fef129 + the
H200 serving tree's 26-file overlay + port07); driver `tasks/pd_colocation/mn_matrix_20261016.sh`.
- [x] **C=64 + chat 16 req/s (overloaded: stock chat mean TTFT 43 s, median 95 ms), all requests
  completed, 0 tracebacks / fallbacks:**
  | arm | agent tput / TTFT mean, med / TPOT / ITL / P99 ITL | chat tput / TTFT mean, med / TPOT / ITL |
  |---|---|---|
  | stock MTP, cache-affinity router | 671 / 21.9 s, 8.0 s / 47.4 / 27.2 / 360 | 3305 / 43.0 s, 95 ms / 15.1 / 13.5 |
  | **ours** (split + EDF + uncapped preemption, chunk 2048), same router | 767 / 18.0 s, 8.1 s / 29.8 / 21.5 / 149 | 3305 / 26.5 s, 82 ms / 17.3 / 13.8 |
  Ours vs stock colocation: agent tput +14%, TTFT -18%, TPOT -37%, ITL -21%, P99 ITL -59%; chat
  TTFT -38%, TPOT +15%, throughput = offered load. Mixed at this overload point; the sglang
  cache-aware-router arm failed to start (sglang_router 0.3.2 cannot bind `::`; fixed, rerun queued),
  disagg 2p2d and the lighter point (C=32, chat 8 req/s) are running.

## Multi-node measurements with the final single-GPU stack (H20) — 2026-10-16

Workload: TraceLab 32-64K agent sessions (2 rounds each) at program concurrency C (ThunderAgent) +
ShareGPT chat at a fixed request rate; EAGLE3 MTP on every worker. Arms: stock sglang colocation
(sglang_router cache_aware, stock MTP) | our cache-affinity router + stock MTP | **ours: our router +
4-mode MTP on the split + worker EDF (deadline order) + chunk preemption + load/context gate** | stock
PD disaggregation (sglang_router pd, stock MTP). Driver `tasks/pd_colocation/mn_run.py` + `mn_bench.sh`.

- [x] **16 GPUs (8 pods x 2, job 1c90eac0a6848b48, measured before the user cut H20 to 8 GPUs), C=32 +
  chat 8 req/s**, ours vs stock colocation: agent throughput +14% (929 vs 815 tok/s), TTFT -12%, ITL
  -19%; chat TTFT -51%. Disagg never started (prefill workers died on import, see below).
- [~] **8 GPUs (4 pods x 2, job bb854a10b8954223), `mn_matrix_8gpu_20261016.sh`, all reported arms 0
  tracebacks / fallbacks.** Out tok/s / mean TTFT / mean ITL / P99 ITL:
  | point | arm | agent | chat |
  |---|---|---|---|
  | C=16, chat 4/s | stock coloc | 343 / 8.3 s / 19.3 / 227 | 798 / 1697 ms / 19.9 / 219 |
  | C=16, chat 4/s | our router, stock MTP | 320 / 7.8 s / 19.5 / 173 | 798 / 1429 / 17.5 / 80 |
  | C=16, chat 4/s | **ours** | **420 / 6.9 s / 14.2 / 185** | 798 / **287** / 16.5 / 347 |
  | C=16, chat 4/s | stock PD disagg (2 prefill + 2 decode pods, clean upstream) | 137 / 34.3 s / 10.8 / 34 | 796 / 21 546 / 5.0 / 25 |
  | C=32, chat 8/s | stock coloc | 221 / 12.1 s / 97.4 / 1815 | 1461 / 7310 / 185 / 2265 |
  | ~~C=32, chat 8/s~~ | ~~our router, stock MTP~~ (router capped at 100 connections) | ~~(agent stream lost)~~ | ~~1658 / 35.1 s / 30.3 / 421~~ |
  | ~~C=32, chat 8/s~~ | ~~ours~~ (router capped at 100 connections) | ~~391 / 12.6 s / 32.2 / 315~~ | ~~1658 / 19.7 s / 28.3 / 425~~ |
  | C=32, chat 8/s | our router, stock MTP (fixed router) | 221 / 13.8 s / 95.0 / 1767 | 1433 / 10.1 s / 189 / 2294 |
  | C=32, chat 8/s | **ours** (fixed router) | **263 / 11.6 s / 74.6 / 391** | **1657 / 2.54 s / 135 / 651** |
  | C=32, chat 8/s | stock PD disagg (clean upstream) | 152 / 74.5 s / 13.6 / 51 | 1075 / 78.3 s / 6.5 / 43 |
  Ours vs stock colocation: C=16 agent throughput +22%, TTFT -17%, TPOT -59%, ITL -26%, chat TTFT -83%,
  chat ITL -17% (chat P99 ITL +58%). **C=32 (fixed router, 2026-10-16):** agent throughput +19%, TTFT -4%,
  ITL -23%, P99 ITL -78%; chat throughput +13% (serves the full offered 8 req/s; stock 1461 of ~1650
  tok/s), TTFT -65%, ITL -27%, P99 ITL -71%. Bar misses at C=32: agent TTFT (-4%) and agent/chat ITL
  (-23 / -27% vs -30%). ~~C=32 agent throughput +77%, ITL -67%, P99 ITL -83%, chat throughput +14%,
  chat ITL -85%, but agent TTFT +4% and chat mean TTFT +169% (median 85 vs 2957 ms: a starved tail).~~
  **Retracted:** measured behind our router's 100-connection cap, which queued chat inside the router
  (fewer requests on the GPUs, hence the low ITL and the starved chat tail). Rerun audit: 8 workers 0
  tracebacks / 0 fallbacks, 256-394 requests each; composition per worker FUSED_SPLIT_CALLS 6.5K-11.5K,
  MIXEDSPEC_STEPS 201-401, MIXEDSPEC_GATED >= 201, EDF_PARK firing. Command:
  `bash tasks/pd_colocation/mn_rerun_c32_cache_mtps_20261016.sh` (router.py `limit=0`). ~~The C=32 "our router, stock MTP" agent client died after warmup without output
  (silent kill; pods run at half the memory request to get scheduled).~~ **Retracted 2026-10-16:** not a
  kill. One upstream POST in our `router.py` hit EPIPE on a stale keep-alive connection (the retry caught
  only `ServerDisconnectedError`), and the concurrency replay aborts the client on any failed round.
  Also found: `router.py`'s aiohttp session used the default connector, **capped at 100 concurrent
  upstream connections**; at C=32 + chat 8 req/s ~200+ requests are in flight, so both our-router C=32
  rows above were queued inside the router -- the suspected cause of the chat starved tail. Both C=32
  our-router rows are therefore not final; router fixed (`limit=0`, keep-alive 3 s, retry on
  `ClientOSError`) and both arms rerun (`mn_rerun_c32_cache_mtps_20261016.sh`). C=16 (~2 chat in flight)
  is unaffected.
  **Ours vs stock PD disagg at C=16:** agent throughput 3.1x (420 vs 137 tok/s), agent TTFT -80%, chat
  TTFT -98.7% (0.29 vs 21.5 s); disagg keeps lower ITL (agent 10.8 vs 14.2 ms, chat 5.0 vs 16.5 ms,
  P99 34 vs 185 ms). Disagg is prefill-bound: its 2 prefill pods hit 16.6% of prompt tokens (862K /
  5.19M, `mn_audit.py`), 8 workers 0 tracebacks / 0 fallbacks, 302-344 requests each. Command:
  `bash tasks/pd_colocation/mn_disagg_rerun_8gpu_20261016.sh` (SGL_PY=up_clean, KERNEL=lpt).
  - [x] **Chat request-rate sweep at reference-list rates (2026-10-16, 23:5x):** the matrix's chat 4 / 8 req/s
    is 0.5 / 1 req/s per GPU, at or below the bottom of the papers' per-GPU rates (SLOs-Serve 1.5-4.5 per
    A100, semi-PD 11-16 per A100 for Llama3-8B ShareGPT, DistServe / Sarathi-Serve 0.5-10). Swept 1 / 2 /
    4 / 8 req/s per GPU (8-64 total; plus the matrix's 4 req/s point) at agent C=16, stock coloc | ours |
    stock PD disagg. Figure [`h20x8_chatsweep_c16.png`](pd_colocation/figures_20261016/h20x8_chatsweep_c16.png)
    (throughput vs tok/s/user: median marker, p10-p90 bar, p1-p99 whisker; SLO attainment, TTFT, TPOT vs
    chat rate; data [`h20x8_chatsweep_c16.csv`](pd_colocation/figures_20261016/h20x8_chatsweep_c16.csv)).
    Every arm completed every request; all 14 audited arms 8 workers 0 tracebacks / 0 fallbacks; ours'
    composition per worker at 8 / 16 / 32 / 64 req/s: FUSED_SPLIT_CALLS >= 3K / 3.5K / 10K / 15.5K,
    MIXEDSPEC_STEPS >= 101 / 101 / 401 / 651, EDF_PARK firing. Prefill cache hit: ours 44-68%, stock coloc
    25-39%, disagg 5-13%.
    | chat req/s (per GPU) | ours vs stock coloc: agent tput / TTFT / ITL | chat tput / TTFT / ITL |
    |---|---|---|
    | 4 (0.5) | +22% / -17% / -26% | = (offered) / -83% / -17% |
    | 8 (1) | +20% / -3% / -32% | = (offered) / -81% / -30% |
    | 16 (2) | +6% / -9% / -12% | +12% / -75% / -7% |
    | 32 (4) | +12% / -11% / -17% | +10% / -40% / -1% |
    | 64 (8) | +25% / -16% / -28% | +20% / -30% / -6% |
    **Reading.** (1) 8 H20s with this agent tenant saturate between 1 and 2 chat req/s per GPU, so the
    papers' upper per-GPU rates (their ShareGPT-only workloads on A100) are deep overload here; above 8
    req/s every arm queues (TTFT 10-144 s) and colocated chat TPOT falls again past 16 req/s because
    fewer requests decode at once. (2) Ours beats stock colocation on throughput (+6..+25%) and TTFT at
    every rate; the >=30% ITL bar is met only at 8 req/s (both tenants) and 64 req/s agent (-28%, just
    short); agent TTFT never clears -30%. (3) Stock PD disagg has the worst TTFT (21-144 s) and the
    lowest throughput at every rate, but the lowest TPOT and a far better slow-user tail at low load
    (chat p1 47 vs 4 tok/s/user at 4 req/s): its prefill pods throttle admission, so its decode pods are
    lightly loaded. Command: `bash tasks/pd_colocation/mn_chat_ratesweep_8gpu_20261016.sh`, then
    `PYTHONPATH=<matplotlib> python tasks/pd_colocation/mn_sweep_figure.py tasks/pd_colocation/figures_20261016/h20x8_chatsweep_c16`;
    audit `SID=bb854a10b8954223 NPODS=4 python tasks/pd_colocation/mn_audit.py <arm>_8g_chatsweep_c16_r<R>`.
  - [x] **Stock PD-disagg ITL/TPOT are understated by burst delivery (2026-10-16):** 17.5-29.1% of its
    requests stream every token at once at the end (ITLs 0.0 s; never on colocated workers, 0 of 1 200),
    so their decode time sits in TTFT and adds zero-gap ITLs. Excluding those requests, disagg mean ITL /
    TPOT: C=16 chat 5.5 / 6.2 ms (reported 5.0 / 5.2), agent 11.4 / 11.2 (10.8 / 8.4); C=32 chat 7.4 / 8.4
    (6.5 / 6.0), agent 14.1 / 12.6 (13.6 / 10.3). Disagg still has the lower ITL; its TPOT edge shrinks.
    `eval_curves.py` now reports `burst_frac` and excludes burst requests from tok/s/user. Per-request
    spread at chat 4 req/s, C=16 (`mn_sweep_figure.py`): chat tok/s/user p1 / median / p99 -- ours 4.3 /
    243 / 394, stock coloc 4.8 / 239 / 371, stock disagg 47 / 207 / 685: colocation's slow-user tail
    (p1 ~4-5 tok/s/user) is much worse than disagg's.
  **Stock PD disagg at C=32** (same script): saturated on prefill -- prefill hit 10.0% (1.05M / 10.5M
  tok), agent and chat mean TTFT 74.5 / 78.3 s, agent 152 tok/s, chat 1075 tok/s (offered ~1 650); 8
  workers 0 tracebacks / 0 fallbacks, 617-677 requests each. Ours (fixed router) vs disagg at C=32: agent
  throughput 1.7x (263 vs 152), agent TTFT -84%, chat throughput +54%, chat TTFT -97% (2.5 vs 78.3 s);
  disagg keeps far lower ITL (agent 14.1 vs 74.6 ms, chat 7.4 vs 135 ms, burst-excluded).
  **PD disaggregation baseline (history):** its prefill workers failed on import at C=16 -- mooncake links
  `libcudart.so.12`, the venv is cu130 -- fixed by installing `nvidia-cuda-runtime-cu12` on all pods;
  then a cross-GPU port collision (ports now claimed under a lock), the serving overlay's spec-v2 assert in
  disagg prefill (baseline now runs clean upstream), and orphan workers relaunched by surviving
  `mn_node.sh` supervisors that OOMed new prefill workers (`mn_kill.sh` now kills supervisors first).

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
  elsewhere (an SLO check or an overloaded holder, point 2).
- **Not yet shown: the congestion advantage (point 4).** At 16 GPUs on 4 hosts the store moved
  ~236 GB without measurable link contention, so random vs load-aware source choice cannot
  differ yet. That needs either more nodes than the 16-GPU cap allows, or a deliberately
  hot-prefix workload where many nodes pull one session's prefix at once. The bipartite
  disagg design forces exactly that on its few prefill nodes.
- The single failed chat request in coloc_rr_hc was a worker dropping a reused keep-alive
  connection (`ServerDisconnectedError`). The router now retries once before streaming starts
  (a8465ecc7), as the stock router does.

**The SLO check, calibrated, with worker-side EDF (point 1), 2026-10-08.** The router's
`PerfModel` was fitted on one H20 worker (`mn_run.py calibrate`, `calibrate_h20.py`): prefill
9.0K tok/s at 4-8K falling to 4.9K at 49K, so 5 531 tok/s over 16-49K, and decode ITL 5.05 ms +
0.065 ms per running sequence. The router uses the same length-scaled TTFT deadline as the
workers (2 s + 1 s per 5K tokens beyond 4 096). Workers run deadline-ordered admission with no
chunk cap (`_edf` suffix; the configuration that won on the H200). Command:
`ROUTER_ARGS="--prefill-tok-s 5531 --decode-step-ms 5.05 --decode-ms-per-seq 0.065 --ttft-slo-ms 2000"
python mn_run.py coloc_cache_edf coloc_slo_edf` (commit bf96b6cd3). 0 tracebacks and 0 fallbacks;
every request completed.

| arm | agent TTFT ms mean / P99 | chat TTFT ms mean / P99 | ITL ms agent / chat | SLO-check moves |
|---|---|---|---|---|
| coloc_cache | 1 596 / 8 371 | 101 / 1 539 | 7.2 / 6.4 | - |
| coloc_cache_edf (worker EDF) | 1 611 / 8 102 | 117 / 2 269 | 7.4 / 6.4 | - |
| **coloc_slo_edf** (SLO check + worker EDF) | **1 582** / 8 132 | 108 / **1 313** | 7.2 / 6.4 | 71 of 3 587 (2.0%) |

- **The SLO check fires once calibrated** (the uncalibrated run moved no request). It moves 2% of
  requests off their highest-hit node and trims the chat TTFT tail (P99 -15% vs affinity alone),
  with means unchanged. At this load affinity rarely pushes a worker past its SLO, so there is
  little to correct. The SLO check is a tail guard, and its value should grow with hotspots, which the
  HOT experiment below creates.

**Hot-prefix stress (points 1, 2, 4), 2026-10-08.** `HOT=1`: the chat tenant becomes 4 groups
sharing 16K-token system prompts (`generated-shared-prefix`, 512-token questions, 16/s, 3 200
requests); the agent tenant is unchanged. Command: `HOT=1 RUN_TAG=_hot ROUTER_ARGS="<calibrated>
--ttft-slo-ms 2000" python mn_run.py coloc_cache coloc_slo_edf coloc_slo_edf_hc disagg_2p2d`
(commits 498d60155, daaedef6a; the cache tie-break was already randomized). 0 tracebacks, 0
fallbacks, every request completed.

| arm | agent TTFT ms mean / P99 | chat TTFT ms mean / P99 | ITL ms agent / chat (chat P99) | prefix hit | SLO-check moves | max requests/worker |
|---|---|---|---|---|---|---|
| coloc_cache_hot | 1 673 / 8 602 | 340 / 2 263 | 7.1 / 23.4 (224) | 93.5% | - | 803 |
| coloc_slo_edf_hot | 1 605 / 8 408 | 347 / 3 290 | 7.8 / 19.1 (230) | 93.5% | 96 (2.7%) | 808 |
| coloc_slo_edf_hc_hot | 1 648 / 8 189 | 377 / 3 414 | 7.5 / **18.9 (200)** | 93.4% | 87 (2.4%) | 802 |
| disagg_2p2d_hot | 27 776 / 76 129 | 29 513 / 85 682 | 6.3 / **3.8 (30.5)** | 81.9% | - | 861 |

- **Colocation still wins TTFT by an order of magnitude** (agent -94%, chat -99%), even when the
  workload is designed to create hotspots.
- **CORRECTED 2026-10-08 (see the decode-aware SLO check below): the "SLO check is too weak" reading in
  this bullet held only while the SLO check priced TTFT alone. Pricing the decode slowdown cuts chat
  mean ITL to 8.3 ms and P99 to 44 ms.** Original reading: **Disagg wins chat ITL by 5x here.** A hot group's decode concentrates on its holder in
  colocation (~800 requests on the busiest worker), while disagg spreads decode over 8 dedicated
  GPUs. This is colocation's real weakness, and the SLO check is too weak to fix it: it moves only
  ~2.5% of requests, because the calibrated model predicts TTFT, and TTFT stays fine on a holder
  whose prefix is cached. The overload shows up as ITL, which the SLO check does not price. The SLO check
  plus store trims chat ITL 19% (23.4 -> 18.9 ms) and P99 to 200 ms, and does not close the gap.
- **What point 4 needs and does not yet have:** a SLO check that prices the decode (TPOT) slowdown a
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
decode-aware SLO check does that -- the SLO check spreads each shared-prefix group over several workers, which
both replicates the prefix and forces the spread requests to pull. New policy `cache_slo_rand`
(c87aa83b9) is `cache_slo` with the load term dropped inside a hit bucket, i.e. the goal's "pick
randomly among near-equal holders"; it differs from `cache_slo` only when holders really tie.
Running: `HOT=1 ONE_NIC=1 mn_run.py coloc_slo_edf_hc coloc_slo_rand_hc`.

**Randomized source selection under contention: measured, and it LOSES (2026-10-08).** The
isolating pair, both arms identical except the policy -- hot shared prefixes, `ONE_NIC=1`,
decode-aware SLO check (`--decode-grid`, 12 points), L3 store, worker EDF on both sides. Command:
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

**[RETRACTED 2026-10-19: these arms randomized the serving node, not the pull source; see the correction at point 4.
What follows holds for where a request runs, not for which copy a pull reads.]** **Point 4's algorithm, as proposed, is therefore refuted at this scale**: "when several nodes have
hit lengths within a threshold, pull from one at random" is dominated by "pull from the least
loaded of them". Randomness is not harmless here; it discards the signal that matters. What the
goal's underlying intuition gets RIGHT is that concentration is the risk -- but the measured cure
is the load-aware tie-break plus the decode-aware SLO check (chat ITL -65%, P99 -80%, v0.18.0), not
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

**The SLO check has to price TPOT, not TTFT (point 1, resolved 2026-10-08).** The hot-prefix holders'
overload shows up as slow decode, not slow prefill, so a TTFT-only SLO check never fired on them. Two
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

| hot-prefix arm | chat ITL mean / P99 ms | chat TTFT mean / P99 ms | agent ITL mean ms | SLO-check moves |
|---|---|---|---|---|
| coloc_cache_hot (affinity) | 23.4 / 224 | **340** / 2 263 | **7.1** | - |
| coloc_slo_edf_hot (TTFT-only SLO check) | 19.1 / 230 | 347 / 3 290 | 7.8 | 96 (2.7%) |
| **coloc_slo_edf_hot2 (decode-aware SLO check, ITL SLO 40 ms)** | **8.3 / 44.3** | 424 / 5 161 | 8.7 | **192 (5.4%)**, 20 least-bad |
| coloc_slo_edf_hc_hot2 (decode-aware SLO check + L3 store) | 9.9 / 80.0 | 468 / 6 304 | 9.8 | - |
| disagg_2p2d_hot | 3.8 / 30.5 | 29 513 / 85 682 | 6.3 | - |

**Pricing the decode cost cuts chat mean ITL 65% (23.4 -> 8.3 ms) and P99 80% (224 -> 44 ms)**,
closing most of the gap to disaggregation, for a 25% chat TTFT cost (340 -> 424 ms) and slightly
worse agent ITL. Colocation keeps a ~70x TTFT advantage. This is point 1's mechanism working as
designed: route to the cache holder first, and SLO check it only when the load it would carry is
predicted to break the SLO -- with the SLO term that the overload actually appears in.

**The L3 store does not help once the SLO check spreads the load, and costs ~19% chat mean ITL / 80%
P99** (8.3 -> 9.9 ms, 44 -> 80 ms; agent ITL 8.7 -> 9.8 ms). Its write-through traffic and
prefetches compete with decode on the same GPUs, while the moved requests already replicate the
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


## Evidence state of the four points — 2026-10-29 (supersedes the 2026-10-21 table below)

Rewritten because the 16-GPU TraceLab replay arms of 2026-10-28/29 change what each point rests on, and because
two arms that the older sections rely on are now VOID. Everything below is Qwen2.5-7B on 16 H20s (4 pods x 4
GPUs), superposed TraceLab coding replay plus ShareGPT chat, load factor alpha, both systems on FA3 unless the row
says otherwise. Commands and logs are in `debug_kernel_server_perf_4_mode.md`.

| point | measured now | still argument only |
|---|---|---|
| **1.** highest-hit-first with an EDF/SLO check | **The strongest evidence yet, on the realistic replay: against conditional disaggregation at alpha 3 we hold +40.7 points chat attainment (91.2 vs 50.5%), +29.4 agent, 2.75x agent SLO goodput (2493 vs 905) and -85% chat mean-TPOT p90** (29.8 vs 193.8 ms), both arms failed=0 at matched acceptance. Conditional disagg is the right control because it prefills short appends LOCALLY, so the remaining gap is the session history being in the wrong place. | The SLO-check *variant* of the router still loses on session-affine traces (chat 65.1-68.5%); the contribution is the cheap load tie-break, not the feasibility check. Stock PD at either load has NO valid number (see below). |
| **2.** best-effort cache migration / replication | Unchanged from 2026-10-21: the replication effect as cached fraction under the thrashing sweep (ours -3.1 points and -10.6% throughput, disagg -28.5 points and -55%), scheduler-only settings. | A full-stack rerun; migration is still never the deciding mechanism in any arm we have. |
| **3.** EDF predictor with acceptance length | **UNMEASURED, not refuted (corrected 2026-10-29): the decode KV length is a DEAD INPUT.** `dkv` is zero on all 73 770 decode steps in the traces and non-zero on 10 189 of 14 162 mixed/prefill steps, so decode step time was being predicted from the row count alone, without the dominant term. The -13% bias and 29% MAPE are the signature of that, not of an unmodelable quantity -- and on a random split the held-out median bias is +0.1%. The tracer must populate `dkv` on decode steps before the premise can be tested at all. (Superseded reading: **the premise is REFUTED.**) The step model predicts merged steps to 4.7% MAPE, but DECODE steps -- which are almost all of TPOT -- carry a **-13.0% signed bias** (29.0% MAPE) that a scalar calibration cannot remove (held-out -13.3%), and a nonlinear fit on the same features is WORSE (-19.9%). So predicted TPOT is systematically ~13% optimistic, the worst direction for admission control. `tpot_bias.py`, `fit_step_mlp_20261028.py`. | That a *better-featured* model would fix it. The missing information is likely the decode batch's KV-length distribution (only its mean is logged) and the resident tenant mix. No TPOT-accuracy claim may appear in the paper. |
| **4.** two-way RDMA, complete vs bipartite graph | **New, and it is the motivation rather than the mechanism:** the prefill-to-decode demand RATIO moves 2.2x inside a chat surge (median 38.4 against 85.1 outside) and spans 3.4-4.2x within a run with no surge at all, while re-partitioning roles takes tens of minutes (`pd_ratio_20261028.py`). Independent trace analysis finds 4.7x peak-to-mean at minute scale and 24.5x hourly within a day. So a static split is a control loop slower than its signal. | Congestion itself. The random-pull rule remains refuted at the scale we can test, and we have no per-link measurement. |

## Every 16-GPU arm re-scored in the reported metrics — 2026-10-29

Reporting changed to SLO attainment plus mean TTFT and mean TPOT with a p10-p99 spread, with no token rate (user).
Rather than restate old numbers, **all 48 surviving arm record sets were re-scored from their raw per-request
records**, so every figure below is measured in the metric it is now reported in and none is converted or
estimated. Raw lines: `tasks/paper_figures/data/rescore_means_20261029.txt`. Command:
`python tasks/perf_model/tracelab_windows.py <run dir>...` (sglang 3fef129ff; overlay at this commit).
Comparability was checked rather than assumed: every arm at a given load ran 1016-1083 s (alpha 6) or 1804-2080 s
(alpha 3) with the same chat request count (7200 / 14400) and `failed=0`.

**Against conditional disaggregation** -- the strong PD baseline, our router's implementation of Dynamo's rule, on
the same workers and tree. Two repeats of the baseline at each load:

| load | chat attainment | agent attainment | chat mean TPOT | agent mean TPOT | chat mean TTFT | agent rounds issued |
|---|---|---|---|---|---|---|
| alpha 3 | +40.4 / +43.4 points | +29.0 / +30.2 points | -76% / -79% | -69% / -72% | -79% / -86% | +65% / +82% |
| alpha 6 | +8.3 / +13.1 points | +25.8 / +30.6 points | -30% / -39% | -44% / -49% | -75% / -81% | +73% / +78% |

**Against sglang's own colocated schedules** on our tree, which is the leave-one-out that isolates the schedule:

| comparison | chat attainment | agent attainment | chat mean TPOT | agent mean TPOT | chat mean TTFT |
|---|---|---|---|---|---|
| alpha 3 vs prefill-prioritized | +7.7 points | +0.7 points | -30% | -18% | -63% |
| alpha 6 vs prefill-prioritized | +4.0 points | +0.4 points | -7% | -15% | -42% |
| alpha 6 vs mixed chunk | +8.0 points | +2.9 points | -3% | -14% | -42% |

Read against the ~5-point attainment noise floor: the agent-attainment differences against sglang's own schedules
are inside it, the chat ones at alpha 3 and against mixed chunk are at or just past it, and the mean-TPOT and
mean-TTFT differences are the solid part. The margins against conditional disaggregation are far outside it.

### The SLO-budgeted chunk is a NEGATIVE result on these metrics, and the rate had hidden it

Measured against L0 -- the same tree, same kernel, same four-mode step, same router, differing ONLY in using
sglang's own chunk policy -- at load factor 6, best chunk arm (demote fix, 2000 ms cap):

| metric | sglang's chunk policy (L0) | SLO-budgeted chunk | change |
|---|---|---|---|
| chat attainment | 61.5% | 41.1% | **-20.4 points** |
| agent attainment | 79.1% | 63.5% | **-15.6 points** |
| chat mean TPOT | 57.1 ms | 46.3 ms | -18.9% |
| agent mean TPOT | 56.1 ms | 43.0 ms | -23.4% |
| chat mean TTFT | 2.28 s | 8.26 s | **3.6x worse** |
| agent mean TTFT | 2.92 s | 11.45 s | **3.9x worse** |
| agent rounds issued | 6284 | 5545 | -11.8% |

The mechanism is coherent and not a bug: a budget solved from a latency model makes chunks smaller, which shortens
every step and so genuinely improves time per output token by 19-23%; but a smaller chunk makes each prefill take
more steps, so time to first token nearly quadruples, the cluster admits 12% less work, and attainment -- which
requires a request to meet BOTH its targets -- collapses. The whole cap sweep agrees: 1000 / 1500 / 2000 / 3000 ms
give chat 32.7 / 33.2 / 35.3 / 28.8%, all far below L0.

**Relaxing the cap further does not rescue it (2026-10-29, corrected sweep).** A larger tail cap means the budget
binds less often, so one might expect the arm to converge on L0's behaviour. It does not, at either value tried so far:

| tail cap | chat attain | agent attain | chat mean TPOT | chat mean TTFT | agent rounds issued | accept |
|---:|---:|---:|---:|---:|---:|---:|
| L0 (sglang's own chunk policy) | **61.5%** | **79.1%** | 57.1 ms | 2.28 s | 6284 | 2.0 |
| 1000 / 1500 / 2000 / 3000 ms | 32.7 / 33.2 / 35.3 / 28.8% | -- | ~41 ms | -- | -- | 2.0 |
| 4000 ms | 42.8% | 64.8% | 49.7 ms | 7.93 s | 5618 | 2.027 |
| 8000 ms | 37.7% | 62.9% | 50.7 ms | 9.18 s | 5452 | 2.017 |

Every arm has `failed=0` and acceptance about 2.0, so all are valid. The sweep is **not even monotonic** -- 8000 ms
is worse than 4000 -- and all six caps from 1000 to 8000 ms sit 19 to 33 points of chat attainment below the
control while improving mean TPOT by 11-28%. 16000 ms is running. A cap that rarely binds should converge on the
control's behaviour; that it does not, across six values spanning an order of magnitude, says the loss comes from
sizing the chunk from a latency budget at all on this workload rather than from the cap's value. The trade is therefore a property of sizing the chunk from a latency budget at all on this workload,
not of the cap's value, which is what makes it a finding rather than a tuning gap.
`mn_capsweep2_16g_20261029.sh`, log `/tmp/h20/mn_capsweep2_r1.log`.
- **This reverses the 2026-10-28 reading**, which was taken on the SLO-attaining token rate and the TPOT tail and
  concluded the rule "cuts the tail 46% past capacity". That is still true of the tail, and it is not the question:
  on attainment the rule loses at both loads. The component is therefore **excluded from the contribution**, not
  scoped to overload. It would only pay where TPOT binds and the TTFT target is loose, which is not the regime any
  tenant here is in.
- Four earlier bugs in the rule were real and are fixed; this result is measured with all four fixed, so it is a
  property of the policy and not of its implementation.

**Two arms the older sections lean on are now VOID, and both failures flattered us:**
- **Stock PD disaggregation has no valid measurement at either load.** The 2026-10-29 re-measurement reported chat
  4.9% with TTFT p50/p90 190/311 s -- but `failed=1890`, `lost_after_failure=7470`, `accept=NONE`, and the decode
  pod logged **2 819 `transfer failed for request`** plus KVTransferError aborts. It measured a broken KV-transfer
  path, not a schedule. A clean-upstream re-run is queued (`mn_pdre_clean_16g_20261028.sh`), since `mn_node.sh`
  requires stock PD on `up_clean/python`.
- **PD multiplexing is unmeasured at both loads, and the THIRD attempt failed for a reason of our own making.**
  The first failure was real: every worker died with `AttributeError: 'NoneType' object has no attribute
  `swa_out_cache_loc'` -- `forward_metadata` is None inside the FA3 backend on the pdmux path, a backend
  incompatibility and not the capture memory two MEM_FRAC guesses had assumed. The retry on sglang's default
  backend then reported `only 0/16 workers healthy after 40 min`, which looks like the same class of fault and is
  not. The cause (2026-10-29) is that our own per-tenant acceptance patch had inserted `import os as _os` at the
  TOP of `metrics_reporter.py`, above its `from __future__ import annotations`; Python requires every `__future__`
  import to come first, so the module raised `SyntaxError` and every scheduler process on four trees
  (`up`, `python`, `python_slo_life`, `h20py`) on all four nodes died before opening a port.
  **Three consequences worth keeping:**
  1. A cluster-wide "no workers healthy" report says nothing about the cluster. Read a worker log first: an
     import-time `SyntaxError` and an RDMA fault present identically at the driver.
  2. The repair tool's own gate used `ast.parse`, which does NOT apply the `__future__` ordering rule, so it
     reported `already-ok` for all four broken files and the first repair pass changed nothing while printing
     success. Validation of patched Python must use `compile()`.
  3. **No recorded result is contaminated.** Every arm that ran before the patch has healthy acceptance (~2.0),
     full request counts and `failed=0`: the demotefix and capsweep arms at both loads were checked individually.
     Only the two pdmux arms were killed by it.
  All four trees were repaired in place and verified with `compile()`; the patch now inserts after the last
  `__future__` line and refuses to write a file the interpreter would reject.
- **With that cleared, the ACTUAL pdmux defect surfaced, and it is structural: pdmux cannot run with speculative
  decoding on this sglang base.** The repaired workers reached CUDA-graph capture and then died with
  `AttributeError: 'NoneType' object has no attribute 'decode_wrappers'` on flashinfer -- the same fault as FA3's
  `swa_out_cache_loc`, which is why switching backends never helped. Both are `self.forward_metadata` being None in
  `forward_decode`. The mechanism, read off the deployed tree:
  - pdmux holds one attention backend per SM group (`decode_attn_backend_group`), and `eager_runner._execute_decode`
    resolves one and publishes it through `ForwardContext` for the model layers to read via `get_attn_backend()`.
  - Whether to PLAN is decided by `ForwardBatch.needs_forward_metadata_init()`, which is a property of the BATCH: a
    pre-planner calls `mark_forward_metadata_ready()` and the forward then skips planning. **Speculative decoding
    is such a pre-planner.**
  - Readiness is therefore tracked per batch while metadata lives per backend OBJECT, and pdmux has N+1 objects.
    The speculative worker plans the main or draft backend and marks the batch ready; the decode forward then runs
    the model under a per-stream backend nobody planned, and the first attention layer dereferences None.
  **Fixed, not worked around** (`apply_pdmux_spec_metadata.py`, deployed and compiling on all four nodes): under
  pdmux, plan whenever the resolved backend is not the object the mark could refer to. This is sound because a
  pre-planner never plans a per-stream decode backend, so planning one cannot disturb pre-planned multi-step draft
  metadata -- that lives on a different object. It is deliberately stronger than testing `forward_metadata is
  None`: from the second decode step on, a per-stream backend still holds the PREVIOUS step's plan, so the None
  test would pass on stale metadata and corrupt silently instead of crashing. Outside pdmux the resolved backend is
  `model_runner.attn_backend`, so the condition is unchanged and no other arm is touched. Cost is one plan call per
  decode step per stream, which non-speculative pdmux already pays.
  **Why fix the baseline rather than run it without MTP:** pdmux is the colocation baseline the single-node bar is
  judged against, and running it without speculation while our arm has it would hold its configuration below ours.
  **The smoke arm then failed a FOURTH time, on a fault the fix itself introduced.** The draft model's CUDA graph
  capture died with `Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is
  pinned`, raised from `eagle_draft_cuda_graph_runner.py`: planning copies plan data host-to-device, and capture
  rejects an unpinned host source. Forcing the plan unconditionally therefore broke capture, where the readiness
  mark had previously made the forward skip planning. Capture has its own planning path -- the decode graph runner
  plans each per-stream backend as it captures -- so the forced plan belongs only on the eager forward.
  `apply_pdmux_capture_guard_20261029.py` adds `not torch.cuda.is_current_stream_capturing()` to the condition;
  deployed and compiling on all four nodes.
  **This is the third distinct cause behind one symptom.** "0/16 workers healthy after 40 min" has now meant an
  import-time SyntaxError in our own patch, a backend/metadata mismatch, and an illegal copy inside graph capture.
  The symptom carries no information; only a worker log does, and every one of these was found by reading one.
  **pdmux stays unmeasured until an arm gets past launch, and no pdmux row may be filled from a failed arm.**

**Standing:** point 1 is the paper's claim and now rests on the realistic replay rather than on fan-out, where
shared prefixes let the baseline nearly match us. Point 4 has become an argument about timescales, supported by
measurement. Point 2 is unchanged. **Point 3's premise is untested, not refuted** -- the decode KV length was never recorded on
decode steps, so no model could have used it. It stays out of the contribution list until the tracer is fixed, one
arm re-traced, and the bias re-measured on a chronological split.

## Evidence state of the four points — 2026-10-21 (superseded by the table above)

Rewritten because the table below says "nothing measured" for points 2 and 4 and "no multi-node
run" for point 1, all of which are now out of date, and because every baseline in it ran
`--attention-backend flashinfer` while upstream sglang's own default on Hopper is **FA3**
(`get_default_attn_backend`). Every number here has both systems on FA3, Qwen2.5-7B unless noted,
H20 (4-GPU pods), agent load = program concurrency, chat by request rate.

| point | what is now measured | what is still argument only |
|---|---|---|
| 1. highest-hit-first with an EDF/SLO check | **the multi-node comparison the old table said was missing.** 32 GPUs, 1x load: ours 596 vs disagg 5+3 383 / 4+4 325 agent tok/s, agent TTFT 0.74 vs 4.6 / 5.8 s, chat TTFT 0.05 vs 3.0 / 3.6 s. 2x load: 1172 vs 444 / 393, TTFT 0.72 vs 12.3 / 14.7 s. Qwen3-8B + real MTP, 16 GPUs: 448 vs 105 / 128, TTFT 1.7 vs 31 / 24 s. Disagg swept over P:D ratios in every case. EDF ordering measured on the single-node MTP server | that the *SLO check* specifically (rather than hit-first routing plus colocation) is what buys this: no arm isolates it at 32 GPUs |
| 2. best-effort cache migration / replication | **the replication effect, as cached fraction.** At 32 GPUs ours holds prefix-cached fraction 0.98 vs disagg 0.76-0.85, and disagg's falls with load (0.85 -> 0.76 from 1x to 2x) because its prefill pods re-prefill agent history. Attribution: ours *without* the cluster pool still does 2092 vs 2153 tok/s, and disagg 5+3 *with* the pool 353 vs 303 -- so the pool is not the gain; colocated placement is | that explicit best-effort *migration* helps: moving a request to a non-holder and pulling its prefix through the pool was **negative** on H20 (section at point 2). The ThunderAgent thrashing comparison is still unrun |
| 3. EDF predictor with acceptance length | **the premise, measured and SUPPORTED (2026-10-22)**: our 4-mode step on the fused backend predicts GPU time at 0.4% decode / 3.2% prefill MAPE vs stock + MTP's 1.5% / 5.7% on the FA3 backend (3.8x and 1.8x better), and the same schedule on upstream's FA3 backend is 2.7-4.5x worse -- the fused split is what makes the step predictable. EDF *ordering* also wins on the MTP server (`edf-order-not-cap`) | that the accuracy converts into better end-to-end SLO control: the matched fused/unfused EDF A/B still shows no ITL or throughput advantage |
| 4. two-way RDMA, complete vs bipartite graph | **scaling with load, not congestion.** Ours doubles agent throughput 1x -> 2x (596 -> 1172) at flat TTFT; disagg grows 16-21% and its TTFT triples. Pull-source choice measured directly (2026-10-19, `random` / `least_inflight` / stock order, one NIC, hot prefix): no congestion regime exists at <= 16 H20s, so point 4's randomized-pull rule is **refuted at this scale** | the congestion claim itself at larger N. `O(N^2)` vs `O(N_p*N_d)` remains topology arithmetic |

**So two of the four points now rest on measurement** (1 and 2's replication effect, both
multi-node and both with disagg at its best ratio), one is refuted at the scale we can test (4's
pull rule), and one has an inverted premise that no measurement has rescued (3). The single-node
bar is **not met** against stock on FA3 (best ours: -3.4% throughput, +9 points agent attainment,
-1.5 points chat attainment; with MTP -16% throughput), so the argument of this doc is the
multi-node one.

## Evidence state of the four points — 2026-10-10 (superseded; kept for the labelling it introduced)

Added because the sections below carry 17 "measured" markers and **no** markers for the claims
that are argument only, so a reader cannot tell which is which. Nothing here changes the
arguments; it labels them.

| point | mechanism exists in sglang | measured | argument only |
|---|---|---|---|
| 1. highest-hit-first with an EDF/SLO check | yes — radix/prefix cache + router hit lengths; EDF ordering is **measured** on the MTP server (`edf-order-not-cap`) | the EDF *ordering* win, and the linear-in-KV step cost the SLO check's predictor needs | that routing hit-first **beats** disagg's prefill-first on TTFT: no multi-node run |
| 2. best-effort cache migration | yes — hier cache + dist cache pool | nothing | the whole replication advantage, incl. the ThunderAgent thrashing comparison |
| 3. EDF predictor with acceptance length | yes — EWMA of accept length is already tracked | step cost linear in total KV, on one GPU, 4-mode MTP (section 3) | a matched fused/unfused EDFG A/B now runs cleanly, but shows no ITL/throughput advantage and does not establish better predictor accuracy |
| 4. two-way RDMA, complete vs bipartite graph | partially — transfer paths exist, congestion-aware pull selection does not | nothing | the entire scaling claim; `O(N^2)` vs `O(N_p*N_d)` is topology arithmetic, not a measurement |

So **one** of the four points has supporting measurement (3's cost model, plus 1's EDF
ordering), and it is single-GPU. Points 2 and 4 are unmeasured end to end. The multi-node runs
that would test them are gated on the kernel items per the standing instruction, and separately
on the FA3 defect, since a serving A/B today would measure a corrupted FA3 path.

**This labelling does not weaken the doc's purpose** — the four points are a design argument the
goal asked to be written down, and they are written down. It records which parts a reviewer can
challenge on data and which on reasoning.

## 1. Route to the highest cache-hit node first, with an EDF/SLO check

**The disagg baseline.** PD disaggregation picks a *prefill* node first, because prefill and
decode live on different pools. Prefix cache locality is then a second-order concern: the
request goes where prefill capacity is, and whatever prefix that node happens to hold is luck.
With agentic traffic this is the wrong order of preference, because the prefix is enormous
(64-100K) relative to the new tokens (~3.8K), so a hit is worth far more than a marginally
emptier queue.

**The colocated policy.** Rank candidate nodes by prefix-hit length, then **SLO check** by SLO
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
popular prefixes. The SLO check is the only thing preventing hotspotting, so the experiment must
report the **hit-rate vs p99-TTFT frontier**, not just mean TTFT. If the SLO check fires often, the
policy degenerates toward load-balancing and the advantage shrinks -- that is the honest
failure mode and it should be looked for.

## 2. Best-effort cache migration, and why colocation replicates better

When the SLO check rejects the best-hit node, we route to a worse-hit node **and** pull the missing
prefix to it, using sglang's hierarchical cache plus the distributed cache pool. The pull is
best-effort: if it does not complete before the request is scheduled, the node recomputes the
tail, so correctness never depends on the transfer and a slow link degrades latency rather
than breaking the request.

**The replication argument.** Every migration leaves a copy. Under cache-hit-first routing with
an SLO check, popular prefixes are therefore *pulled toward wherever load forces requests to
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
attention implementation whose cost is now **linear in total KV** (*measured*, above).

> **[!] CORRECTION 2026-10-10 — this premise holds for the UNFUSED colocated step, not the fused
> one.** With the two-role fused kernel the optimal `sm_split` moves with batch composition
> (128 / 96 / 64 at 2 / 8 / 32 decode rows) and a mis-set split costs **23x** (8.68 ms vs 0.367 ms
> at dec=32). A predictor would have to model the split too, over a target with a 23x
> discontinuity. The claim should rest on FA3-only colocation — one kernel, linear in KV, no split
> parameter — which is both simpler to model AND currently faster than the fusion (1.46–2.01x).
> See the 2026-10-10 section at the top of this doc. Features
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
        router's admission/SLO check across nodes (point 1). The in-node scheduler should order by
        deadline and stay work-conserving.

### Router support for point 3 (2026-10-14)

`tasks/pd_colocation/router.py` now predicts per-token TPOT under speculative decoding: each
worker keeps an EWMA of acceptance length from finished requests' `meta_info`
(`completion_tokens / spec_verify_ct`, `--accept-ewma-alpha`), and the SLO check divides the
step-time prediction by it; `--step-model` loads the linear MTP step model from
`tasks/perf_model/fit_mtp_step.py` (incl. its `c_acc` term) -- the S4 refit on the split kernel
is what will feed it. Defaults leave the old per-step behaviour unchanged. Checks:
`router_accept_test.py` (EWMA and TPOT arithmetic, PASS) and `router_test.py` (fake workers;
same routing behaviour as the committed router on all four policies).

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

## 5. Agentic append rounds make disagg re-ship the session's KV every round — 2026-10-13

(User point, 2026-10-12.) An agent session is many short *append* rounds against a long, growing
context: tool result appended, a few hundred output tokens, repeat. Under disaggregation each
round is a new request: the prefill node computes the appended tokens and the decode node must
receive the request's KV before its first decode step. Stock sglang PD frees a request's KV on
the decode node when it finishes, so every round ships the **whole context**; even a decode
node that kept the session prefix would still ship every round's **append**. Under colocation
the session's KV stays on the node that decodes it (cache-hit-first routing, point 1), so a
round moves **nothing** and only computes its append.

**Measured from the full TraceLab release** (353,598 follow-up rounds in 4,109 sessions; rounds
per session p50 18, p90 163, p99 1209; context p50/p90/p99 124K/256K/823K tokens; append
p50/p90/p99 860/6.5K/105K tokens). Transfer time per round at line rate, no contention:

| model (KV bytes/token) | link | disagg, full KV per round p50 / p90 / p99 | disagg, append only p50 / p90 / p99 | colocation |
|---|---|---|---|---|
| Qwen2.5-7B (57.3 KB) | 200 Gb/s | 285 / 587 / 1887 ms | 2.0 / 14.8 / 240 ms | 0 |
| Qwen2.5-7B (57.3 KB) | 400 Gb/s | 143 / 294 / 944 ms | 1.0 / 7.4 / 120 ms | 0 |
| 70B GQA-8 (320 KB) | 200 Gb/s | 1.63 / 3.36 / 10.8 s | 11 / 85 / 1373 ms | 0 |
| 70B GQA-8 (320 KB) | 400 Gb/s | 0.81 / 1.68 / 5.39 s | 6 / 42 / 687 ms | 0 |

Over the whole release that is 3.12 PB moved (full KV) against 107 TB (append only) for a 7B
model -- 29x -- and colocation moves neither.

**Where it shows up:** in **TTFT** of every round (the transfer precedes the first decode
step), and in **ITL of the requests already decoding** on the receiving node, because each
incoming transfer competes with them for the NIC and for HBM write bandwidth. Disagg's usual
mitigation -- layer-wise streaming overlapped with prefill -- hides transfer behind *prefill
compute*, but an append round has almost none (median 860 tokens), so there is nothing to hide
it behind. This complements points 1 (keep the session where its cache is) and 2 (when a session
must move, it moves once, not every round).

**Prior art that already blunts this, stated plainly.** NVIDIA Dynamo's *conditional
disaggregation* prefills locally on the decode worker when the uncached part of the prompt is
small (defaults: effective ISL under 2048 tokens and under 0.7 of the prompt) or remote prefill
is loaded; with the median append at 860 tokens and the session prefix resident on its decode
worker, most agent rounds then move no KV at all. A delta-KV RFC would cut the remaining remote
rounds to append-only transfer (the 1-15 ms column above, not 143-587 ms), but it is an open
proposal with no replies (2026-10-24), not shipped behaviour. semi-PD avoids transfer
entirely by splitting SMs inside one GPU over a shared KV store. So point 5 is decisive against
stock sglang PD and against Dynamo's remote-prefill path; against Dynamo with conditional
disaggregation the advantage that remains is (a) making the local prefill + speculative decode
co-execution efficient -- that is our 4-mode step and fused kernel, measured at +20-36%
throughput and -31..-79% TTFT over stock MTP under load -- (b) symmetric nodes instead of fixed
prefill/decode pools, so routing and cache replication (points 1-2) can use every node, and
(c) EDF/SLO-aware admission (point 3). Dynamo with conditional disaggregation must be run as a
baseline rather than argued around.

**What must be measured (H20, <= 16 GPUs, trial `vibecuda`):** full-length TraceLab replay
(round-weighted session sampling, both providers) on sglang PD (e.g. 2P2D) vs colocation with
cache-hit-first routing, at matched rates; report per-round transferred bytes, round TTFT, ITL
of co-resident decodes on decode nodes, and NIC utilisation. Not run yet. The single-node
full-length run (S2, H200) measures the colocation side of this first.

---

## 6. NVIDIA Dynamo's scheduling policy vs ours, in detail — 2026-10-13

Sources: Dynamo docs -- [KV router routing concepts][dyn-router], [conditional
disaggregation][dyn-cd] (v1.5.1, experimental), [planner][dyn-planner] (and the older [SLA
planner][dyn-sla]), [KVBM design][dyn-kvbm] / [overall architecture][dyn-arch], and the [delta-KV
transfer RFC][dyn-rfc]; full list under [References](#references).

**How Dynamo schedules, layer by layer:**

| layer | how Dynamo decides |
|---|---|
| **Router** (per request) | Picks the worker with the lowest block-count cost: pending prefill blocks minus a prefix-hit credit, plus active decode blocks, plus an optional per-request charge. Cache hits are just a load discount. There is no SLO or deadline term anywhere, and no time model. |
| **Queueing** | Weighted round-robin across request classes, then the engine's own first-come order. No deadline ordering. |
| **Conditional disaggregation** (experimental) | Separate prefill and decode pools. A request prefills locally on its decode worker only below static thresholds: uncached prompt under 2048 tokens and under 70% of the prompt, or when the prefill workers are busy. The local prefill then runs in the engine's ordinary mixed batch. It doesn't support SGLang (vLLM and TensorRT-LLM only). |
| **Planner** | Sizes the prefill and decode replica counts separately to meet TTFT/ITL targets: a throughput mode every 180 s from load prediction, and a load-based mode every 5 s from per-iteration engine metrics to react to bursts (the two can run together). The older SLA planner works only with disaggregated pools. |
| **KV placement** | KV is offloaded across memory tiers and fetched point to point when needed. Copies of hot prefixes exist only where routing and transfers happened to leave them. |

**Where ours differs:**

- **Routing:** we rank nodes by cache hit, then SLO check any node where a step-time model plus the
  earlier-deadline requests queued there predict a missed SLO (point 1). Dynamo can send a request
  to a cache-hot worker that will miss its deadline.
- **Load model:** we predict step time from a model that includes speculative acceptance length
  (point 3). Dynamo counts KV blocks, which ignore context length and acceptance.
- **Prefill placement:** every node prefills and decodes, and each step co-runs prefill chunks
  with speculative verify work using the fused kernel. That is exactly the local path Dynamo
  leaves to an ordinary batch, and the one we measured at TTFT -31% to -79% and throughput +11%
  to +36% versus stock MTP under load.
- **Reacting to shifts:** the prefill/decode balance is re-chosen every step, against Dynamo's
  planner, which changes how many prefill and decode workers exist (5 s decisions in load-based
  mode, 180 s in throughput mode) rather than the mix of work inside a step.
- **Cache:** we deliberately replicate hot prefixes through migration (point 2) and spread pulls
  across near-equal holders (point 4). Dynamo's copies are incidental.

**Worth adopting from Dynamo:** hit credits weighted by memory tier, a decaying estimate of
prefill load, and output-length hints for agent rounds.

**Baseline:** a fair comparison must run on vLLM, since Dynamo's conditional disaggregation
doesn't support SGLang: conditional disaggregation tuned to the full-length TraceLab distribution,
plus Dynamo's router in aggregated mode as a routing-only control. Both are H20 work. Per the user
(2026-10-13) this comparison is **optional** because it lacks an SGLang backend; sglang PD
disaggregation is the required disaggregation baseline.

---

## Honest summary of standing — 2026-10-21

The split-row corruption that gated everything is fixed and there is now a lot of serving data, so
the 2026-10-10 version of this section (no serving data, corruption blocking) is gone.

- **Multi-node is where the case holds, the confound is now removed, and the result is STRONGER and attributable to
  the schedule alone.** Every earlier multi-node run had ours on sglang's **flashinfer** backend (carrying our FA3
  patches) while disagg ran the **FA3** backend, which the backend-class control measured at twice the throughput on
  this workload -- so ours was handicapped. The rerun with every arm on FA3 and **no split planning at all**
  (`bash tasks/pd_colocation/mn_fa3_16g_20261022.sh`, `KERNEL=lpt ATTN_BACKEND=fa3`; 16 H20s, 32 tasks / 64 agent
  programs, chat 16 req/s, 900 s), first two arms:
  | 16 GPUs, no spec, all on FA3 | agent tok/s | agent TTFT | chat TTFT | chat att. 0.5s/20ms | agent SLO goodput | chat TPOT p90 | cached |
  |---|---|---|---|---|---|---|---|
  | **ours** (schedule + router only, no split) | **962.7** | **0.89 s** | **0.07 s** | **96.5%** | **939 tok/s** | **11.5 ms** | 0.974 |
  | disagg 3 + 1 (its best ratio) | 344.9 | 17.24 s | 14.93 s | 23.3% | 129 tok/s | 13.3 ms | 0.775 |
  | disagg 2 + 2 | 242.3 | 27.57 s | 25.66 s | 15.6% | 56 tok/s | 7.4 ms | 0.761 |
  **THE BAR IS MET HERE, AND WITH NO KERNEL WORK IN THE PICTURE.** Against disagg's best ratio: chat SLO attainment
  **+314%** (96.5% vs 23.3%), agent SLO goodput **7.3x** (939 vs 129 tok/s), raw agent throughput **+179%**, agent
  TTFT **-95%**, chat TTFT **-99.5%** -- all far beyond the >=30% attainment / >=10% throughput requirement. Ours'
  chat TPOT p90 (11.5 ms) also beats disagg 3+1's (13.3 ms), so the ITL axis favours us at this scale too; only
  disagg 2+2, which starves agents to protect decode, has a tighter chat TPOT (7.4 ms) and it pays with 15.6%
  attainment. Ours ran `KERNEL=lpt ATTN_BACKEND=fa3` -- upstream's unmodified attention backend, no split, no fused
  dispatch -- so **the entire multi-node advantage is the colocated schedule plus the cache-first SLO router**. The
  fused kernel's measured contributions are elsewhere: step predictability (point 3, 0.4% / 3.2% MAPE) and ~2.5% on
  our own single-GPU schedule.
  - [x] **MTP pays off at cluster scale, the opposite of single node (2026-10-22, same rerun):** ours + MTP on the
    FA3 backend with no split reaches **1120.7 agent tok/s** against the no-spec arm's 962.7 (+16%), at agent TTFT
    0.80 s, chat TTFT 185 ms, cached 0.972, 0 errors. On one GPU the 4-mode MTP arm was the *worst* configuration at
    saturation (agent attainment 19.4% at C=16); at 16 GPUs it is the best. The reason is the same routing dimension
    that makes the whole multi-node case work: the router can place a verify-heavy step on a node that can absorb it,
    where a single GPU must interleave it with everything else. **With disagg's MTP arms in, the MTP comparison is the
    widest of all (all on FA3, disagg at both ratios):**
    | 16 GPUs, MTP, all FA3 | agent tok/s | agent TTFT | chat att. 0.5s/20ms | agent SLO goodput | chat TPOT p90 | cached |
    |---|---|---|---|---|---|---|
    | **ours + MTP** | **1120.7** | **0.80 s** | **92.8%** | **1099 tok/s** | **9.5 ms** | 0.972 |
    | disagg 2 + 2 + MTP | 195.9 | 34.25 s | 4.9% | 25 tok/s | 6.5 ms | 0.74 |
    | disagg 3 + 1 + MTP | 204.6 | 31.81 s | 3.8% | 22 tok/s | 13.8 ms | 0.705 |
    Against disagg's better ratio: **chat SLO attainment 19x (92.8% vs 4.9%), agent SLO goodput 44x (1099 vs 25
    tok/s), raw agent throughput 5.2x, agent TTFT -98%**. Ours' chat TPOT p90 (9.5 ms) also beats disagg 3+1's
    (13.8 ms); only disagg 2+2 is tighter (6.5 ms) and it attains 4.9%. Speculation widens the gap because disagg must
    ship the draft hidden states to its decode pods every round on top of re-prefilling the session, while colocation
    verifies where the KV already is.
  - [ ] **What the kernel adds at cluster scale is a separate question, and it is now queued
    (`mn_kernel_on_16g_20261022.sh`).** Turning the kernel off was deliberate -- it attributes the win -- but it
    leaves the fused stack's own multi-node contribution unmeasured. The queued arms run the SAME workload, router,
    EDF and pool with the full fused stack ON (split rows on FA3, per-step split policy, per-layer fastpath, verify
    on FA3), no-spec and MTP, so the difference against the `KERNEL=lpt` arms above is the kernel's cluster-scale
    value with everything else held fixed. On one GPU that difference was +2.5%; whether it is larger at 16 GPUs,
    where mixed steps carry more and deeper decode rows, is exactly what the fused split was built for.
  Against PD disaggregation at its best P:D ratio, both systems on FA3, ours wins
  by margins far above the bar: +56% agent throughput at 32 GPUs / 1x
  load, +164% at 2x, 3.5x with Qwen3-8B and real MTP at 16 GPUs, with agent TTFT 6-18x lower.
  Disagg's weakness is structural here -- it re-prefills agent session history every round, so its
  cached fraction falls with load while ours stays at 0.98.
- **Single node does not meet the bar.** Against stock sglang on its own Hopper default (FA3),
  best ours is -3.4% throughput, +9 points agent SLO attainment, -78% chat TTFT, but -1.5 points
  chat attainment; with MTP -16% throughput. The deficit is ours' chat TPOT tail, and smaller
  prefill chunks trade it straight back for agent TTFT. Every earlier single-node headline that
  beat stock was measured against stock forced onto the flashinfer backend and is superseded.
- **Where ours still loses on latency tails:** agent ITL p99 is 1.5-2.5x disagg's at 32 GPUs (its
  decode pods never see a prefill) and 6-11x at 16 GPUs with MTP.
- **The kernel contribution is narrower than the project name suggests.** With both halves of the
  mixed step on FA3, no FA2 role remains in serving; what is left is the per-step decision to split
  one FA3 call into two co-running launches, worth 4-25% on kernel microbenchmarks (up to 77% when
  a short chunk sits beside long-context rows) and nothing measurable in serving. Multicast
  (-0.2..+1.7%) and in-kernel cross-role backfill are closed as negatives or by bound.

Open, in priority order: the ~9% backend gap behind stock's FA3 backend (decode-step test in
flight; the structural alternative is porting the split onto sglang's FA3 backend class), point 3's
EDF predictor-accuracy comparison against the unfused schedule, point 2's ThunderAgent-style
thrashing comparison, and an intermittent TP4 forward-step hang on GLM-4.5-Air.

---

## References

Systems and papers this doc argues against or borrows from. *(sent)* marks the ones the user sent;
the kernel-side reference list is at the end of `debug_kernel_server_perf_4_mode.md`.

- **NVIDIA Dynamo** -- KV router, conditional disaggregation, planner, KVBM; the main baseline for
  points 1, 5 and 6. Docs: [KV router: routing concepts][dyn-router]; [conditional
  disaggregation][dyn-cd]; [planner][dyn-planner] and the older [SLA-based planner][dyn-sla];
  [KVBM design][dyn-kvbm], [overall architecture][dyn-arch]; [delta-KV transfer RFC][dyn-rfc].
  - *Summary:* the router picks the worker with the lowest block-count cost (pending prefill blocks
    minus a prefix-hit credit, plus decode blocks), weighted DRR across policy classes, no SLO or
    deadline term. Conditional disaggregation (experimental) prefills on the decode worker when
    the effective ISL is under 2048 tokens and under 0.7 of the prompt (defaults), or when prefill
    is busy; vLLM / TRT-LLM only. The planner sizes P and D replicas to TTFT/ITL targets (180 s
    throughput mode, 5 s load-based mode). KVBM: G1-G4 KV tiers, NIXL point-to-point transfer,
    replication only where routing happened to leave copies. The delta-KV RFC (append-only
    transfer for remote rounds) is an open proposal with no replies, not shipped.
  - *Core method/contributions:* KV-aware routing (prefill-block cost with cache-hit credits
    plus projected decode blocks), conditional disaggregation on effective ISL, a planner sizing P
    and D replicas from load prediction and TTFT/ITL targets, and KVBM tiered KV storage over NIXL.
  - *Relation:* conditional disaggregation keeps most agent appends on the decode worker, so it
    removes most of the per-round KV re-ship that point 5 charges stock sglang PD; our emulation
    (`router.py cond_disagg`, 2 + 6 pods) matched ours on chat attainment at 64 programs (94.6 vs
    94.2%), with ours +14% agent SLO goodput. What it lacks against ours: an SLO / deadline term in
    routing (point 1), deliberate replication (point 2), an acceptance-aware step model (point 3),
    symmetric nodes, and an efficient local path -- its local prefill runs in the engine's
    ordinary mixed batch, where ours co-runs the chunk with MTP verify rows in the 4-mode step.
  - *Learn:* its fixed thresholds and worker-count planner are the comparison for scheduling
    flexibility under load spikes: run a step change in load and report the time to recover TTFT /
    ITL, where our per-request routing and per-step prefill/decode mix need no worker changes; and
    find the load knee (96 / 128 programs) where pinning prefill to 2 of 8 pods starts to cost
    cluster-wide request volume that symmetric nodes keep.
- **semi-PD: Towards Efficient LLM Serving via Phase-Wise Disaggregated Computation and Unified
  Storage** -- K. Hong et al., [arXiv:2504.19867][semipd] ([code][semipd-code])
  - *Summary:* SM-level prefill/decode split inside one GPU (CUDA MPS) over a shared KV store, so
    no KV transfer and no weight replication. A ratio change starts new workers on the weights and
    KV held by a resident process and retires the old ones after their current iteration; an
    SLO-aware algorithm refits TTFT / TPOT models every window of iterations and steps the ratio.
    1.27-2.58x lower mean end-to-end latency (DeepSeek), 1.55-1.72x more requests within latency
    constraints (Llama).
  - *Core method/contributions:* a computation resource controller (SM-level P/D partition), a
    unified memory manager over one KV store, a low-overhead resource adjustment mechanism between
    phases, and an SLO-aware dynamic partitioning algorithm.
  - *Relation:* like ours it avoids transfer by keeping both phases on the GPU that holds the KV;
    it differs in granularity -- its partition moves between process generations every window of
    iterations, ours re-chooses the prefill/decode mix every step and co-runs prefill chunks with
    MTP verify rows. Its released code is a separate sglang v0.4.4.post1 fork with radix cache off
    and no speculative decoding, and it has not been run here (sgl-kernel build blocked).
  - *Learn:* the flexibility comparison is reaction granularity, not static vs dynamic: measure how
    many iterations its window needs to follow a shift in the prefill/decode mix against our
    per-step split, and report per-node request volume next to it.
- **Towards High-Goodput LLM Serving with Prefill-decode Multiplexing (MuxWise)** -- Y. Chen,
  W. Cui et al., [arXiv:2504.14489][muxwise] *(sent)*
  - *Summary:* intra-GPU P/D multiplexing with an SLO-aware dispatcher; peak throughput under SLO
    2.20x on average (up to 3.06x) over state-of-the-art baselines. Run here as sglang
    `--enable-pdmux`, the direct colocation baseline.
  - *Core method/contributions:* intra-GPU P/D multiplexing with independent phase execution, a
    bubble-less multiplex engine, a contention-tolerant latency estimator and an SLO-aware
    dispatcher.
  - *Relation:* the colocation design closest to ours at node level. On 8 H20s (pdmux + worker EDF,
    non-speculative) it clears the bar against stock colocation + MTP except agent TTFT, but loses
    TTFT 1.9-3.2x to stock colocation without speculation, because its SM groups cap prefill at 56
    of H20's 78 SMs. Ours keeps the whole GPU for every step and splits work inside one forward.
  - *Learn:* its H20 result shows a fixed SM partition costs prefill capacity, which lowers the
    request volume a node sustains; report our per-step split against it on prefill-bound hardware,
    with TTFT and ITL per tenant.
- **Crossflow: Prefill–Decode Elasticity for Agentic LLM Serving** -- Y. Xu et al.,
  [arXiv:2609.27085v1](https://arxiv.org/abs/2609.27085v1) *(sent)*
  - *Summary:* fixed P/D roles with decode-local prefill admitted through revocable leases and SLO
    feedback; one-hour trace replays compressed by a load factor on six TP4 GB300 servers; 17.4% /
    16.2% geometric-mean input / output throughput over static P/D, with GLM-5.2 TraceLab mean ITL
    20.9-34.3% worse. No new kernels.
    [Detailed analysis](debug_kernel_server_perf/resume_20261012/crossflow_analysis_20261014.md).
  - *Core method/contributions:* expiring, versioned decode-worker leases (compute, KV, output,
    transfers, spill slots), a router that compares P-to-D service with decode-local prefill per
    request and reserves capacity atomically under a cluster-wide borrowing cap, and per-unit local
    revalidation before each bounded prefill.
  - *Relation:* the direct elastic-P/D baseline for points 1-4, not merely static disaggregation:
    it keeps decode homes and lends bounded local-prefill capacity; ours makes every node symmetric.
    Our placement, replication and network benefits over it remain hypotheses until a matched
    comparison is run (it has not been; the paper links no implementation).
  - *Learn:* its ITL regression is the cost of decode-local prefill in an ordinary mixed batch,
    which our 4-mode step and fused kernel target; report mean and p99 ITL per tenant next to TTFT
    on the same replay, and use its timestamp compression as the request-rate load axis for chat.
- **ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System** -- H. Kang et al.,
  [arXiv:2602.13692][thunder] *(sent)*
  - *Summary:* KV-cache thrashing in agentic serving (a session's cache evicted during a tool call
    and re-prefilled); load axis is the number of concurrent programs; 1.5-3.6x serving
    throughput, 1.8-3.9x RL rollout, up to 4.2x disk-memory savings.
  - *Core method/contributions:* agentic workflows modelled as LLM Programs; a program-aware
    scheduler that maximizes KV-cache hit rate and mitigates memory imbalance; a tool resource
    manager that prepares tool environments asynchronously.
  - *Relation:* motivates point 2: colocation plus the cluster pool keeps session prefixes resident
    where disagg's prefill pods recompute them. Measured as it poses it (program concurrency 32,
    working set 1M-8M KV tokens): ours loses 10.6% throughput, disagg 55%.
  - *Learn:* load is program concurrency C, never sessions/s; report C with measured in-flight
    requests, and the cached fraction as the direct re-prefill measure behind cluster-wide request
    volume.
- **TraceLab: Characterizing Coding Agent Workloads for LLM Serving** -- K. Zhu et al.,
  [arXiv:2606.30560][tracelab] ([dataset][tracelab-gh]) *(sent)*
  - *Summary:* characterization of Claude Code / Codex sessions: long contexts (p50 ~124K) with
    short outputs, long autonomous loops, heavy-tailed tool calls, high but imperfect prefix hits.
  - *Core method/contributions:* an open trace of ~4,300 Claude Code / Codex sessions (~350K LLM
    steps, ~430K tool calls), its collection pipeline and analysis code, and the characterization
    (long loops, long contexts with short outputs, high but imperfect prefix-cache hits).
  - *Relation:* the agentic workload of this doc and the source of point 5's per-round transfer
    numbers (median append 860 tokens on a resident prefix).
  - *Learn:* append-heavy rounds over long prefixes are where colocation avoids transfer and where
    prefill meets speculative decode in one step; report results by uncached-prompt-length bucket.
- **SLOs-Serve: Optimized Serving of Multi-SLO LLMs** -- S. Chen et al., [arXiv:2504.08784][slos]
  *(sent)*
  - *Summary:* application- and stage-specific SLOs over mixtures (summarization, coding, chat,
    tool calling, reasoning) crossed with load; 2.2x average per-GPU serving capacity.
  - *Core method/contributions:* a multi-SLO dynamic-programming planner over chunked-prefill token
    allocations and (optional) speculative decoding, with multi-replica dynamic request routing and
    resilience to bursty arrivals.
  - *Relation:* plans token budgets per step on unchanged kernels and routes across replicas; ours
    changes what one step computes (4-mode step, fused kernel) and admits by EDF / SLO checks.
  - *Learn:* its evaluation -- per-class SLOs, goodput rather than raw tok/s, mixture x rate
    sweeps -- is the frame for experiments 5c / 5d, where scheduling flexibility under mixed
    workloads and workload variation is visible.

[dyn-router]: https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/routing-concepts
[dyn-cd]: https://docs.nvidia.com/dynamo/advanced-customizations/conditional-disaggregation
[dyn-planner]: https://docs.nvidia.com/dynamo/v1.5.0/knowledge-base/modular-components/planner/overview
[dyn-sla]: https://docs.nvidia.com/dynamo/v-0-8-1/components/planner/sla-based-planner
[dyn-kvbm]: https://docs.nvidia.com/dynamo/v-0-9-1/design-docs/kvbm-design
[dyn-arch]: https://docs.nvidia.com/dynamo/v1.1.1/design-docs/overall-architecture
[dyn-rfc]: https://github.com/ai-dynamo/dynamo/discussions/7861
[semipd]: https://arxiv.org/abs/2504.19867
[semipd-code]: https://github.com/infinigence/Semi-PD
[muxwise]: https://arxiv.org/abs/2504.14489
[thunder]: https://arxiv.org/abs/2602.13692
[tracelab]: https://arxiv.org/abs/2606.30560
[tracelab-gh]: https://github.com/uw-syfi/TraceLab
[slos]: https://arxiv.org/abs/2504.08784
