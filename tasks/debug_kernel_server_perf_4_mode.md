# Current goal

Fused FA3+FA2 SM pre-assignment attention for a 4-mode MTP (EAGLE3) serving schedule, optimised
on real serving workloads, then the EDF model and the PD-colocation argument
(`tasks/pd_colocation_vs_disagg.md`). Bar: >=30% TTFT and ITL, >=10% throughput on realistic
chat + TraceLab workloads. Single H200 trial `095f01a3e1585eea` (`vibecuda`); H20 multi-node only
after the kernel items close (<=16 GPUs).

**Where it stands (2026-10-13):** under load the fused 4-mode arm beats stock MTP by TTFT
-31..-79%, throughput +11..+36%, per-token latency -19..-30%; below capacity the arms are equal.
TTFT and throughput clear the bar wherever the server is loaded; per-token latency clears it only
at some points. The graphs-on 4-mode crash is fixed (a schedule-stream publish raced the forward stream); 4-mode arms now run without the mixed-step barrier.

## Tasks, in priority order

A and B are the two kernel items; S is the server/workload evaluation the user specified.
Supporting gates are prerequisites, not substitutes. `[x]` = measured, `[~]` = partly, `[ ]` = not run.

### [~] A. Ultra fine-grained tile dispatcher

- [x] **A-fix, FA2 out of the co-compiled kernel (2026-10-12):** FA2 spilled inside the fused
  kernel (168 regs, 4200 B stack, 21x slower per item than `BatchAttention`); now FA3 grid on
  `sm_split` CTAs + `FA2ExecutorKernel` at native geometry as its PDL dependent
  (`FLASHINFER_FUSED_COLAUNCH=1`). 3.3-6.5x on FA2 rows, every gate exact; still 1.2-2x behind the
  non-persistent unfused two-launch control. ([evidence](debug_kernel_server_perf/resume_20261012/colaunch_20261012.md))
- [x] **A1, FA3 q64 vs q128 per work item (2026-10-12):** closed by measurement -- q64 never beats
  q128 beyond noise and loses up to 2.3x; nothing to select. Not built for that reason.
- [x] **A2, FA2 tile classes (2026-10-12):** 48/32/16 refuted (tile 48 = 1 q-warp x 4 kv-warps,
  3x accumulator, spills; 1.6-2.2x slower at q8); **64/32/16** adopted (96 B stack). Per-item tail
  dispatch kept (bit-identical outputs). FA2 split-KV chunk = `auto` (sized to the FA2 CTA slots;
  matches or beats the best fixed chunk on every fixture). ([evidence](debug_kernel_server_perf/resume_20261012/colaunch_20261012.md))
- [~] **A3, fitted FA2/FA3 dispatch rule (2026-10-12):** winner map on the fixed kernel, 216/216
  exact: FA2 wins every verify-width row; rule "FA3 iff packed_qo >= 112 and FA3 work >= 3.5M"
  agrees on 69/72 cells (worst loss 6%) vs 55/72 (98%) for the serving rule -- **but loses in
  serving** (-3.5% throughput, +29% TTFT). Cause, from the split counters (2026-10-13): the work
  floor was fitted on band rows only (qo <= 32) but also vetoed 512-token chat prefill chunks
  (3584 packed x 512 kv = 1.8M), so 1039/1601 split plans went one-sided onto the stock path vs
  122/1801 under the default rule. Routing one-sided batches through the co-launch instead would
  not help: on one-sided band shapes the fused kernel is 1.0-2.2x slower than `BatchAttention`.
  ([table](debug_kernel_server_perf/resume_20261012/a3_winner_map/a3_v2_table_20261012.txt))
  - [ ] **A3b:** the floor applies only inside the fitted regime (all FA3 candidates qo <= 32,
    `a3b_dispatch_rule.patch`); gated serving A/B vs the default rule, interleaved twice for a
    noise floor, `a3b_serving_ab_20261013.sh` -- queued.

- [ ] **A4, FA2 verify-row bandwidth (2026-10-13):** attention is 75-80% of serving kernel time,
  dominated by memory-bound verify rows. At serving verify shapes (q=4, GQA 28/4, all arms exact,
  `fa2_verify_bw_20261013.sh`), KV bytes / time per layer:
  | fixture | fused co-launch | unfused: FA3 + stock BatchPrefill FA2 | BatchAttention |
  |---|---|---|---|
  | 46 rows x 16K | 0.850 ms, 1.82 TB/s (38%) | **0.509 ms, 3.03 TB/s (63%)** | 0.831 ms |
  | 46 rows x 48K | 2.523 ms, 1.84 TB/s (38%) | **1.511 ms, 3.06 TB/s (64%)** | 2.446 ms |
  | 16 rows x 128K | 1.732 ms, 2.48 TB/s (52%) | **1.083 ms, 3.97 TB/s (83%)** | 1.985 ms |
  | 512-chunk + 46 x 48K | 2.567 ms | **1.532 ms** | 2.294 ms |
  The co-launched FA2 executor inherits the persistent BatchAttention design and reaches 38-52%
  of HBM; stock non-persistent BatchPrefill reaches 63-83%. On verify rows the fused path is
  ~1.65x slower than it need be -- the A-fix colaunch doc's "1.2-2x behind the two-launch
  control" is this. Serving's `MTPMIX_SPLIT` arm is exactly that two-launch split (FA3 via
  `flash_attn_with_kvcache` + `BatchPrefillWithPagedKVCacheWrapper` fa2). Next: gated serving A/B
  split vs fused at mix32 / mix128 x 16 / 24/s (`split_vs_fused_20261013.sh`, queued; the gate now
  requires split calls > 0, `split_gate_counter.patch`); then a PDL split (stock FA2 launched as
  the FA3 grid's dependent on the SMs FA3 leaves) and an FA2 executor whose KV pipeline matches
  BatchPrefill's.
  - [x] **Split A/B v1 invalid (2026-10-13 06:5x, recorded):** `MTPMIX_SPLIT` planned 2201
    two-sided splits but ran **zero** -- forward_extend dispatches the split on the paged path
    only, and ragged was forced off only under FUSED_KERNEL_EXEC, so the two-launch split fell
    through to stock BatchPrefill every layer. Its gate "PASS" was false: the driver raised "zero
    split calls" but the gate subshell returned the compare's status. Fixed: ragged off whenever a
    two-sided split plan exists (`split_dispatch_paged.patch`, in `up_racefix`); gate subshells in
    the split and A3b scripts now fail on the run's own status. A/B aborted, re-queued as v2.
  - [x] **FA2 executor tuning (2026-10-13, all exact):** CTAs/SM 2x vs 3x: no change (2.52 vs
    2.52 ms on 46 x 48K) -- not MLP-bound. KV chunk is the lever: auto 2.52 ms, 1024 2.01 ms,
    4096 1.84 ms (unfused control 1.49-1.52 ms); same on chunk+verify (2.55 / 2.03 / 1.84 ms). The
    auto rule splits long verify rows too finely. Larger chunks: 8192 1.92 ms, 16384 2.16 ms,
    65536 (no split; 184 items for 264 CTAs) 2.53 ms -- **optimum 4096: 1.84 ms, still ~20% behind
    the two-launch control (1.49-1.51 ms)**, a gap in the persistent mainloop itself, not its
    configuration (`fa2_exec_tune_20261013.sh`, `fa2_exec_tune2_20261013.sh`). Next: the auto
    rule needs a floor near 4096 for long verify rows (re-check on the A2 fixtures before
    changing it); the serving decision waits on split v2 (gate PASS, long prompts agree on 52
    tokens, split calls now required > 0; running).
  - [~] **Split v2, first point (2026-10-13 07:14; provisional until mix32 24/s and mix128 land):**
    mix32 chat 16/s + agent 0.1/s, race-fixed tree, no barrier, both arms 0 tracebacks / 0
    fallbacks; split ran 80,001 split calls, fused 66,901 fused-kernel calls.
    | metric | split (FA3 + stock FA2 BatchPrefill) | fused co-launch | stock MTP (S1) |
    |---|---|---|---|
    | chat TTFT mean | **689 ms** | 4932 ms | 24740 ms |
    | chat TPOT mean | **14.5 ms** | 28.1 ms | 39.0 ms |
    | chat ITL p99 | 108 ms | 149 ms | 138 ms |
    | agent TTFT mean | **1941 ms** | 5102 ms | 14500 ms |
    | agent TPOT mean | **11.3 ms** | 16.0 ms | 21.9 ms |
    | agent output tok/s | **308** | 205 | 162 |
    Chat throughput equals the offered load in all arms (~1136 tok/s). The barrier-free fused arm
    (4932 ms TTFT) is within 7% of S1's fused-with-barrier (5276 ms). Command:
    `INHERIT_LOCK=1 bash split_vs_fused_20261013.sh` (root `split_vs_fused_v2_20261013`), analysed
    with `drivers/serving/eval_curves.py --slo '{"chat":[2000,50],"codex":[10000,80]}'`.

### [~] B. Fuse draft/verify epilogue ops

- [x] **B1, fusion headroom (2026-10-13, measured):** 60 serving steps per arm profiled (torch
  profiler, GPU activity), mix32 chat 16/s + agent 0.1/s, race-fixed tree, no barrier
  (`b1_tail_profile_20261013.sh`, `drivers/serving/parse_kernel_tail.py`).
  | arm | span / GPU busy | attention | GEMM | small ops (index+elementwise+sampling+other) |
  |---|---|---|---|---|
  | fused 4-mode | 4147 ms / 87.2% | 2739 ms (74.9%) | 809 ms (22.1%) | 2.9% (29.6K launches) |
  | unfused 4-mode | 4895 ms / 91.1% | 3574 ms (80.1%) | 783 ms (17.6%) | 2.4% (27.6K launches) |
  | stock MTP | 3635 ms / 90.8% | 2500 ms (75.7%) | 718 ms (21.7%) | 2.7% (24.4K launches) |
  **Epilogue fusion is bounded at ~3% of kernel time** (+ the 9-13% GPU-idle share, part of it
  host-side), far below the bar; B1's further fusion is not worth building. Kineto's
  "est. achieved occupancy" reads 0.4-1% for attention/GEMM (persistent / graph-launched
  kernels) and is not usable -- real occupancy needs Nsight Compute. packed_qo histogram (fused,
  5202 plans): 82,833 verify rows with packed_qo <= 32 carry 4.4% of useful FLOPs at 56.8%
  in-tile waste; 11,141 rows >= 512 carry 95.6% at 1.0%; overall waste 6.3%. Verify rows are
  memory-bound, so their FLOP waste is not the cost -- their KV bandwidth is (A4). Unfused vs fused
  over the same 60 steps: attention 3574 vs 2739 ms (-23%), span -15%. Per context range (chat 16/s + agent 0.1/s, 60 steps each,
  `b1_tail_profile_mix128_20261013.sh`; steps are sampled from live traffic, so step mixes differ
  between arms -- per-call times are the cleaner comparison):
  | range | arm | span | attention | small ops | main attention kernels (avg per call) |
  |---|---|---|---|---|---|
  | mix64 | fused | 16.2 s | 13.3 s (84.7%) | ~1.4% | BatchPrefill 7.20 ms (decode-only steps), FA2Executor 5.34 ms (mixed) |
  | mix64 | unfused 4-mode | 17.7 s | 14.2 s (82.7%) | ~1.5% | BatchPrefill 7.22 ms (all steps) |
  | mix64 | stock | 22.5 s | 17.7 s (80.0%) | ~1.6% | BatchPrefill 9.44 ms |
  | mix128 | fused | 3.6 s | 2.2 s (70.1%) | ~3.9% | BatchPrefill 1.29 ms (decode-only), FA2Executor 0.83 ms (mixed) |
  | mix128 | unfused 4-mode | 7.4 s | 6.0 s (86.1%) | ~1.8% | BatchPrefill 3.15 ms (all steps, long prefill rows on FA2) |
  | mix128 | stock | 6.8 s | 5.8 s (88.6%) | ~1.4% | BatchPrefill 3.14 ms |
  Small ops stay at 1.4-3.9% at every range, so B1's bound holds everywhere. At mix128 the unfused
  merged batch runs its 128-256K prefill rows through FA2 BatchPrefill (3.15 ms/call); the fused
  step sends them to FA3 and the verify rows to the FA2 executor (0.83 ms/call) -- the kernel-side
  mechanism behind the per-range attribution above. Tile waste falls with context (mix32 6.3%,
  mix64 2.7%, mix128 0.6%) because long prefill rows dominate useful work.
- [x] **B2, cross-role backfill (2026-10-12):** co-launch hardware backfill worth up to 2.9x when FA2
  work exceeds the slots beside FA3, nothing on small mixes; in-kernel `tma_store_wait` backfill is
  correct and worth 1.4-5.5x over none, but that whole path is 2-6x slower. Co-launch ships.
  ([evidence](debug_kernel_server_perf/resume_20261012/colaunch_20261012.md))
- [~] **B3, multicast:** not implemented (`mcast_mask=0`; needs page-size-128 paged TMA). Ceiling
  measured first on the current kernel (2026-10-13, `b3_multicast_ceiling_20261013.sh`, diagnostic
  `fa3_skip_kv_load.patch`): zero-filling EVERY FA3 K/V global load saves 4.9% on mixed_verify q4,
  6.6% on prefill 2048 + 32 verify rows, 5.1% on the FA3 default fixture, and 16-20% on FA3 rows of
  32 new tokens against 16-64K of context (all normal arms exact). Multicast removes only the loads
  CTAs of a cluster share, at most half for a 2-CTA cluster, so its bound is ~3% on serving mixes
  and ~10% on short-append FA3 rows. Pending before deciding: the same ceiling on agent-append
  shapes (256-1024 new tokens on 32-131K of context), `b3_append_ceiling_20261013.sh`, queued after S2b.
  ([raw](debug_kernel_server_perf/resume_20261012/b3_multicast_ceiling_20261013.txt))

### [ ] S. Server / workload evaluation

Arms everywhere: stock MTP, unfused 4-mode, fused 4-mode (co-launch, 64/32/16, auto chunk);
EAGLE3 chain q4; CUDA graphs on; 4-mode arms with the measured mixed-step barrier (see gates);
each arm reports fallbacks and composition counters. Analyzer: `drivers/serving/eval_curves.py`.

- [x] **S1 x S2, rate sweep x context mixtures (2026-10-12 19:22 -> 2026-10-13 02:57, 54 runs,
  `eval_matrix_s12_20261012.sh`):** chat 4-24/s + TraceLab agent at 0.1 sessions/s; per-tenant
  SLOs chat 2 s / 50 ms TPOT, agent 10 s / 80 ms. 0 fallbacks, 0 tracebacks in all 54 runs.
  - [x] **mix32 (32-64K):** arms equal at 4-8/s (4-mode P99 ITL worse: a verify rides with a
    chunk). From 12/s fused vs stock: TTFT -47..-79%, throughput +11..+25%, TPOT -19..-28%, SLO
    attainment up to 2x (agent 0.83 vs 0.47 at 16/s); fused beats unfused 4-mode from 16/s.
  - [x] **mix64 (64-128K):** already saturated at 4/s (the agent tenant alone fills the GPU). At
    12-20/s fused vs stock: chat throughput +15..+36%, TTFT -31..-35%, TPOT -19..-30%; agent
    throughput +18..+26%, TPOT -11..-25%; fused beats unfused by 9-14% chat throughput at 16-20/s.
  - [x] **mix128 (128-256K, YaRN x8), 2026-10-13:** the largest gains -- fused vs stock at
    4-24/s: chat throughput +11..+49%, TTFT -35..-46%, TPOT -22..-41%; agent throughput
    +10..+38%, TPOT -11..-31%, TTFT -8..-30%. At chat 24/s every chat metric clears the bar
    (+49% / -46% / -41%) and the agent reaches -30% TTFT / -31% TPOT. Fused beats unfused 4-mode
    at every rate.
  - [x] **Where the speedup comes from, per context range (2026-10-13):** each stock->fused change
    split into scheduling (stock MTP -> unfused 4-mode) and kernel (unfused 4-mode -> fused), from
    `eval_s12_v2_20261012.csv`. Acceptance is not a source: tokens/verify is within 1-2% across the
    three arms everywhere (chat ~2.03-2.08, agent ~3.1-3.7; stock marginally highest).
    | range | scheduling share | kernel share |
    |---|---|---|
    | mix32 (32-64K) | carries the onset: at 12/s chat TTFT -46% / TPOT -34% all scheduling; at 20-24/s TTFT -25..-32% | appears from 16/s: TTFT -29..-45%, TPOT -9..-15%, throughput +8..+11% at 20-24/s |
    | mix64 (64-128K) | steady: TTFT -18..-24%, TPOT -14..-20%, throughput +8..+20% | grows with load: TTFT -9..-21%, TPOT -5..-17%, throughput +3..+19% (chat 24/s: +19% kernel vs +13% scheduling) |
    | mix128 (128-256K) | small and for the agent zero or negative (agent TTFT +15..-9%) | dominant: chat TTFT -27..-33%, throughput +17..+25% at 8-24/s; almost all agent TTFT/throughput gain |
    Mechanism. *Scheduling:* stock sglang keeps mixed chunk on for EAGLE3 but degrades running
    requests to 1-token decodes in a mixed step (no speculation), or runs verify in separate steps;
    the 4-mode step keeps them speculating beside the prefill chunk. That pays most where steps
    are short and many requests are decoding (mix32 onset) and saturates once the prefill chunk
    dominates step time. *Kernel:* unfused 4-mode runs the whole merged batch through one FA2
    BatchPrefill, so long-context prefill rows run on the FA2 path; the fused kernel sends them to
    FA3 (WGMMA) on 112 SMs and co-launches FA2 for the verify rows. Attention's share of the step
    grows with context (B1 at mix32: 75-80% of kernel time; fused attention 2739 ms vs unfused 3574
    ms over 60 profiled steps, -23%), so the kernel's share of the gain grows with context: second
    to scheduling at 32-64K, comparable at 64-128K, dominant at 128-256K. Below capacity
    (mix32/mix64 at 4-8/s) neither applies and the arms are equal or 4-mode is slightly worse on
    TPOT (a verify riding with a chunk). Per-range kernel attribution at 128K+ is being profiled
    (`b1_tail_profile_mix128_20261013.sh`).
  - [x] **Figures (2026-10-13):** [`eval_s12_v2_20261012.png`](debug_kernel_server_perf/resume_20261012/eval_s12_v2_20261012.png) -- per
    mixture and tenant: throughput vs median tok/s/user, SLO attainment, mean TTFT and TPOT vs chat
    rate (data [`eval_s12_v2_20261012.csv`](debug_kernel_server_perf/resume_20261012/eval_s12_v2_20261012.csv), `drivers/serving/plot_eval.py`).
    Fused is lowest on TTFT and TPOT at nearly every loaded point of all three mixtures. Two
    limits of this sweep, to fix in the next pass: (1) the agent-tenant Pareto panels are not
    meaningful -- the swept rate is the chat rate, the agent rate is fixed, so agent throughput
    falls as chat load rises; the agent tenant needs its own rate sweep (the full-length run does
    this); (2) a fixed 2 s chat TTFT SLO is met by no arm at 64-128K and above, so SLO attainment
    stops discriminating there -- SLOs should scale with each workload's unloaded latency
    (SLOs-Serve style, e.g. 5x / 10x).
  - [ ] **Relative SLOs (2026-10-13):** `eval_curves.py --slo-relative K_TTFT,K_TPOT --ref-root`
    scales each tenant's SLO from its median TTFT/TPOT running ALONE on stock MTP. The swept runs'
    lowest rate cannot be the reference (mix128 chat already waits 38 s at 4/s; tried and
    discarded). Unloaded runs, chat 0.5/s and agent 0.02 sessions/s per mixture,
    `unloaded_ref_20261013.sh` (driver gets a `STREAMS` filter, `streams_filter.patch`) -- queued.
- [ ] **S2b, full-length TraceLab (user, 2026-10-13):** the release as it is -- Claude Code context
  p50/p90/p99 133K/483K/923K (25% >= 256K), Codex 120K/204K/233K, outputs p50 ~200 tokens; the
  32-64K slice is ~16% of rounds. Loader replays both providers, first rounds, round-weighted
  sessions; `tracelab_full_env` at YaRN x32 (1M tokens), agent tenant alone, rate = session starts/s.
  - [x] **v1 aborted (2026-10-13 03:16):** 120 sessions x 2 rounds at 0.2-0.8/s queued 16.8M prompt
    tokens at ~2K tok/s -- one 1M-token session alone is ~400 s of prefill; could not finish.
  - [x] **v2 aborted (2026-10-13 04:54):** 24 sessions x 10 rounds, seed 187 (loader-checked p10/50/90/99
    40K/127K/235K/869K). The stock arm hit its 90-min timeout having prefilled **14.4M** tokens for a
    replay with 5.8M uncached tokens: the KV pool (1,823,088 tokens) cannot hold 24 sessions reaching
    869K, so prefixes were evicted between rounds (usage 0.99, no retractions) and rounds re-prefilled.
    No data. Lesson: a single-GPU replay must keep the sessions' combined peak context in the pool.
  - [ ] **v3 (queued after the race-fix check and B1):** 8 sessions x 30 rounds, seed 1014 --
    combined peak 1.37M tokens; loader-checked context p10/50/90/99 38K/123K/223K/390K vs the
    release's 38K/124K/256K/822K. The >400K tail cannot be co-resident with other sessions on one GPU
    (excluded by KV capacity, not by a length filter) and belongs to the multi-GPU stage. YaRN x16
    (524288), session rates 0.005 and 0.02/s, three arms, 4-mode arms on the verdict's tree
    (`tracelab_full_v2b_20261013.sh`, root `tracelab_full_v3_20261013`).

- [ ] **S3, standalone workloads:** ShareGPT (4-32/s), GSM8K, MATH-500, HumanEval, Code-Feedback
  (8-32/s), Mooncake conversation / tool-agent (recorded timestamps at 0.5-2x speed), LongBench v2
  (0.25-1/s, documents that do not fit 131072 dropped); three arms each, YaRN x4.
  `s3_standalone_20261013.sh` (`s3_standalone_knobs.patch`: Mooncake slowdown and LongBench length
  filter were not plumbed) -- queued last.

- [ ] **S4, EDF step model** re-fitted on the fused kernel and compared with the unfused schedule
  (goal point 3); earlier fits used the RoPE-corrupted model and are void (incl. the uncommitted
  2026-10-12 05:10 "fused-vs-unfused EDF check" in `server_perf_model.md`). The STEPTRACE emitter
  did not survive the re-port; ported to the overlap path with the 4-mode verify rows counted
  (`steptrace_port.patch`). `s4_edf_20261013.sh`, on the race-fixed tree without the barrier (a
  per-step sync would distort the modelled step times), runs only if the race fix passes: traces
  on both 4-mode arms (mix32 8/16/24/s, mix128 12/s), one linear model per arm (held-out MAPE),
  then EDF-R with each arm's own model vs no EDF at mix32 16/24/s -- queued after S3.

- [x] **Chunk size is not the ITL lever at saturation (2026-10-12):** chunk 8192/4096/2048 leaves
  both arms unchanged (prefill per step is many ~512-token prompts).
- [ ] **Direct baselines:** MuxWise (arXiv 2504.14489), semi-PD, NVIDIA Dynamo conditional
  disaggregation and sglang PD -- see References.
  - [x] **sglang mixed-chunk is already the stock arm (2026-10-13):** upstream keeps
    `--enable-mixed-chunk` on for EAGLE3 and degrades running requests to plain 1-token decodes
    in a mixed step. The S1 `MTP_LPT` server log shows `enable_mixed_chunk: True`, no disable
    warning, and 3151 prefill batches carrying running requests (mix32 16/s). So every "stock
    MTP" number is stock mixed-chunk + MTP; the 4-mode step's difference is that those requests
    keep speculating (verify rows) instead of dropping to single-token decode.
  - [ ] **MuxWise = sglang `--enable-pdmux` (2026-10-13):** no public MuxWise repo; upstream
    sglang's PD-multiplexing (GreenContext SM split, layer-split prefill) is that design and is in
    our fork. Non-speculative only, chunked prefill and overlap off, and it warns of degradation
    with CUDA graphs on torch > 2.6 (pod: 2.14). Arms `PDMUX` vs stock non-speculative `LPT` on
    mix32 / mix128 at chat 8/16/24/s, S1 layout -- `baselines_pdmux_20261013.sh`, queued after the
    chain.
  - [ ] **semi-PD -- blocked on a user decision (2026-10-13):** fork of sglang v0.4.4.post1
    (torch 2.5 / cu124), MPS-based SM partition, radix cache off, no speculative decoding. Venv
    `venv_semipd` + clone are on the pod (torch 2.5.1+cu124 installed), but its pinned
    `sgl-kernel==0.0.5` no longer exists on PyPI (every 0.0.x release deleted), so it must be built
    from the repo's `sgl-kernel/`, which needs a CUDA 12 nvcc (pod: 13.1). Installing NVIDIA's
    CUDA 12.4 toolkit runfile was denied by the permission policy; needs the user's go-ahead (or
    a CUDA 12 toolchain provided another way).
  - [ ] **Dynamo conditional disaggregation -- optional (user, 2026-10-13):** not mandatory when it
    does not support sglang. Dynamo has an sglang worker, but conditional disaggregation (local
    prefill on the decode worker) is documented for vLLM / TRT-LLM only, so sglang PD is the
    disaggregation baseline; Dynamo stays a discussion point (PD doc point 6).

### [~] Supporting gates

- [x] **RoPE root cause (2026-10-12):** the YaRN override dropped `rope_theta` (1e6 -> 1e4) under
  transformers 5.x; fixed in sglang's config loader with a regression test. All serving numbers
  before 2026-10-12 are void. ([evidence](debug_kernel_server_perf/resume_20261012/a2_serving_ropefix_20261012.md))
- [x] **Serving-output gate (q4 and q8; graphs off and on):** every 4-mode arm agrees with stock on
  the long prompts; first tokens are the HF reference argmax or within one bf16 ulp.
- [x] **Graphs-on 4-mode crash (2026-10-12; cause found and fixed 2026-10-13):** both 4-mode arms crash under
  load (async index assert). Bisect: a device sync after mixed steps (or launch-blocking) cures it;
  phase syncs, no in-step graph, synchronous flashinfer plans and a sync after non-mixed steps do
  not. **Cause (code, 2026-10-13):** in the overlap loop the verify half's
  `future_map.publish(new_seq_lens)` and relay stash run after the forward-stream block, i.e. on
  the schedule stream, unordered against the forward stream that is still producing those
  tensors; stale seq_lens get published and the next step resolves them into out-of-range
  indices. The mixed-step sync orders exactly this. Fix: issue both on the forward stream
  ([patch](debug_kernel_server_perf/resume_20261012/racefix_scheduler.patch), tree `up_racefix`).
  **Retracted:** "overlap off does not cure" -- that arm died at its FIRST mixed step on
  `assert next_token_ids.is_cpu` (the synchronous path had no 4-mode handoff at all; now added),
  so it never tested the race. Validation (`graphs_racefix_20261013.sh`), 2026-10-13:
  - [x] **null case:** unfixed tree, no barrier, MTPMIX_LPT chat 24/s + agent 0.1/s -> crashed
    (`Assertion -sizes[i] <= index && index < sizes[i]`, 1 traceback), so the race reproduces.
  - [x] **output gate, fixed tree, graphs on, no barrier:** PASS (long prompts agree on >= 96
    tokens with stock `MTP_LPT_Q4G`).
  - [x] **fixed tree, no barrier (05:02-05:23):** MTPMIX_LPT 1651 / 1701 mixed steps, MTPMIXFUSED
    1751 / 1751, 0 tracebacks each, CUDA graphs on; synchronous path (`--disable-overlap-schedule`)
    1651 mixed steps, 0 tracebacks. **Verdict PASS.** From here every 4-mode arm runs on
    `up_racefix` without the mixed-step barrier (`tree_env.sh`); S1 x S2 numbers above were taken
    WITH the barrier, which can only have slowed the 4-mode arms.
    Tree: sglang-up `cc739bd2d` + `racefix_scheduler.patch` + `racefix_fs_upload.patch`
    (+ `steptrace_port.patch`, off unless `SGLANG_STEP_TRACE=1`); command
    `bash graphs_racefix_20261013.sh` then `bash racefix_verdict.sh` (pod `/dev/shm/w`). Until it passes, measurements keep the mixed-step barrier. The latent race
  of the same class -- `_fs_upload` rewrote its reused pinned host buffer while the previous step's
  non-blocking copy out of it could still be queued -- now waits on a per-buffer copy event first
  ([patch](debug_kernel_server_perf/resume_20261012/racefix_fs_upload.patch), same tree and validation).
- [x] **Pod re-forked (2026-10-12):** `f096519a4d17aa1c` expired; `095f01a3e1585eea` rebuilt from the
  fork's upstream base + serving diff, reproducing the old pod's kernel results within 1%.

---

## Evidence index

- Per-result evidence (commands, SHAs, tables, artifacts): `tasks/debug_kernel_server_perf/resume_20261012/`
  -- `colaunch_20261012.md` (A-fix, class set, KV chunk, serving), `postfix_realistic_ab_20261012.md`
  (post-RoPE A/B, FA2 spill diagnosis), `a2_serving_ropefix_20261012.md` (RoPE root cause),
  `a3_winner_map/` (A3 tables); earlier days under `resume_20261011/`.
- Full working log to 2026-10-12, verbatim (history, retractions, bring-up):
  [`archive/debug_kernel_server_perf_4_mode_log_to_20261012.md`](archive/debug_kernel_server_perf_4_mode_log_to_20261012.md).
- Scheduling argument (PD colocation vs disaggregation, points 1-5): `tasks/pd_colocation_vs_disagg.md`.

---

## References

Papers and systems this work draws on. *(sent)* marks the ones the user sent.

- **Taming Dynamism on GPUs: Cross-SM Kernel Fusion via SM Cooperation and Just-in-Time Reduction**
  *(sent; PDF)* -- note: technique. Taken: one persistent launch whose CTAs pull heterogeneous,
  dynamically sized work from a device-side queue (our four-task plan and atomic cluster cursor,
  and cross-role backfill); just-in-time reduction of split-KV partials (built, measured at <=0.6%
  of GPU time, replaced by flashinfer 0.7's scheduled merge); its task representation as the
  template for an optional epilogue task (EAGLE tree build, item B). Not taken, by measurement:
  co-compiling roles with different register needs into one body -- our FA2 role spilled 21x
  there, so FA2 runs as a PDL-co-launched executor and backfill is done by the hardware scheduler.
- **SLOs-Serve: Optimized Serving of Multi-SLO LLMs** -- S. Chen, Z. Jia, S. Khan, A. Krishnamurthy,
  P. B. Gibbons, [arXiv:2504.08784](https://arxiv.org/abs/2504.08784) *(sent)* -- note: setting.
  Application- and stage-specific SLOs, several application mixtures (summarization, coding,
  chat, tool calling, reasoning) crossed with load, capacity measured under SLOs (goodput / SLO
  attainment): the source of S1-S3's per-tenant SLOs and mixture x rate sweeps. Its chunked-prefill
  planner that co-plans speculative decoding is the closest scheduling analogue of our 4-mode step.
- **Towards High-Goodput LLM Serving with Prefill-decode Multiplexing (MuxWise)** -- Y. Chen,
  W. Cui et al., [arXiv:2504.14489](https://arxiv.org/abs/2504.14489) *(sent; "the ASPLOS PD
  multiplexing paper")* -- note: baseline. Intra-GPU prefill/decode multiplexing with an SLO-aware
  dispatcher and a contention-tolerant estimator; goodput under SLO as the metric. Direct
  colocation baseline (not yet run); its estimator is the comparison point for our EDF step model.
- **ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System** -- H. Kang et al.,
  [arXiv:2602.13692](https://arxiv.org/abs/2602.13692) *(sent)* -- note: argument and setting. KV-cache
  thrashing in agentic serving (a session's cache evicted during a tool call and re-prefilled, up
  to 7.14x end-to-end): motivates PD point 2 (migration leaves replicas where demand is) and
  evaluating on multi-turn agent sessions rather than independent prompts.
- **TraceLab: Characterizing Coding Agent Workloads for LLM Serving** -- K. Zhu, M. Jacob, C. Ma,
  Y. Pan, S. Wang, A. Krishnamurthy, B. Kasikci, [arXiv:2606.30560](https://arxiv.org/abs/2606.30560);
  dataset `syfi_coding_trace.jsonl.gz` ([github](https://github.com/uw-syfi/TraceLab)) *(sent)* --
  note: workload. The coding-agent tenant in every mixed run (causal replay of prefix / append /
  output tokens per round); its statistics set the mixtures and the full-length replay (context
  p50 ~124K, a quarter of Claude rounds >= 256K, short outputs, high prefix hit) and PD point 5.
- **semi-PD: Towards Efficient LLM Serving via Phase-Wise Disaggregated Computation and Unified
  Storage** -- K. Hong et al., [arXiv:2504.19867](https://arxiv.org/abs/2504.19867)
  ([code](https://github.com/infinigence/Semi-PD)) -- note: closest intra-GPU design. Splits each
  GPU's SMs between a prefill and a decode process over one shared KV store, so there is no KV
  transfer and no weight replication. Difference from ours: its SM ratio is a deployment-level
  setting changed by relaunching (criticised as static in arXiv:2504.19516), while our split is
  decided per step inside one forward, co-runs prefill chunks with MTP verify rows, and lets the
  hardware backfill FA2 work onto SMs FA3 vacates.
- **NVIDIA Dynamo -- KV router, conditional disaggregation, planner, KVBM**
  ([conditional disagg](https://docs.nvidia.com/dynamo/advanced-customizations/conditional-disaggregation);
  [router](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/routing-concepts);
  [planner](https://docs.nvidia.com/dynamo/v1.5.0/knowledge-base/modular-components/planner/overview);
  [delta-KV RFC](https://github.com/ai-dynamo/dynamo/discussions/7861)). Full policy-by-policy
  comparison: `pd_colocation_vs_disagg.md` point 6 (block-count cost router with no SLO term,
  static local-prefill thresholds, minute-scale P/D autoscaling, incidental replication) -- note: closest
  multi-node baseline for agentic rounds. The KV-aware router prefills locally on the decode worker
  when the uncached part of the prompt is small (default effective-ISL threshold 4096 tokens) or
  remote prefill is loaded, so most agent append rounds avoid the KV transfer that PD point 5
  measures; the RFC would cut remote rounds to append-only transfer. What remains ours: making that
  local prefill + speculative decode co-execution efficient (4-mode step + fused kernel, S1:
  +20-25% throughput, -50-80% TTFT vs stock MTP under load), symmetric nodes instead of fixed
  prefill/decode pools (routing and replication free to use every node), and EDF/SLO-aware
  admission. Must be run as a baseline, not argued around.
