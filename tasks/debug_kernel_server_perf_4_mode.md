# Current goal

Fused FA3+FA2 SM pre-assignment attention for a 4-mode MTP (EAGLE3) serving schedule, optimised
on real serving workloads, then the EDF model and the PD-colocation argument
(`tasks/pd_colocation_vs_disagg.md`). Bar: >=30% TTFT and ITL, >=10% throughput on realistic
chat + TraceLab workloads. Single H200 trial `095f01a3e1585eea` (`vibecuda`); H20 multi-node only
after the kernel items close (<=16 GPUs).

**Where it stands (2026-10-13, updated 08:00):** the serving kernel is now the two-launch split
(FA3 prefill rows + stock FA2 BatchPrefill verify rows): vs stock MTP it clears the bar at mix32
16/24 and mix128 24/s for both tenants (e.g. mix128 24/s chat +56% tput / -55% TTFT / -45% TPOT)
and misses only agent TTFT/TPOT at mix128 16/s (-26%). Earlier summary, fused co-launch: under load the fused 4-mode arm beats stock MTP by TTFT
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
  - [x] **A3b on the serving split (Codex kernel worker, 2026-10-13, all 12 arms):** DEFAULT vs A3B
    on `MTPMIX_SPLIT`, interleaved twice at mix32/mix64/mix128 x chat 24/s + agent 0.1/s; both rules
    pass the q4 output gate (long prefix 52), 0 fallbacks / tracebacks. A3B vs DEFAULT: mix32 chat
    TTFT -26.7%, agent tput +5.4% / TTFT -25.5%; mix64 chat TTFT -14.1%, agent tput +11.2%;
    mix128 chat tput +6.1% / TTFT -28.2% / P99 ITL -33.3%, agent tput +10.9%. **Not promoted as a
    universal switch:** A3B replaces the whole rule (incl. the decode-population guard), mean chat
    interactivity falls (mix128 -16.9%), agent fixed-SLO attainment drops at mix128 (0.19 vs
    0.44/0.31), acceptance differs between arms, and only 16-24 agent requests per arm.
    ([findings](debug_kernel_server_perf/resume_20261012/a3b_split_findings_20261013.md), `a3b_split_serving_20261013.sh`,
    `a3b_split_midrange_20261013.sh`.) My older `a3b_serving_ab_20261013.sh` is superseded.
  - [ ] **A3 at guard 0, clean A/B (queued 2026-10-15):** the A3b test above compared A3B against
    DEFAULT with the decode guard ON, and A3B drops the guard, so its gains are not attributable to
    the fitted thresholds; guard 0 is now the default. Both rules at guard 0 on the split (chunk
    3072, no EDF): q4 gate per rule, interleaved DEFAULT/A3B x2 on the cohort at C=16 + chat 8 req/s,
    one pair on TraceLab C=8 (`a3_guard0_20261015.sh`).

- [x] **A4, FA2 verify-row bandwidth (2026-10-13; closed by measurement 2026-10-15):** serving runs
  verify rows on stock FA2 BatchPrefill (63-83% of HBM) inside the two-launch split, and every fused /
  PDL / entry / locality / unpacked candidate measured no serving gain over it (sub-items below). The
  persistent co-launched FA2 executor stays ~20% behind stock (a mainloop gap) -- relevant only if the
  fused co-launch is revived. attention is 75-80% of serving kernel time,
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
    **mix32 24/s (07:30, both arms clean; split 103,501 split calls, fused 80,801 fused calls):**
    | metric | split | fused | stock MTP (S1) | split vs stock |
    |---|---|---|---|---|
    | chat output tok/s | **1641** | 1489 | 1226 | +34% |
    | chat TTFT mean | **8684 ms** | 33662 ms | 63512 ms | -86% |
    | chat TPOT mean | **20.7 ms** | 27.7 ms | 34.8 ms | -40% |
    | agent output tok/s | **212** | 162 | 130 | +64% |
    | agent TTFT mean | **8768 ms** | 22597 ms | 43663 ms | -80% |
    | agent TPOT mean | **13.4 ms** | 16.7 ms | 23.8 ms | -44% |
    At both mix32 points the split clears the bar vs stock MTP on TTFT, TPOT and throughput for
    both tenants.
  - [x] **Split v2 complete -> serving kernel = the two-launch split (2026-10-13 08:00):** all 8 arms
    clean (0 tracebacks, 0 fallbacks; split 23.5K-103.5K split calls, fused 23.0K-80.8K fused
    calls). Split vs stock MTP (S1 numbers for stock):
    | point | chat tput | chat TTFT | chat TPOT | agent tput | agent TTFT | agent TPOT |
    |---|---|---|---|---|---|---|
    | mix32 16/s | = (offered load) | -97% | -63% | +90% | -87% | -48% |
    | mix32 24/s | +34% | -86% | -40% | +64% | -80% | -44% |
    | mix128 16/s | +31% | -54% | -33% | +29% | -26% | -26% |
    | mix128 24/s | +56% | -55% | -45% | +46% | -37% | -32% |
    The bar is met everywhere except the agent tenant at mix128 16/s (-26%). Split vs fused
    co-launch: mix32 chat TTFT -74..-86%, TPOT -25..-48%, agent tput +31..+50%; mix128 chat TTFT
    -14..-18%, tput +5..+6%, agent TPOT +1..+9% (fused better there). The gap shrinks with context
    because long prefill rows (FA3 in both arms) dominate; the split's edge is the verify rows
    (stock FA2 at 63-83% of HBM vs the executor's 38-52%). Data:
    [`split_vs_fused_v2_20261013.csv`](debug_kernel_server_perf/resume_20261012/split_vs_fused_v2_20261013.csv).
    **Overlap A/B queued (2026-10-13 08:1x):** `SGLANG_SPLIT_OVERLAP=1` runs FA3 on a side stream
    holding all but `SGLANG_FUSED_SM_MARGIN` SMs while stock FA2 runs on the main stream
    (`split_overlap.patch`; logged as `overlapped=` in FUSED_SPLIT_CALLS). Gate on the m16 arm, then
    split / overlap m16 / m32 at mix32 and mix128 x 24/s (`split_overlap_20261013.sh`). The queue is
    paused after the full-length replay's stock arm (`stop_v3.sh`) so the replay's and S3's 4-mode
    arms use the winner.
  - [x] **Overlap A/B: no gain, overlap stays off (2026-10-13 08:36-09:22):** gate PASS on the m16
    arm (long prompts agree on 52 tokens; 3001/3001 split calls overlapped). All 6 serving arms
    clean, overlap arms 100% overlapped (e.g. 35,501/35,501 at mix128).
    | 24/s | chat TTFT split / m16 / m32 | chat TPOT | agent tok/s |
    |---|---|---|---|
    | mix32 | 8944 / 8614 / 8675 ms | 20.9 / 20.4 / 20.8 ms | 211 / 215 / 212 |
    | mix128 | 68130 / 68108 / 68064 ms | 31.2 / 31.4 / 31.3 ms | 80.6 / 80.6 / 80.6 |
    mix32 differences are inside the +/-1.5-3% noise floor; mix128 within 0.3%. In these steps one
    role dominates (verify FA2 at mix32, prefill FA3 at mix128), so concurrent execution has
    little to hide. The serving kernel stays the plain two-launch split.
    ([csv](debug_kernel_server_perf/resume_20261012/split_overlap_20261013.csv),
    `split_overlap_20261013.sh`, `split_overlap.patch`)
    Originally planned next (A4): overlap the split's two launches (FA3 on a side stream with `sm_margin`, stock FA2
    on the SMs it leaves) and re-run the S1 x S2 sweep with the split as the 4-mode arm.
  - [x] **Chunk floor rejected (2026-10-13, all exact):** a fixed 4096 floor wrecks small-KV cases
    (1 row x 16K: auto 0.028 ms vs 4096 0.099 ms; 16 x 4K q8: 0.094 vs 0.163 ms) while helping
    long verify rows; and the two-launch control beats the fused executor on every A2 fixture too
    (e.g. 16 x 16K q8: 0.156 vs 0.323 ms). With the split as the serving kernel the auto rule is
    moot for serving (`fa2_chunk_floor_20261013.sh`).
    Chat throughput at 16/s equals the offered load in all arms (~1136 tok/s). The barrier-free fused arm
    (4932 ms TTFT) is within 7% of S1's fused-with-barrier (5276 ms). Command:
    `INHERIT_LOCK=1 bash split_vs_fused_20261013.sh` (root `split_vs_fused_v2_20261013`), analysed
    with `drivers/serving/eval_curves.py --slo '{"chat":[2000,50],"codex":[10000,80]}'`.

  - [x] **Kernel candidates vs the split (Codex kernel worker, 2026-10-13/14), none promoted:**
    - native nonpersistent FA2 as the PDL dependent of an SM-limited FA3 grid: v1 failed (planner
      reused the FA2 chunk for FA3 -> 512/512 nonfinite), fixed in v2 (0/544 bad); closes much of
      the FA2 gap (46 q4 x 48K: 1.495 vs stock two-launch 1.527 ms) but not the FA3-dominated one
      (prefill2048+8: 5.872 vs 3.950 ms) ([proto](debug_kernel_server_perf/resume_20261012/native_pdl_prototype_20261013.md)).
      Serving v3 with split-KV FA3: no mean gain (mix32 chat TTFT +9.8%)
      ([v3](debug_kernel_server_perf/resume_20261012/native_pdl_v3_serving_20261013.md)).
    - dedicated FA3 entry: -9..-23% kernel time vs the combined entry, still behind the unfused
      control on large-query fixtures ([entry](debug_kernel_server_perf/resume_20261012/native_fa3_entry_20261013.md)); compact planning
      validated ([compact](debug_kernel_server_perf/resume_20261012/native_pdl_compact_20261013.md)); combined serving: no general win
      (mix64 agent +9.7% is acceptance-confounded, 2.753 vs 3.177)
      ([combined](debug_kernel_server_perf/resume_20261012/native_pdl_combined_20261013.md)).
    - `auto4096` cap (min(auto, 4096)), opt-in: -20..-27% on long verify fixtures, +9..+18% serving
      tput vs the uncapped co-launch -- moot for the split, which uses stock FA2
      ([cap](debug_kernel_server_perf/resume_20261012/fa2_auto4096_candidate_20261013.md)).
    - FA3 diagnosis: NCU shows custom FA3 at ~43% tensor activity vs stock 65% (stock: KV96, 2
      stages, 132 CTAs) ([ncu](debug_kernel_server_perf/resume_20261012/fa3_ncu_20261014.md)); round-robin work pairs raise L2 hit
      11.9% -> 94.9% and cut kernel time -14% / -26% on two fixtures
      ([locality](debug_kernel_server_perf/resume_20261012/fa3_locality_20261014.md)); KV96 is +0.1..+4% slower than KV128 but exposed a
      real packed-Q bug (Q gather looped over the KV width, leaving 32 Q columns unwritten), fixed
      ([kv96](debug_kernel_server_perf/resume_20261012/fa3_kv96_prototype_20261014.md)); SM budget: keep 112
      ([budget](debug_kernel_server_perf/resume_20261012/tma_sm_budget_20261013.md)); one-warp TMA producer: no gain
      ([warp](debug_kernel_server_perf/resume_20261012/tma_warp_prototype_20261014.md)).
    - [x] paired serving split vs native PDL + dedicated entry + round-robin pairs (complete
      2026-10-14 04:4x, two repeats): no uniform gain -- 32-64K and 64-128K within +/-4% on every
      metric; 128-256K agent ITL -12.4% but agent TTFT +6.3%, chat tput -1.9%; split stays
      ([paired](debug_kernel_server_perf/resume_20261012/native_pdl_paired_serving_20261014.md)).
    - [x] unpacked query heads for FA3 (complete 2026-10-14 04:40, 16 gates + 40 timings pass):
      KV128 within 0.6% of packed auto on the four large fixtures, KV96 slower, unsplit 3.7x slower
      on irregular short queries; packed auto retained
      ([unpacked](debug_kernel_server_perf/resume_20261012/fa3_unpacked_20261014.md)).

### [x] B. Fuse draft/verify epilogue ops

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
  - [x] **True GEMM-epilogue fusion bounded (2026-10-15, user: "fuse them if you can get a real
    gain over PDL"):** the existing fused verify/chain epilogues are separate Triton launches, not
    epilogues. The largest candidate for a real epilogue, the target argmax inside the LM-head GEMM
    (Qwen2.5-7B, vocab 151936), can save at most the argmax pass plus the logits round trip:
    19 / 72 / 235 us at 48 / 184 / 1024 verify tokens (LM-head GEMM 270 / 338 / 1717 us), i.e.
    ~0.2-1% of a 20-40 ms serving step -- below the 2-3% serving noise floor, so no measurable
    gain over PDL chaining; not built. bf16-logit argmax disagrees with an fp32 reference on 2-3%
    of rows (ties), so an fp32-accumulator epilogue would be more exact, not faster.
    (`b1_lmhead_epilogue_ceiling_20261015.py`, log `b1_lmhead_ceiling_20261015.log`)
- [x] **B2, cross-role backfill (2026-10-12):** co-launch hardware backfill worth up to 2.9x when FA2
  work exceeds the slots beside FA3, nothing on small mixes; in-kernel `tma_store_wait` backfill is
  correct and worth 1.4-5.5x over none, but that whole path is 2-6x slower. Co-launch ships.
  ([evidence](debug_kernel_server_perf/resume_20261012/colaunch_20261012.md))
- [x] **B3, multicast:** implemented by the Codex kernel worker on 2026-10-13 (true TMA multicast at page 128, below) and closed by measurement; until then `mcast_mask=0`. Ceiling
  measured first on the current kernel (2026-10-13, `b3_multicast_ceiling_20261013.sh`, diagnostic
  `fa3_skip_kv_load.patch`): zero-filling EVERY FA3 K/V global load saves 4.9% on mixed_verify q4,
  6.6% on prefill 2048 + 32 verify rows, 5.1% on the FA3 default fixture, and 16-20% on FA3 rows of
  32 new tokens against 16-64K of context (all normal arms exact). Multicast removes only the loads
  CTAs of a cluster share, at most half for a 2-CTA cluster, so its bound is ~3% on serving mixes
  and ~10% on short-append FA3 rows. Pending before deciding: the same ceiling on agent-append
  shapes (256-1024 new tokens on 32-131K of context), `b3_append_ceiling_20261013.sh`, queued after S2b.
  ([raw](debug_kernel_server_perf/resume_20261012/b3_multicast_ceiling_20261013.txt))
  - [x] **Agent-append ceiling (Codex, 2026-10-13):** zero-KV-load headroom on append shapes is
    -19..-22% (4 x q512 @ 64K 6.192 -> 4.812 ms) -- a ceiling, not a multicast gain
    ([append](debug_kernel_server_perf/resume_20261012/b3_append_findings_20261013.md)).
  - [x] **True TMA multicast implemented and gated (Codex, 2026-10-13):** `cp.async.bulk.tensor`
    multicast over 2-CTA clusters at page 128, 9 gates + 48 timing cases pass (58 multicast tiles
    on CTA0); **no clear benefit: multicast -0.2..+1.7% vs unicast TMA**
    ([multicast](debug_kernel_server_perf/resume_20261012/tma_multicast_prototype_20261013.md)); page-128 serving integration passes all
    5 EAGLE3 arms with up to 188 multicast tiles executed
    ([paged serving](debug_kernel_server_perf/resume_20261012/tma_paged_serving_20261013.md)). Software sharing via DSMEM exchange loses
    badly (4 x 512 @ 64K: 18.21 vs redundant 5.94 ms; bulk exchange still 60-63% slower)
    ([dsmem](debug_kernel_server_perf/resume_20261012/kvshare_prototype_20261013.md), [bulk](debug_kernel_server_perf/resume_20261012/kvshare_bulk_20261013.md)).
    **B3 closes by measurement: multicast is built and does not pay on these shapes.**

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
  - [x] **Session-rate axis retracted (2026-10-15):** `tracelab_full_final_20261015.sh` swept
    "0.005 / 0.02 sessions/s" (Poisson session starts, rounds back to back), an axis no reference
    uses (Crossflow: req/s; ThunderAgent: concurrency). It controls neither: at 0.005/s the 8 sessions averaged 1.3-1.5 requests in flight (MTP / split), a
    near-idle server. Cancelled by the user after 2 of 6 arms; those numbers are not a result.
    Agent load is program concurrency only (ThunderAgent; references).
  - [x] **Concurrency sweep (2026-10-15 12:36-15:56, all 18 arms clean, peak = C every arm):** same 8x30-round seed-1014 replay, program concurrency
    C = 2/4/8 (KV pool caps C at p99 390K context), arms MTP | split static 3072 | split + EDF +
    floor 2048; then the frozen 32-64K cohort at C = 8/16/24 + ShareGPT 8 req/s (replaces the
    retracted EDF load points). Running since 12:36: `concurrency_sweep_20261015.sh`, root
    `concurrency_sweep_20261015`; each arm logs `PROGRAM_CONCURRENCY` (peak must equal C).
    Partial (TraceLab, peak = C and mean in-flight 1.99-2.00 / 3.76-3.79 in every arm, 0 tracebacks /
    fallbacks): **C=2** split / EDF vs MTP: tput +7.7% / +13.9%, mean TTFT +4% / -12%, TPOT -23% /
    -22%, mean ITL -19% / -13%, P99 ITL 28 -> 85 / 88 ms. **C=4**: tput +29% / +26%, mean TTFT
    **+15% / +14%** (median -22% / -17%, P99 -25% / -27%), TPOT -29% / -31%, mean ITL -34% / -31%,
    P99 ITL -67% / -73%. TTFT by uncached tokens at C=4: cache-hit rounds (<1K uncached) keep a
    170-200 ms median in all arms but their mean rises 1.1 s (MTP) -> 1.7 s (split) / 3.1 s (EDF):
    head-of-line blocking -- an in-progress chunked prefill takes the whole 2-3K budget every step
    and EDF never ranks waiting requests against it. Fix under test: `edf_reserve_earlier.patch`
    (`SGLANG_EDF_RESERVE_EARLIER`); its A/B `edf_reserve_20261015.sh` was cancelled at 16:3x as a
    duplicate of the parallel session's `edf_reserve_ab_20261015.sh` (`edf_reserve_earlier_deadlines.patch`).
    That variant ranks waiting requests by their enqueue deadline from TOTAL prompt length and fired
    once in the whole C=4 run (`EDF_RESERVE calls=1`); a cache-hit round of a long session is never
    earlier than the chunked prompt under that deadline. `edf_reserve_earlier.patch` rescores on
    uncached work instead -- not yet run.
    **Complete table, % vs MTP** (tput / mean TTFT / median TTFT / TPOT / mean ITL / P99 ITL):
    | point | split static 3072 | split + EDF + floor |
    |---|---|---|
    | TraceLab C=2 | +8 / +4 / -19 / -23 / -19 / +209 | +14 / -12 / -20 / -22 / -13 / +217 |
    | TraceLab C=4 | +29 / +15 / -22 / -29 / -34 / -67 | +26 / +14 / -17 / -31 / -31 / -73 |
    | TraceLab C=8 | +32 / +7 / -37 / -49 / -33 / -80 | +34 / +28 / -26 / -50 / -38 / -82 |
    | cohort C=8 agent | +24 / -23 / -24 / -37 / -23 / -81 | +25 / -22 / -25 / -42 / -27 / -82 |
    | cohort C=8 chat | +22 / -38 / -35 / -18 / -19 / -66 | +23 / **-75** / -74 / -19 / -17 / -73 |
    | cohort C=16 agent | +30 / -28 / -26 / -53 / -36 / -84 | +30 / -27 / -27 / -55 / -39 / -86 |
    | cohort C=16 chat | +28 / -37 / -35 / -25 / -25 / -66 | +29 / -55 / -53 / -25 / -24 / -73 |
    | cohort C=24 agent | +32 / -25 / -13 / -72 / -52 / -83 | +30 / -21 / -5 / -77 / -60 / -88 |
    | cohort C=24 chat | +29 / -39 / -42 / -25 / -27 / -64 | +29 / -48 / -50 / -22 / -20 / -69 |
    Mean in-flight agent requests: TraceLab 1.99-2.00 / 3.76-3.79 / 7.07-7.13; cohort 7.3-7.7 /
    13.6-14.9 / 19.2-21.0. Acceptance 2.12-2.16 (cohort), 2.89-3.02 (TraceLab) in every arm.
    **Verdict:** throughput (+22..+34%), TPOT and P99 ITL clear the bar at every loaded point; agent
    TTFT misses (-21..-28% on the cohort, worse than MTP on TraceLab from head-of-line blocking);
    chat mean ITL misses (-17..-27%). **Load caveat:** chat at 8 req/s alone exceeds capacity at
    every C (MTP chat TTFT 24-74 s), so every cohort point is overloaded; a below-capacity chat
    rate is still missing. Data [`concurrency_sweep_20261015.csv`](debug_kernel_server_perf/resume_20261012/concurrency_sweep_20261015.csv),
    `bash concurrency_sweep_20261015.sh` (pod `/dev/shm/w`), table `drivers/serving/conc_table.py
    concurrency_sweep_20261015`; sglang `118c5985b` + pod trees `up_edf_floor` (+ ported replay).

- [ ] **S3, standalone workloads:** ShareGPT (4-32/s), GSM8K, MATH-500, HumanEval, Code-Feedback
  (8-32/s), Mooncake conversation / tool-agent (recorded timestamps at 0.5-2x speed), LongBench v2
  (0.25-1/s, documents that do not fit 131072 dropped); three arms each, YaRN x4.
  `s3_standalone_20261013.sh` (`s3_standalone_knobs.patch`: Mooncake slowdown and LongBench length
  filter were not plumbed) -- queued last.
    - [x] **C = 2 / 4 done (14:13):** all 6 arms clean, peak = C, mean in-flight 1.99-2.00 / 3.76-3.79.
      | C | arm | out tok/s | TTFT mean / p50 / p99 | TPOT | ITL mean / p99 |
      |---|---|---|---|---|---|
      | 2 | MTP | 100.3 | 3846 / 341 / 96.5K | 10.9 | 10.0 / 27.6 |
      | 2 | split 3072 | 108.0 (+8%) | 4007 / 277 / 102.8K | 8.4 | 8.1 / 85.5 |
      | 2 | EDF+floor | 114.3 (+14%) | 3383 (-12%) / 273 / 94.3K | 8.5 (-22%) | 8.7 (-13%) / 87.6 |
      | 4 | MTP | 93.2 | 3466 / 422 / 96.1K | 30.1 | 31.8 / 590 |
      | 4 | split 3072 | 120.5 (+29%) | 3995 (+15%) / 330 / 72.2K | 21.4 (-29%) | 21.1 (-34%) / 194 (-67%) |
      | 4 | EDF+floor | 117.9 (+26%) | 3945 (+14%) / 349 / 70.0K | 20.8 (-31%) | 21.8 (-31%) / 160 (-73%) |
      At C = 4 the split clears throughput, ITL and P99 ITL; mean TTFT misses. **TTFT by uncached
      prompt length (C = 4):** short appends < 4K (185 of 240 rounds) MTP 926 ms -> split 1994 /
      EDF 2344; 4-32K 2435 -> 3618 / 2407; cold >= 32K (19 rounds) 30.1 s -> 24.2 / 22.4 s (-20 /
      -26%). The kernel helps the long prefills; the loss is head-of-line blocking: sglang continues
      the in-progress chunked prefill with the whole step budget, so a short append waits for every
      remaining chunk of a long prompt, and 2-3K chunks hold that slot longer than MTP's 8192.
      EDF's floor binds at every logged step (admitted 2048 of 8192); the step model's prediction is
      physical (8192 tokens at a 100K prefix ~ 670 ms of attention vs ~145 ms of budget at 50 ms x
      ~2.9 tokens/step), so this is the SLO, not a model error.
      (`ttft_breakdown.py` on the pod; per-arm `*_agent.json` under `concurrency_sweep_20261015/`.)
    - [x] **C = 8 done (15:00):** all arms clean, peak 8, mean in-flight 7.07-7.13.
      | arm | out tok/s | TTFT mean / p50 / p99 | TPOT | ITL mean / p99 | TTFT <4K / 4-32K / >=32K uncached |
      |---|---|---|---|---|---|
      | MTP | 93.7 | 6133 / 651 / 100.9K | 74.0 | 59.8 / 1335 | 2179 / 7166 / 42678 |
      | split 3072 | 124.1 (+32%) | 6580 (+7%) / 411 / 82.2K | 37.7 (-49%) | 40.1 (-33%) / 271 (-80%) | 4312 / 4536 / 32534 |
      | EDF+floor | 125.2 (+34%) | 7846 (+28%) / 483 / 72.1K | 36.6 (-50%) | 36.8 (-38%) / 236 (-82%) | 6157 / 3681 / 32180 |
      Throughput, TPOT, ITL and P99 ITL clear the bar; mean TTFT does not, for the same reason as
      C = 4 and more so: short appends (185/240 rounds) wait 2-2.8x longer behind long prompts'
      chunks, while 4-32K and cold prefills improve 37-49% and 24-25%. EDF's 2048 floor makes the
      blocking worse than static 3072. The reservation A/B below is the direct test.
    - [x] **Cohort C = 8 + ShareGPT 8 req/s (15:18):** all arms clean, peak 8 programs, mean
      7.29-7.67; acceptance 2.12-2.15 in every arm. Out tok/s / TTFT mean / TPOT / ITL mean / P99 ITL:
      | arm | agent | chat |
      |---|---|---|
      | MTP | 110.3 / 18992 / 29.6 / 20.7 / 413 | 1288 / 23560 / 30.0 / 30.4 / 476 |
      | split 3072 | +24% / -23% / -37% / -23% / -81% | +22% / -38% / -18% / -19% / -66% |
      | EDF+floor | +25% / -22% / -42% / -27% / -82% | +23% / **-75%** / -19% / -17% / -73% |
      Chat is still queueing (MTP chat TTFT 23.6 s at 8 req/s), so this point is past chat capacity.
      Misses: agent TTFT (-22%), agent ITL (-23..-27%), chat mean ITL / TPOT (-17..-19%).
    - [x] **Cohort C = 16 + ShareGPT 8 req/s (15:37):** all arms clean, peak 16, mean 13.6-14.9.
      | arm | agent | chat |
      |---|---|---|
      | MTP | 106.9 / 43548 / 44.9 / 27.4 / 501 | 1207 / 52023 / 31.5 / 31.4 / 471 |
      | split 3072 | +30% / -28% / -53% / -36% / -84% | +28% / -37% / -25% / -25% / -66% |
      | EDF+floor | +30% / -27% / -55% / -39% / -86% | +29% / -55% / -25% / -24% / -73% |
      Clears throughput, agent TPOT/ITL/P99 ITL, chat TTFT and P99 ITL; misses agent TTFT (-27%)
      and chat mean ITL (-24%). Both arms deeply overloaded (MTP TTFT 44-52 s).
    - [x] **Cohort C = 24 + ShareGPT 8 req/s (15:56):** all arms clean, peak 24, mean 19.2-21.0.
      | arm | agent | chat |
      |---|---|---|
      | MTP | 108.1 / 58964 / 87.6 / 42.6 / 597 | 1208 / 73577 / 29.3 / 28.8 / 406 |
      | split 3072 | +32% / -25% / -72% / -52% / -83% | +29% / -39% / -25% / -27% / -64% |
      | EDF+floor | +30% / -21% / -77% / -60% / -88% | +29% / -48% / -22% / -20% / -69% |
      **Sweep summary (all 18 arms clean, every peak = C):** throughput +22..+34% and P99 ITL
      -64..-88% everywhere above C = 2; agent TPOT/ITL clear the bar from C = 4 (TraceLab) and C = 16
      (cohort). Two metrics miss at every loaded point: **agent mean TTFT** (cohort -21..-28%;
      TraceLab +7..+28%, short appends blocked behind long prompts' chunks) and **chat mean ITL**
      (-17..-27%). The reservation and floor A/Bs below target exactly these.
    - [ ] **EDF earlier-deadline reservation (queued after the sweep):** under EDF, cap the chunked
      request's share so waiting requests with an earlier TTFT deadline are admitted in the same
      step (`edf_reserve_earlier_deadlines.patch`, tree `up_edf_resv`, `SGLANG_EDF_RESERVE=1`).
      A/B vs a same-queue EDF control at TraceLab C = 4 / 8 and cohort C = 16 + chat 8 req/s:
      `edf_reserve_ab_20261015.sh`; S3 now waits for it.
      **TraceLab C = 4 (16:26):** both arms clean. Reservation vs same-queue control: mean TTFT
      4633 -> 2882 ms (-38%); short appends <4K 2670 -> 917 ms (-66%, back to MTP's 926); cold
      >=32K 28.8 -> 23.5 s; 4-32K 1991 -> 2074; throughput 115.2 -> 118.2; mean ITL 20.8 -> 23.5 ms
      (+13%, short requests join the decode batch sooner). The control repeated 17% above the
      sweep's EDF arm in mean TTFT, so cross-queue comparisons are not trusted:
      `confirm_resv_20261015.sh` reruns MTP / split 3072 / EDF + reservation interleaved twice in one
      queue (TraceLab C = 4 / 8, cohort C = 16 + chat 8 req/s) before any verdict; S3 behind it.
    - [ ] **Reservation + lower floor (queued after it):** chat mean ITL misses (-24% at cohort C=16)
      because the 2048 floor binds every step; floor 1024 and 512 with the reservation on, TraceLab
      C = 8 and cohort C = 16 (`edf_floor_resv_20261015.sh`). S3 moved behind it.
  - [ ] **Final-config arms (2026-10-15, queued):** MTP | split static 3072 | split + EDF 50 ms +
    floor 2048, tree `up_edf_floor`, guard 0, no barrier (`s3_standalone_final_20261015.sh`, root
    `s3_standalone_final_20261015`; yields to `codex_needs_gpu`). Replaces the never-run
    2026-10-13 arms. Re-queued 13:1x behind `concurrency_sweep_20261015.sh`; load axis is request
    rate only (Mooncake: recorded per-request timestamps compressed, achieved req/s reported).

- [ ] **S4, EDF step model** re-fitted on the fused kernel and compared with the unfused schedule
  (goal point 3); earlier fits used the RoPE-corrupted model and are void (incl. the uncommitted
  2026-10-12 05:10 "fused-vs-unfused EDF check" in `server_perf_model.md`). The STEPTRACE emitter
  did not survive the re-port; ported to the overlap path with the 4-mode verify rows counted
  (`steptrace_port.patch`). `s4_edf_20261013.sh`, on the race-fixed tree without the barrier (a
  per-step sync would distort the modelled step times), runs only if the race fix passes: traces
  on both 4-mode arms (mix32 8/16/24/s, mix128 12/s), one linear model per arm (held-out MAPE),
  then EDF-R with each arm's own model vs no EDF at mix32 16/24/s -- queued after S3.

  - [x] **Guard-0 refit + EDF dynamic chunk, overloaded cohort (Claude, 2026-10-15; builds on the
    Codex kernel worker's EDF runtime / budget fix / mode-aware fitter):** step traces with the
    decode guard off (split and unfused, chunk 8192 and 2048; 5.4-6.6K rows each), fitted with
    `fit_step_matrix_piecewise` (piecewise-relative): held-out MAPE **split 10.1% vs unfused
    10.7%** (goal point 3: the split is slightly easier to predict; Codex's guard-on fit was
    13.9% vs 16.3%). Acceptance stays out of the step model because the traced value is the
    lifetime average (2.12-2.19, not identifiable); it enters the EDF budget instead
    (slack = ITL_SLO x per-request acceptance EWMA - time since the row's last token).
    Serving on the frozen 32-64K cohort, all arms in Codex's `up_edf_budgetfix` tree:
    | arm (2 runs where shown) | chat tput / TTFT / mean ITL / P99 | agent tput / TTFT / mean ITL |
    |---|---|---|
    | MTP | 928 / 179 s / 32.2 / 486 ms | 83 / 121 s / 93 ms |
    | split static 3072 (x2) | 1240 / 125 s / 22.0 / 168 ms | 116 / 95 s / 29 ms |
    | EDF itl 50 ms | 1074 / 149 s / 19.9 / 75 ms | 105 / 115 s / 18 ms |
    | EDF itl 30 ms (x2) | 849 / 211 s / 16.2 / 50 ms | 80 / 148 s / 13 ms |
    EDF meets its ITL target but admits only ~1.3K (50 ms) / ~300 (30 ms) prefill tokens per step:
    this cohort is past saturation in every arm (TTFT 40-200 s), every decode row is late, and a
    slack-sized chunk starves prefill -- at 30 ms TTFT is worse than stock MTP. Confirms
    `EDF: order, don't cap` on the current stack. **Caveat:** all absolute numbers in this tree
    are ~30% slower than the same frozen cohort in the chunk-sweep tree, MTP included (identical
    server args and requests; GPU unthrottled; scheduler host-bound at ~100% CPU) -- compare only
    within a run. ([script](debug_kernel_server_perf/resume_20261012/edf_dynchunk_20261015.sh))
    **Correction (2026-10-15 12:40):** the EDF trees (`up_edf_budgetfix`, `up_edf_floor`) lacked the
    program-concurrency replay (`BENCH_SESSION_CONCURRENCY` silently ignored; no `PROGRAM_CONCURRENCY`
    line in any bench log), so every program was released at once: measured mean in-flight
    31.4-34.8 in every arm, not the stated 24 / 8 / 16 programs. Arms within a run stay matched, but
    the load labels are wrong and "overloaded at any program concurrency" was never tested.
    Rerun with the replay ported (`edf_floor_session_concurrency.patch`): `concurrency_sweep_20261015.sh`.
  - [x] **EDF with a prefill floor (2026-10-15 07:20-08:01):** `SGLANG_EDF_PREFILL_FLOOR`
    (`edf_prefill_floor.patch`, tree `up_edf_floor`), MTP / static 3072 / EDF 50 ms + floor 2048,
    all arms clean. Both lighter points turned out still past saturation (chat TTFT 95-158 s): the
    frozen cohort's ~4.7M agent prefill tokens keep it overloaded at any program concurrency. The
    floor binds (admitted prefill = 2048 every logged step), so EDF acts as chunk 2048 + deadline
    order. EDF+floor vs static 3072: equal-or-better on everything except agent TTFT -- 8p/4/s: chat
    TTFT -10%, chat ITL -10%, chat P99 ITL -27%, agent ITL -16% (tput, agent TTFT within noise);
    16p/6/s: chat TTFT -6%, P99 ITL -24%, agent ITL -21%, agent TTFT +5%. Vs MTP: 8p chat
    +38% / -35% / -42% (tput/TTFT/ITL), agent +41% / **-16%** / -74%; 16p chat +34% / -30% / -34%,
    agent +41% / **-15%** / -78%. Agent TTFT stays the gap (queueing behind its own prefill
    backlog). Next: across capacity on the S1 mixture. **Retracted (2026-10-15 12:40):** "8p" and
    "16p" ran with all 48 programs released (mean in-flight 33.1-35.5), see the correction above.
  - [ ] **EDF + floor across capacity (2026-10-15 08:01-09:08) -- load axis retracted 12:40:** the
    agent tenant was driven by a session-start rate (0.1 sessions/s), not program concurrency, so
    the agent load is uncontrolled; these numbers are not evidence for the bar. Superseded by
    `concurrency_sweep_20261015.sh` (cohort C = 8/16/24 + chat 8 req/s). Original record: S1 mixture mix32 (32-64K agent sessions at 0.1/s) + short chat at 8/12/16/s,
    tree `up_edf_floor`, guard 0, split step model refit on guard-0 traces; all 9 arms clean
    (0 tracebacks / fallbacks; split calls 71.5K-130.5K; EDF admitted 2.4K-3.1K prefill tokens per
    step -- above the floor once slack exists). SLO chat 2 s / 50 ms, agent 10 s / 80 ms.
    | chat rate | arm | chat tput / TTFT / TPOT / P99 ITL / SLO | agent tput / TTFT / TPOT |
    |---|---|---|---|
    | 8 | MTP | 586 / 242 ms / 11.2 / 92.8 / 0.950 | 352 / 1220 ms / 9.1 |
    | 8 | static 3072 | 586 / 151 ms / 8.0 / 54.9 / 0.974 | 405 / 998 ms / 6.7 |
    | 8 | EDF+floor | 586 / 141 ms / 8.1 / 55.2 / 0.984 | 400 / 1087 ms / 6.8 |
    | 12 | MTP | 851 / 995 ms / 24.9 / 137 / 0.755 | 238 / 1782 ms / 16.5 |
    | 12 | static 3072 | 851 / 172 ms / 9.4 / 64.8 / 0.970 | 373 / 1004 ms / 7.7 |
    | 12 | EDF+floor | 852 / 146 ms / 9.3 / 60.7 / 0.988 | 374 / 1167 ms / 7.7 |
    | 16 | MTP | 1017 / 25.7 s / 39.5 / 140 / 0.119 | 161 / 14.9 s / 22.6 |
    | 16 | static 3072 | 1136 / 419 ms / 12.7 / 97.0 / 0.909 | 324 / 1305 ms / 10.2 |
    | 16 | EDF+floor | 1136 / 234 ms / 11.7 / 76.3 / 0.975 | 334 / 1518 ms / 9.4 |
    **EDF+floor vs MTP:** 12/s chat TTFT -85%, TPOT -63%, P99 ITL -56%; agent tput +57%, TTFT
    -35%, TPOT -54% (total output +13%). 16/s chat tput +12%, TTFT -99%, TPOT -71%; agent tput
    +108%, TTFT -90%, TPOT -59%. **Every metric of both tenants clears >=30% / >=10% at 12 and
    16/s** (chat throughput is capped by offered load below 16/s). Below capacity (8/s) all arms are
    fast and the agent gain is smaller (TTFT -11%). Vs static chunk 3072, EDF+floor raises chat SLO
    attainment (0.984 / 0.988 / 0.975 vs 0.974 / 0.970 / 0.909) and cuts chat TTFT -6..-44% by
    serving tighter deadlines first, at +9..+16% agent TTFT. Caveat: one run per arm; same-tree
    MTP/static controls in this run. Data [`edf_ratesweep_20261015.csv`](debug_kernel_server_perf/resume_20261012/edf_ratesweep_20261015.csv),
    command `bash edf_ratesweep_20261015.sh` (pod `/dev/shm/w`), analysis
    `drivers/serving/eval_curves.py --root edf_ratesweep_20261015 --slo '{"chat":[2000,50],"codex":[10000,80]}'`.
  - [ ] **Repeats + mix128 (sweep 2, same retracted session-rate axis; 09:11-10:17; `edf_ratesweep2_20261015.sh`, all arms clean):**
    mix32 repeats reproduce sweep 1 within ~3-7% for EDF and static, MTP within ~1% at 16/s and ~11%
    at 12/s. Two-run means, EDF+floor vs MTP: **12/s** chat TTFT -84%, TPOT -62%; agent tput +55%,
    TPOT -53%, TTFT **-30.9%** (runs -35% / -27%: the borderline metric); **16/s** chat tput +12%,
    TTFT -99%, TPOT -71%; agent tput +109%, TTFT -89%, TPOT -59%. **mix128 12/s** (overloaded:
    MTP chat TTFT 89 s): chat tput +47%, TTFT -81%, TPOT -55%, P99 ITL -84%; agent tput +43%,
    TPOT -50%, TTFT only **-18%**. **mix128 16/s:** chat tput +59%, TTFT -83%, TPOT -56%, P99 ITL
    -83%; agent tput +47%, TTFT **-33%**, TPOT -41% -- both tenants clear the bar (vs static 3072:
    chat tput +12%, TTFT -49%, TPOT -21%). Data
    [`edf_ratesweep2_20261015.csv`](debug_kernel_server_perf/resume_20261012/edf_ratesweep2_20261015.csv).
    Agent TTFT is the recurring gap: EDF's TTFT deadline grows with
    prompt length (+1 s per 5K tokens past 4096), so a 200K agent prompt gets ~40 s vs chat's 2 s
    and always waits, while the evaluation scores agents against 10 s.
  - [x] **Two-tier TTFT SLO in EDF -- not promoted (2026-10-15 10:39-11:20; same session-rate axis, the
    rejection stands as a within-run comparison):** prompts >= 16K get
    the flat agent SLO (10 s), shorter ones the chat SLO (2 s) (`edf_two_tier_slo.patch`, opt-in
    `SGLANG_EDF_LONG_PROMPT_TOKENS`), EDF+floor with vs without, same run, all arms clean.
    mix32 12/s: no effect (32-64K prompts already had ~8-14 s deadlines; agent TTFT is its own
    prefill time). mix128 12/s: agent TTFT 26.0 -> 21.3 s (-18%, P99 -37%) but chat TTFT
    16.7 -> 22.5 s (+35%) -- it moves TTFT between tenants. mix128 16/s: worse for both (chat
    tput -8%, chat TTFT +73%, agent TTFT +11%). Standard length-scaled deadlines stay. This run's
    standard-deadline control at mix128 16/s reproduces sweep 2 with margin: agent TTFT 26.1 s vs
    MTP 44.2 s (-41%). Data [`edf_tier_20261015.csv`](debug_kernel_server_perf/resume_20261012/edf_tier_20261015.csv),
    `edf_tier_20261015.sh`.
- [x] **Chunk size: not the lever with the decode guard on (2026-10-12); IS the ITL lever with guard 0 (2026-10-15, supersedes):** chunk 8192/4096/2048 leaves
  both arms unchanged (prefill per step is many ~512-token prompts).
  - **2026-10-15 update (guard 0, split serving kernel):** with long prefills now routed to FA3
    (Codex found the decode-population guard kept every 32K+ prefill off FA3 in timed runs), the
    mixed-step chunk sets step time and hence ITL. Frozen 32-64K cohort (24 agent programs +
    natural ShareGPT 8/s), all arms audited; noise floor from the 8192 repeat: 2-3%.
    | vs MTP (8192) | split 8192 (mean of 2) | split 4096 | split 2048 |
    |---|---|---|---|
    | chat tput / TTFT / mean ITL | +31% / -43% / -26% | +32% / -43% / -28% | **+34% / -43% / -34%** |
    | agent tput / TTFT / mean ITL | +34% / -31% / -33% | +34% / -30% / -42% | **+38% / -28% / -56%** |
    MTP itself does not gain from small chunks (2048: chat ITL 29.40 vs 29.15 ms, tput -8%, TTFT
    +20%), so 8192 is its best setting and the comparison is fair. At chunk 2048 every metric for
    both tenants clears the bar except **agent TTFT (-27.6%)**; at 4096 agent TTFT clears (-30.2%)
    but chat ITL does not (-28.3%).
    **With repeats (sweep 2b, two runs per arm; MTP repeats within 1%):** 2048 gives chat
    +32% / -40% / -32% and agent +35% / **-25%** / -57%; 3072 gives chat +31% / -41% / **-29.1%**
    and agent +33% / **-27.4%** / -49%. No static chunk clears both chat ITL and agent TTFT: a
    smaller chunk protects decode ITL but delays agent prefills. Next: EDF dynamic chunk (prefill
    sized to the tightest decode slack each step), `edf_dynchunk_20261015.sh`, running.
    ([table](debug_kernel_server_perf/resume_20261012/chunk_itl_sweep_20261015.txt),
    `chunk_itl_sweep_20261015.sh`)
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
- **Crossflow: Prefill–Decode Elasticity for Agentic LLM Serving** -- Y. Xu et al.,
  [arXiv:2609.27085v1](https://arxiv.org/abs/2609.27085v1) *(sent)* -- note: direct cluster-policy
  baseline. Fixed P/D roles with decode-local prefill admitted through revocable multi-resource
  leases, cache/shape-aware path comparison, and local SLO revalidation. No new model kernels.
  **How it replays TraceLab** ([Sec. 6](https://arxiv.org/html/2609.27085v1#S6)): one-hour
  segments of the public trace (window not stated), keeping timestamps, conversation order,
  prefix relationships and token work; load factor alpha divides every timestamp offset from the
  first arrival by alpha (8x-20x for GPT-OSS-120B; baselines at 16/18/20x): the recorded
  request timestamps compressed, so **its load axis is request rate (req/s)**. Whether a round
  waits for the previous response is not stated. Six TP4 GB300 servers (2P4D), SGLang, <=256
  running requests per server, no stated evaluation SLO. Its GLM-5.2 TraceLab mean ITL worsens
  20.9-34.3%. Load axes in the literature are request rate (Crossflow) or program concurrency
  (ThunderAgent); neither uses session starts/s. Our agent tenant uses concurrency (below).
  This does not close A/B or justify moving to EDF. [Analysis and comparison](debug_kernel_server_perf/resume_20261012/crossflow_analysis_20261014.md).
- **ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System** -- H. Kang et al.,
  [arXiv:2602.13692](https://arxiv.org/abs/2602.13692) *(sent)* -- note: argument and setting. KV-cache
  thrashing in agentic serving (a session's cache evicted during a tool call and re-prefilled, up
  to 7.14x end-to-end): motivates PD point 2 (migration leaves replicas where demand is) and
  evaluating on multi-turn agent sessions rather than independent prompts. **Load protocol we
  follow:** agent load is the number of parallel programs (workflows) kept active -- its x-axis
  "parallel workflow number", headline point 96 programs on 8xH100 -- not an arrival rate. Every
  agent arm here reports its program concurrency C and the measured mean requests in flight;
  session-start rates (sessions/s) are not used.
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
