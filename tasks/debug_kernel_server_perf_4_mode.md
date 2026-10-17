# Current goal

Fused FA3+FA2 SM pre-assignment attention for a 4-mode MTP (EAGLE3) serving schedule, optimised
on real serving workloads, then the EDF model and the PD-colocation argument
(`tasks/pd_colocation_vs_disagg.md`). Bar: >=30% TTFT and ITL, >=10% throughput on realistic
chat + TraceLab workloads. Single H200 trial `095f01a3e1585eea` (`vibecuda`); H20 multi-node only
after the kernel items close (<=16 GPUs).

**Where it stands (2026-10-18):**
- [x] Kernel items A, B and A5 closed (2026-10-17). The FA3 role runs flash-attention's kernel (paged TMA, `-DNDEBUG`,
  fast math, dynamic scheduler) at parity with sgl-kernel FA3; FA3 || FA2 co-running buys -18% on balanced steps but
  is bounded at ~1.2% of serving GPU time on TraceLab. Multicast, q-tile choice, split reduction and in-kernel
  backfill are closed by measurement or measured bound (A5).
- [x] Best serving config (single H200): PD multiplexing + worker-side EDF with uncached-token TTFT deadlines, decode
  divisor 30 -- vs stock MTP all 12 bar metrics clear in the mean (C=8 four repeats, C=16 three), output gate PASS.
  (Direct baselines -> PD multiplexing + EDF.)
- [x] S4 step model on that schedule: decode ITL predictable to ~9% held-out vs 35-117% on the unfused schedules.
- [x] 8 H20s (`pd_colocation_vs_disagg.md`): the same workers plus per-step SM sharing (TTFT deadlines + 24 ms ITL
  budget) clear all 12 vs stock colocation + MTP in both repeats (smallest -39%), H20 output gate PASS.
- [ ] **4-mode MTP schedule on H200 (2026-10-18, re-forked trial `d6a80befdd4e4da5`, provisioned from the H20 recipe:
  upstream 3fef129 + `h20_overlay.tgz` + port07 + uncached deadlines; harness `mn_h200_fourmode_20261018.sh`, one worker,
  agent C=8 + ShareGPT 8 req/s, 48 agent requests; all runs 0 tracebacks / 0 fallbacks).** vs stock MTP (same harness,
  same session): chunk 2048 agent +42% / -24% / -34%, chat TTFT -68% / ITL -23%; chunk 1024 agent +41% / -17% / -50%, chat
  -64% / -29.7%; chunk 512 agent -8% / +48% / -51%. Chat throughput = offered load in every arm. Misses at 1024: agent TTFT
  and chat ITL (0.3 point short). **Deadline-driven chunk (small 1024, large 4096, only late requests longer than one
  small chunk count; `SKIP_BASE=1 CHUNKS=4096 DEADLINE_CHUNK=1024:4096:<tok/s> bash mn_h200_fourmode_20261018.sh`, both
  clean):** pace 12000 agent +46% / -28.5% / -31.0%, chat TTFT -70.8% / ITL -21.2%; pace 6000 agent +47.7% / -29.9% /
  -31.0%, chat TTFT -71.7% / ITL -22.4%. **The 4-mode MTP schedule on H200 reaches the bar on agent throughput, TTFT
  (-29.9%) and ITL and on chat TTFT; chat ITL (-22%) is the remaining miss** (single repeats; chat throughput is the
  offered load). Repeat 2 of pace 6000: agent +48.0% / -28.9% / -33.3%, chat -72.3% / -23.0%. **Large chunk 2048 instead of
  4096 (`DEADLINE_CHUNK=1024:2048:6000`, clean): agent +51.8% / -29.7% / -40.7%, chat TTFT -73.7% / ITL -26.7%** --
  better on every metric; remaining gaps agent TTFT (0.3 point) and chat ITL (3.3 points). **Chunk pairs 768/2048 and 1024/1536 (clean):** agent +50% / -29.1% / -41.9%, chat -73.1% / -26.1%; agent +44% /
  -25.4% / -36.0%, chat -69.0% / -26.1%. Across all five chunk settings chat ITL stays at 12.1-12.7 ms (-23..-27% vs
  stock MTP's 16.5 ms) whatever the mixed-step length, so the remaining chat-ITL gap is the MTP decode step itself
  (3 draft steps + a 4-token verify at ~2.1 accepted tokens), not prefill interference; -30% needs <= 11.55 ms. Best
  setting 1024/2048: agent +52% / -29.7% / -40.7%, chat TTFT -73.7% / ITL -26.7%. Composition counter (worker log,
  `FUSED_SPLIT_CALLS`, logged every 500): >= 66001 split attention calls (~2.36K 4-mode steps x 28 layers) in the
  1024/2048 arm and in its 2-step / 3-token repeat; 0 in stock MTP; fallbacks 0 everywhere.
  - [x] 2026-10-18 matched repeat of the best setting (stock MTP + ours 1024/2048, same session; clean): rep 2 agent
    +42.4% / -28.9% / -32.0%, chat TTFT -71.8% / ITL -22.8% (rep 1: +51.8% / -29.7% / -40.7%, chat -73.7% / -26.7%).
    Two-rep means: agent 105.9 vs 72.0 tok/s (+47%), TTFT 12.14 vs 17.18 s (-29.3%), ITL 16.5 vs 25.9 ms (-36%); chat
    TTFT 2.67 vs 9.79 s (-72.7%), ITL 12.3 vs 16.4 ms (-25%); chat throughput = offered load. Per-request tok/s/user
    (p1 / p10 / median / p90 / p99), rep 1 | rep 2 -- agent stock 5.2/21.4/49.8/84.8/97.9 | 5.2/17.2/49.8/81.7/94.3, ours
    25.9/37.5/57.0/113.5/143.5 | 21.0/33.4/53.6/98.2/132.3; chat stock 11.5/27.5/81.8/253.0/335.8 |
    11.7/27.3/95.3/253.3/336.4, ours 20.3/33.2/184.3/264.2/354.2 | 19.6/32.8/179.9/262.2/344.6. The tail moves most (agent
    p1 4-5x). Figure `tasks/pd_colocation/figs/h200_fourmode_best_reps_20261018.png` (+ `.csv`). Agent TTFT (-29%) and
    chat ITL (-23..-27%) stay below -30% in both reps; the rest clear. sglang `76cfe56b8`; `SPEC_TAG=_rep2 CHUNKS=2048
    DEADLINE_CHUNK=1024:2048:6000 bash tasks/pd_colocation/mn_h200_fourmode_20261018.sh`; figure `SID=d6a80befdd4e4da5
    MN_CP=i18n-bd python3 tasks/pd_colocation/mn_ab_figure.py figs/h200_fourmode_best_reps_20261018
    stock_mtp=coloc_sgl_mtps_h200_c8,ours_4mode=coloc_cache_edf_mtp_h200_4mode_unc_ch2048_dl1024-2048-6000_c8` (stock rep 1
    is directory `..._c8b`, linked as `_c8_rep1`; `_c8` is the aborted metrics-port attempt).
  - [x] 2026-10-18 where chat ITL goes, and a step-time budget on the chunk (negative). Chat ITL is the mean step
    interval; split by interval (rep 2, share of chat ITL time): stock < 10 ms 21% / 20-80 ms 31% / > 80 ms 34% (mean
    237 ms); ours 24% / 46% / 16% (mean 110 ms -- 2048 chunks over 32-64K prefixes). `apply_edf_step_budget.py`
    (`STEP_BUDGET=budget_ms:a:b`: largest chunk with a*c + b*c*(prefix + c/2) <= budget, a=0.02, b=5e-7) tests whether
    shortening deep-context mixed steps buys chat ITL. vs the two stock reps' means: budget 50 ms on 1024/2048 (caps
    verified depth-dependent: 2048 -> 1101 at a 50K prefix, -> 1902 at 12K; 782 capped steps; >= 71001 split calls; 0
    tracebacks / fallbacks) agent +37.9% / -21.8% / -34.6%, chat -65.7% / -23.5% -- worse TTFT, same chat ITL (12.5 vs
    12.3 ms). The first sweep (budgets 60 / 80 on 1024/4096) read `prefix_len`, which is not refreshed between chunks,
    so it capped by token count only (~3000 / ~4000), not by depth -- not a test of the rule (agent +40% / -25.1% /
    -34.6%, chat -68.3% / -22.3%; +42.4% / -26.2% / -32.3%, -69.4% / -21.1%). Conclusion: at a fixed prefill total,
    re-slicing it does not move the mean step interval chat sees; 1024/2048 stays best. sglang `708cd9dfe` + this
    commit; `SKIP_BASE=1 CHUNKS=2048 DEADLINE_CHUNK=1024:2048:6000 STEP_BUDGET=50:0.02:5e-7 REP_SUFFIX=_v2 bash
    tasks/pd_colocation/mn_h200_fourmode_20261018.sh` (patch applied on the pod tree, removed after).
  - [x] 2026-10-18 traced A/B (STEP_TRACE=1, both arms; clean, >= split calls as above): rep 3 of the best setting vs its
    own stock MTP: agent +45.5% / -29.0% / -35.7%, chat TTFT -71.5% / ITL -23.8% (three reps: agent TTFT -29.7 / -28.9 /
    -29.0%, chat ITL -26.7 / -22.8 / -23.8%). (a) Host overhead is not the remaining lever: 4-mode mixed steps cannot use
    the prefill CUDA graph (`can_run_graph` rejects MIXED) and run eager, but the GPU sits idle only 3.2 ms after each one
    (2742 steps, 8.8 s) and 0.16 ms after each decode step (3.0 s), against 297 s of GPU time -- <= 4% even if all of it
    were recovered. Per kind, ours: decode 9.14 ms x 18808, mixed 44.0 ms x 2742 (743 prefill tokens mean); stock
    MTP: decode 10.94 ms x 15448, mixed 59.0 ms x 2151. (b) EDF step model (S4; `fit_pdmux_step.py`, the stock 5-term
    form, fit on the first half, tested on the second): decode-only GPU time MAPE 3.6% ours / 2.7% stock; mixed steps
    26.7% (p90 38.0%) ours vs 53.5% (p90 94.1%) stock -- the 4-mode step is twice as predictable where EDF needs it.
    `STEP_TRACE=1 SPEC_TAG=_trace CHUNKS=2048 DEADLINE_CHUNK=1024:2048:6000 bash
    tasks/pd_colocation/mn_h200_fourmode_20261018.sh`; `tasks/perf_model/steptrace_util.py` and
    `tasks/perf_model/fit_pdmux_step.py` on the two `worker_r0_g0.log`. sglang `f04f12557` + this commit.
  - [x] 2026-10-18 output gate on this pod, PASS (`tasks/pd_colocation/gate_h200_fourmode_20261018.sh`, srv_correctness
    vs stock MTP; driver now takes SRV_ARGS and counts mixed composition records). Noise floor MTP_REP: long prompts agree
    >= 18 tokens, short >= 63; null case MTP_C2048 (stock MTP at the config's chunk 2048): long >= 11, short 0 (s1's
    first decode token flips on chunking alone); config MTP_MIX_SPLIT_EDF (mtpsplit 112 FA3 SMs + EDF + chunk preemption
    + uncached deadlines + deadline chunk 1024/2048:6000): long >= 11, short 0 -- the same requests diverge at the same
    positions as the null case (s0 132, s2 298, s5 63, l1/l5 11), so the remaining divergence is the chunk size, not the
    split or the schedule. Config arm: >= 1501 split calls, 0 tracebacks, 0 fallbacks, priority scheduling on.
    sglang `3717b3638` + this commit; `bash /dev/shm/w/mn/gate_h200_fourmode_20261018.sh` on pod `d6a80befdd4e4da5`.
  - [x] 2026-10-18 shorter MTP step (2 draft steps, 3 verify tokens, both arms; clean: tracebacks 0, fallbacks 0): stock MTP
    agent 68.6 tok/s / 17.49 s / 27.9 ms, chat 10.19 s / 16.8 ms; ours (1024/2048) agent 99.6 / 12.25 s / 19.4 ms, chat
    2.94 s / 12.9 ms -> agent +45.2% / -30.0% / -30.5%, chat TTFT -71.1% / ITL -23.2%. Stock MTP's chat ITL does not
    move with the shorter step (16.5 -> 16.8 ms) and neither does ours (12.1-12.7 -> 12.9 ms), so the draft depth is not
    what holds chat ITL at -23..-27%; agent ITL gain shrinks (-40.7% -> -30.5%). The 3-step / 4-token setting stays best.
    sglang `dd200bb1d`; `SPEC_STEPS=2 SPEC_TOKENS=3 SPEC_TAG=_s2t3 CHUNKS=2048 DEADLINE_CHUNK=1024:2048:6000 bash
    tasks/pd_colocation/mn_h200_fourmode_20261018.sh` (log `/tmp/h20/mn_h200_s2t3.log`).
- [ ] Not met: vs stock colocation *without* speculation on H20 (agent TTFT at a ~5.9 s compute floor; best -18% /
  -21%), and node-count scaling (needs > 16 GPUs). Both wait on the user: which baseline the bar uses, and the GPU cap.

**Earlier standing (2026-10-13, superseded):** the serving kernel is now the two-launch split
(FA3 prefill rows + stock FA2 BatchPrefill verify rows): vs stock MTP it clears the bar at mix32
16/24 and mix128 24/s for both tenants (e.g. mix128 24/s chat +56% tput / -55% TTFT / -45% TPOT)
and misses only agent TTFT/TPOT at mix128 16/s (-26%). Earlier summary, fused co-launch: under load the fused 4-mode arm beats stock MTP by TTFT
-31..-79%, throughput +11..+36%, per-token latency -19..-30%; below capacity the arms are equal.
TTFT and throughput clear the bar wherever the server is loaded; per-token latency clears it only
at some points. The graphs-on 4-mode crash is fixed (a schedule-stream publish raced the forward stream); 4-mode arms now run without the mixed-step barrier.

## Tasks, in priority order

A and B are the two kernel items; S is the server/workload evaluation the user specified.
Supporting gates are prerequisites, not substitutes. `[x]` = measured, `[~]` = partly, `[ ]` = not run.

### [x] A. Ultra fine-grained tile dispatcher

- [x] **A-fix, FA2 out of the co-compiled kernel (2026-10-12):** FA2 spilled inside the fused
  kernel (168 regs, 4200 B stack, 21x slower per item than `BatchAttention`); now FA3 grid on
  `sm_split` CTAs + `FA2ExecutorKernel` at native geometry as its PDL dependent
  (`FLASHINFER_FUSED_COLAUNCH=1`). 3.3-6.5x on FA2 rows, every gate exact; still 1.2-2x behind the
  non-persistent unfused two-launch control. ([evidence](debug_kernel_server_perf/resume_20261012/colaunch_20261012.md))
- [x] **A1, FA3 q64 vs q128 per work item (2026-10-12; reopened and re-closed 2026-10-17, see A5.1):** *Correction in place:* the
  old line "q64 never beats q128 beyond noise, loses up to 2.3x" misread the winner map. On the fixed
  kernel (`a3_winner_map/a3_v2_table_20261012.txt`, 72 cells, 216/216 arms exact) q64 is slower on 64
  cells (median 1.34x, worst 2.98x at kv 64K x 16 rows x qo 16: half the query rows per loaded KV block,
  so twice the KV traffic on memory-bound items) and faster by >3% on 3 cells, up to 24% (kv 1024, 1 row,
  qo 16 -- short-KV, few-row items). A per-item q-tile choice therefore has a real, narrow win region;
  reopened as part of the FA3 technique study (S-kernel, below), to be measured on serving-shaped items.
- [x] **A2, FA2 tile classes (2026-10-12):** 48/32/16 refuted (tile 48 = 1 q-warp x 4 kv-warps,
  3x accumulator, spills; 1.6-2.2x slower at q8); **64/32/16** adopted (96 B stack). Per-item tail
  dispatch kept (bit-identical outputs). FA2 split-KV chunk = `auto` (sized to the FA2 CTA slots;
  matches or beats the best fixed chunk on every fixture). ([evidence](debug_kernel_server_perf/resume_20261012/colaunch_20261012.md))
- [x] **A3, fitted FA2/FA3 dispatch rule (2026-10-12; closed 2026-10-16, see the guard-0 A/B below):** winner map on the fixed kernel, 216/216
  exact: FA2 wins every verify-width row; rule "FA3 iff packed_qo >= 112 and FA3 work >= 3.5M"
  agrees on 69/72 cells (worst loss 6%) vs 55/72 (98%) for the serving rule -- **but loses in
  serving** (-3.5% throughput, +29% TTFT). Cause, from the split counters (2026-10-13): the work
  floor was fitted on band rows only (qo <= 32) but also rejected 512-token chat prefill chunks
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
  - [x] **A3 at guard 0, clean A/B (2026-10-16 00:19-01:13, all arms clean, peak = C):** the A3b test above compared A3B against
    DEFAULT with the decode guard ON, and A3B drops the guard, so its gains are not attributable to
    the fitted thresholds; guard 0 is now the default. Both rules at guard 0 on the split (chunk
    3072, no EDF): q4 gate per rule, interleaved DEFAULT/A3B x2 on the cohort at C=16 + chat 8 req/s,
    one pair on TraceLab C=8 (`a3_guard0_20261015.sh`). Both rules pass the q4 gate (long prefix 52
    vs the MTP_LPT_Q4G floor). **A3B = DEFAULT within noise:** cohort C=16, two-repeat means, A3B vs
    DEFAULT: agent tput +0.3%, TTFT -0.5%, TPOT -0.3%, ITL -3.5% (repeat spread 1.4%), P99 ITL -12.6%
    (spread 6.3%); chat tput -0.1%, TTFT -2.2%, TPOT -1.8%, ITL -0.8%; acceptance 2.13 in all four.
    TraceLab C=8: tput 0.0%, mean TTFT -3.8% (single runs vary up to 17%), median -0.5%, TPOT +1.3%,
    ITL +2.1%, P99 ITL +9.6%. The 2026-10-13 A3b gains (TTFT -25..-28%) came from dropping the decode
    guard, now the default, not from the fitted thresholds. Split calls 48.0-51.5K (cohort),
    35.5K (TraceLab) in both rules. **A3 closed:** keep the default rule at guard 0.
    Command: `bash a3_guard0_20261015.sh` (pod `/dev/shm/w`), table `a3tab.py a3_guard0_20261015`.

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
  - [x] **Split v2, first point (2026-10-13 07:14; superseded by "Split v2 complete" below):**
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

- [x] **A5, FA3 technique study on serving-shaped items (user, 2026-10-17: "test these techniques more for fa3 ...
  otherwise it's hard to make a story around the kernel stuff if we just port it without any optimizations").**
  **Outcome (2026-10-17):** the custom FA3 role now runs flash-attention's own kernel (paged TMA, `-DNDEBUG`, fast
  math, dynamic scheduler) and matches sgl-kernel FA3 instruction for instruction; every earlier FA3 gap was the build
  (flashinfer's mainloop, asserts compiled in, cp.async gather). Of the techniques, only SM co-running FA3 with FA2
  buys anything (-18% on balanced append + verify steps), and its serving ceiling on TraceLab is 1.2% of GPU time
  because long-context prefill dominates every mixed step. The attention kernel is at the state of the art and is
  not where the serving bar can be won. (The co-launched FA2 executor's A4 gap was also measured without
  `-DNDEBUG`; moot while serving runs stock FA2.)
  Fixtures = FA3 items replayed from serving step traces (TraceLab append 256-1024 on 32-131K, chat prefill
  chunks, split-KV long rows), every arm PASS vs the causal fp32 reference, reported against upstream FA3
  (what serving runs) and the unfused two-launch step. In order:
  - [x] A5.0 mainloop gap (closed 2026-10-17): it was the build, not the mainloop design -- flashinfer's paged mainloop
    instead of flash-attention's, no `-DNDEBUG`, and cp.async page gather instead of paged TMA.
    - [x] **Parity (2026-10-17 13:53-13:58, all runs 11-12/11-12 vs fp32 causal):** sgl-kernel FA3 with
      `pack_gqa=True` forced is no slower (prefill 2048 @ 64K 3.24 vs 3.23 ms), so packed GQA is not the gap. Paged
      TMA KV (`FA3_PAGED_TMA=1`) with `-DNDEBUG` -- earlier measured slower only because the device asserts were in --
      makes our flash-attention build on its dynamic scheduler **match or beat sgl-kernel FA3**: prefill 2048 @ 64K
      3.19 vs 3.27, agent 4 x 512 @ 64K 3.32 vs 3.35, chat 2048 0.0606 vs 0.0601 ms. Final FA3 role build:
      flash-attention mainloop, paged TMA, `-DNDEBUG`, fast math. (Our static-list path at 112: 3.44 / 3.58 /
      0.117 ms -- the dynamic scheduler stays.) Mixed co-run table rerunning on this build
      (`a5_mixed_corun_tma_20261017.sh`).
    - [x] **A5.0f full table + scheduler (2026-10-17 13:26-13:50; every run 11/11 vs fp32 causal):** flash-attention
      mainloop + `-DNDEBUG` + fast math (`a5_fa3_upstream_table_20261017.sh`, two reversed repeats, spread <=0.6%):
      long items within 1.07-1.12x of sgl-kernel FA3 (append 512 @ 48K 0.627 vs 0.586, agent 4 x 512 @ 64K 3.74 vs
      3.34, prefill 2048 @ 64K 3.61 vs 3.31 ms); flashinfer's mainloop with `-DNDEBUG` gains only ~4%, so the
      flash-attention mainloop is the base. **Split-KV fix:** the two split fixtures were nonfinite --
      `fused_primary_finish` re-merged the plan's split rows from flashinfer's (unwritten) partial buffer after the
      dispatch had already run flash-attention's combine; patch item 2 in `fi_a5_patch.py`, now exact (append @ 132
      splits=2, 16 x 16 @ 1K). **Scheduler** (`a5_fa3_budget_20261017.sh`): our build on upstream's own dynamic
      persistent scheduler capped at 112 SMs (`fa3_only_run`, `FA3_SM_BUDGET=112`) vs our static per-CTA lists at
      112, ms: 8 x 1024 @ 32K 7.42 vs 9.05; chat 2048 0.074 vs 0.121; chat 4 x 512 0.040 vs 0.046; 16 x 16 @ 1K
      0.023 vs 0.041; long items tie (3.60 vs 3.63); static wins only append 512 @ 48K (0.648 vs 0.703, one tile per
      CTA). Dynamic at 112 vs sgl-kernel at 132: 1.10-1.24x. *Correction (2026-10-17 13:52, from A5.3 below): I read
      this as the SM-count ratio ("parity at equal SMs") -- wrong for the long items: prefill 2048 @ 64K is 448
      equal tiles, 4 waves at both 112 and 132, and sgl-kernel FA3 capped at 112 SMs is no slower; our 3.60 ms there
      is a real 1.10x per-SM gap.* So the FA3 role should run flash-attention's kernel on its dynamic scheduler over the
      FA3 SM budget; the remaining question is what the other SMs buy when FA2 runs on them concurrently (A5.3),
      which was last measured on the flashinfer mainloop without `-DNDEBUG`.
    - [x] **A5.0d FA3 role on flash-attention's mainloop (2026-10-17 12:55-13:16, every run 11/11 checks vs fp32
      causal reference):** `FA3_UPSTREAM=1` now builds (loader switch, hopper headers beside the merged include
      tree) and runs after one fix -- the launch sized dynamic smem from flashinfer's FA3 traits (163 920 B) while
      the body is flash-attention's kernel (compute-sanitizer: out-of-range shared address); `fi_a5_patch.py` takes
      the max with `UpstreamFA3Kernel::SharedStorageSize`. Upstream mainloop removed the LOP3/PRMT overhead but
      was no faster (prefill 2048 @ 64K 4.71 ms): consumers spin on mbarriers (NANOSLEEP 10M, SYNCS 31M). Paged-TMA
      KV (`FA3_PAGED_TMA=1`) is slower still (5.22 ms). Control arm `fa3only` (our build of flash-attention's
      kernel on upstream's own scheduler, 132 SMs): 4.65 ms = our candidate, so the scheduler adapter was not the
      long-item cost -- the build was.
    - [x] **A5.0e `-DNDEBUG` is the long-item gap (2026-10-17 13:21-13:25, 11/11 checks):** our FA3 extension was
      built without `-DNDEBUG`, so every device `assert` in flash-attention's scheduler / paged-KV / mainloop ran
      per tile (the ISETP 24M vs 6M, BRA 36M vs 10M excess). With it (upstream mainloop, fast math, 112 CTAs):
      prefill 2048 @ 64K 4.68 -> **3.61 ms** (sgl-kernel upstream 3.32 on 132 SMs: 1.09x), chat 2048 0.168 ->
      0.122 ms; `fa3only` 3.59 / 0.065 ms. Upstream's other build flags (GDC, extended MMA shapes, trace level 0)
      add nothing. Left: ~57 us fixed on short items in our compact path (0.122 vs fa3only 0.065 ms). The same
      extension compiles the co-launched FA2 executor, whose A4 "~20% mainloop gap" to stock FA2 was measured
      without `-DNDEBUG` too -- to re-measure. Full fixture table running (`a5_fa3_upstream_table_20261017.sh`).
      Scripts: `a5_fa3_upstream_try_20261017.sh` (UPS / PTMA / XN switches), `extra_nvcc_patch.py`,
      `upstream_patch.py`, `fi_a5_patch.py` (pod trees `up_a5`, `fi_a5`).
    - [x] **A5.0c NCU, ours vs upstream FA3 (2026-10-17 12:47, one launch each, both fixtures PASS):** same block
      (384), registers (168), smem (164 vs 167 KB) and occupancy (18.7% warps), but ours issues **2x the
      instructions** for the same work: prefill 2048 @ 64K 1.67G (fast math) vs 0.88G, chat 2048 34M vs 18M. The
      math is identical -- HGMMA 28.93M, MUFU 119.4M, FMUL / FFMA / FADD / FMNMX / F2FP all within 1% -- and the excess
      is ~790M integer/permute/shuffle instructions: LOP3 203M, PRMT 123M (one per score element), IMAD 111M (vs
      15M), SHFL 69M (vs 7M), SHF 65M (vs 5M), LEA 65M (vs 6M), IADD3 62M, plus 29M LDGSTS (cp.async KV gather; upstream
      uses TMA). **Root cause: the profiled kernel is `DedicatedFA3PrimaryKernel<SparseCollectiveMainloop<
      BatchPrefillPagedParams ...>>` -- flashinfer's own sparse/paged hopper mainloop, not flash-attention's.** The
      upstream-mainloop role (`hopper/fa3_upstream.cuh`, `FLASHINFER_FA3_UPSTREAM`) was never enabled by the loader
      after the 0.7 port (its last attempt failed on a missing flash-attention tree, then a `wgmma_plan` params
      mismatch, archive log 2026-10-09). Every FA3 number since -- NCU 43%, KV96, locality, unpacked, multicast,
      SM budget -- measured flashinfer's mainloop. Next: A5.0d, build the role on the upstream mainloop.
      `bash a5_fa3_ncu_20261017.sh`; `python ncu_compare.py <dir> <fixture>`; per-opcode via `ncu --import <rep>
      --page raw --csv --print-metric-instances details --metrics sass__inst_executed_per_opcode`.
    - [x] **A5.0b `--use_fast_math` (2026-10-17 12:39-12:46, ON/OFF interleaved x2, 112 CTAs, all 28 runs 9/9):**
      ON/OFF = 0.940-0.977 on every fixture (spread <=0.4%), e.g. prefill 2048 @ 64K 4.628 -> 4.411 ms, chat 2048
      0.171 -> 0.161 ms. Real but small: ours/upstream is still 1.37-1.38x on long items and 2.0-2.95x on short
      ones. Adopt fast math (upstream's own build flag); the gap is elsewhere -> A5.0c NCU, ours vs upstream.
      `bash a5_fa3_fastmath_20261017.sh` (tree `up_a5` = `up_tma_multicast` + `fastmath_patch.py`).
    - [x] **Gap located (2026-10-17 12:16-12:23, H200, all 28 runs 9/9 checks vs fp32 causal reference, repeat
      spread <=3.1%):** that NCU compare was confounded (ours on 112 CTAs vs flashinfer FA3 on 132, and flashinfer
      FA3 is not what serving runs). FA3-only fixtures, ms, two-repeat means:
      | fixture | ours 112 | ours 132 | flashinfer FA3 | upstream FA3 (serving) | ours/upstream |
      |---|---|---|---|---|---|
      | 8 x 1024 @ 32K | 11.03 | 10.25 | 7.42 | 5.99 | 1.71x (132) |
      | chat 2048 fresh | 0.171 | 0.173 | 0.081 | 0.060 | 2.88x |
      | chat 4 x 512 fresh | 0.069 | 0.073 | 0.038 | 0.032 | 2.32x |
      | append 512 @ 48K | 0.856 | 1.493 | 0.682 | 0.586 | 1.46x (112) |
      | agent 4 x 512 @ 64K | 4.75 | 4.77 | 4.11 | 3.28 | 1.45x |
      | prefill 2048 @ 64K | 4.65 | 4.65 | 3.96 | 3.21 | 1.45x |
      | 16 x 16 @ 1K | 0.073 | 0.045 | 0.071 | 0.024 | 1.86x |
      Three separate costs, none of them the SM budget: (1) ~1.45x mainloop on long compute-bound items;
      (2) ~0.1 ms fixed per call, dominant on chat-size items; (3) our split-KV merge path: append 512 @ 48K
      plans 112 tiles of 48K keys (one wave at 112 CTAs); at 132 it splits in two and gets 0.64 ms *slower*.
      Flashinfer FA3 is itself 1.16-1.35x behind upstream. Packed vs unpacked Q (Codex, <0.6%) and TMA vs
      gather KV (B3) were already ruled out. Our build lacks `--use_fast_math`, which upstream FA3 uses
      (softmax exp2 throughput bounds FA3 at head dim 128): A5.0b tests it next.
      Commands: `bash a5_fa3_gap_20261017.sh` (pod `/dev/shm/w`, harness
      `drivers/a5_fa3_reference_20261017.py` = tma_multicast_reference + an upstream sgl-kernel arm), both in
      `tasks/debug_kernel_server_perf/a5_fa3_20261017/`; binaries = tma_sm_budget GATHER arm (tree
      `up_tma_multicast`, `fi_tma_multicast`); sglang `ad2d19bab`, flashinfer `1a73e67`.
  - [x] A5.1 per-item q-tile choice (closed by bound, 2026-10-17): q64 wins only on short-KV few-row items (A1, up to
    24%), and those rows are verify-width and route to FA2 (A3: FA2 wins every verify-width row); FA3 rows are
    prefill/append chunks where q128 is never slower. No serving item reaches the win region.
  - [x] A5.2 split-KV reduction (2026-10-17): on the parity FA3 role the split path is exact (flash-attention's
    combine; the double merge fixed), but splitting itself loses where it fires (append 512 @ 48K at 132 CTAs: 0.679
    ms split vs 0.627 unsplit at 112), so the planner should not split FA3 rows; with no splits there is nothing to
    reduce, and the FA2 side's merge is stock BatchPrefill's. A faster reduction (JIT / DSMEM) only pays on a path
    the scheduler should avoid -- not built.
  - [x] A5.3 FA3 co-run with FA2 on the remaining SMs (2026-10-17): -18% where roles balance, <=2% serving bound.
    - [x] **Co-run vs serving split on mixed fixtures (2026-10-17 13:45-13:51, 30 runs, each 13/13 vs fp32 causal,
      repeat spread <=4.3% except chat 6.6%):** arms per run -- *split* (serving: sgl-kernel FA3 on all SMs, then
      stock FA2 BatchPrefill), *corun_sgl* (sgl-kernel FA3 with `sm_margin` = 132-B on one stream || stock FA2 on
      another), *corun_dyn* (our flash-attention build on its dynamic scheduler capped at B || stock FA2), *ours*
      (compact FA3 role, static lists + native FA2 as PDL dependent). Best B per fixture, vs split:
      | fixture (FA3 rows + FA2 q=4 rows) | split ms | ours | corun_sgl | corun_dyn |
      |---|---|---|---|---|
      | fresh 512 + 46 @ 48K | 2.335 | -0.2% (112) | +9.4% (96) | -0.8% (112) |
      | append 512 @ 48K + 46 @ 48K | 3.208 | -6.9% (120) | -16.7% (120) | **-17.9% (120)** |
      | prefill 2048 @ 64K + 8 @ 16K | 3.417 | +9.5% (120) | -3.3% (120) | +11.9% (120) |
      | chat 2048 + 32 @ 8K | 0.282 | +20.3% (120) | -1.7% (120) | +3.0% (112) |
      | agent 2 x 512 @ 64K + 24 @ 32K | 2.497 | +4.7% (112) | +0.1% (120) | +5.4% (120) |
      **Co-running pays when the roles take comparable time and one cannot fill the GPU alone** (append: a 0.6 ms
      FA3 wave hides under ~2.6 ms of bandwidth-bound verify rows, -18%); where one role dominates, sequential
      split ties or wins, and B=96 loses everywhere but append. So B must be chosen per step, "no co-run"
      included -- a dispatch decision (A2/A3), not a constant. corun_dyn trails corun_sgl wherever FA3 dominates:
      the remaining 1.1x per-SM gap of our build. Our static-list path (ours) loses on chat/agent (+5..+20%): its
      static per-CTA lists, as in A5.0f. Script `a5_mixed_corun_20261017.sh` (harness arms in
      `a5_fa3_reference_20261017.py`).
    - [x] **Same table on the parity build (paged TMA, 2026-10-17 13:59-14:04, 30 runs 13/13, spread <=5.7%):**
      corun_dyn (our FA3 role, dynamic scheduler on B SMs || stock FA2) vs split: append 512 @ 48K + 46 verify
      **-18.4%** (B 120); agent 2 x 512 @ 64K + 24 verify -1.8% (112); chat 2048 + 32 verify -1.2% (112); fresh 512 +
      46 verify -0.5%; prefill 2048 @ 64K + 8 verify +1.0% (corun_sgl -3.1%). corun_dyn now beats corun_sgl on every
      fixture but prefill; the static-list path never wins (up to +27% on short items) and is retired for the
      dynamic scheduler. Serving decision: co-run when both roles carry real work (an agent append beside many
      verify rows), split otherwise -- to be wired into the serving split path and measured end to end.
      `bash a5_mixed_corun_tma_20261017.sh`.
    - [x] **Serving bound for co-running (2026-10-17 14:2x, from the TraceLab C=4 step traces of 2026-10-16):** per
      mixed step (prefill rows + decode/verify rows), FA3 time = causal prefill FLOPs at 500 TFLOP/s, FA2 time =
      verify-row KV bytes at 3 TB/s; an ideal co-run saves min of the two. Our arm (EDF + deadline reservation,
      10,629 steps, 1,784 mixed): modelled attention is 79% of mixed-step time, but the roles are badly
      unbalanced -- min/max p10/p50/p90 0.01/0.03/0.18 -- so a perfect co-run saves **1.2% of all GPU time (2.1% of
      mixed-step time)**; stock MTP 0.4%. A long-context prefill chunk dwarfs the verify rows beside it, which is
      also why the 2026-10-13 serving overlap A/B found nothing. **Co-running cannot move serving metrics on agentic
      workloads; FA3 prefill speed is the lever**, and those items already run at ~580-600 TFLOP/s (prefill 2048
      @ 64K: 1.9 TFLOP per layer in 3.19 ms) with our role at parity with upstream. `corun_bound_from_steptrace.py`
      on `steptrace_tracelab_20261016/rep1/tracelab_c4/{MTP,EDF_RESV}/server_*.log`.
  - [x] A5.4 L2 locality (closed by measurement, 2026-10-17): on the parity build L2 hit rate is already 98% for both
    our role and sgl-kernel FA3 (NCU, prefill 2048 @ 64K / agent 4 x 512 @ 64K); the 2026-10-14 pairing gain was on
    the flashinfer mainloop's 12% hit rate.
  - [x] A5.5 TMA multicast (2026-10-17, negative): flash-attention's ClusterM=2 with paged TMA **deadlocks** on the
    first call (GPU 100%, no progress for 8 min, killed by PID): multicast needs both CTAs of a cluster on paired
    tiles in lockstep, which neither our lists nor upstream's varlen scheduler provide -- flash-attention's own
    dispatcher disables clusters for causal / varlen / split. Headroom is also small: on the parity build NCU shows
    98% L2 hit and 72% tensor-pipe activity, instruction-for-instruction equal to sgl-kernel FA3 (881.5M vs 881.2M on
    prefill 2048 @ 64K), so K/V loads are not the bound; the 19-22% zero-KV-load ceiling was measured on the
    flashinfer mainloop. Not pursued. `a5_fa3_ncu_parity_20261017.sh`.
  - [x] A5.6 in-kernel cross-role backfill (closed by bound, 2026-10-17): backfill only matters when FA3 and FA2 run
    concurrently, and the serving bound for any co-run is 1.2% of GPU time on TraceLab (A5.3); the hardware
    co-launch already backfills. Not rebuilt on the flash-attention role.

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
    **With interactivity spread (2026-10-16):** [`eval_s12_v2_20261012_userpct.png`](debug_kernel_server_perf/userpct_20261016/eval_s12_v2_20261012_userpct.png)
    -- the tok/s/user axis now shows each point's per-request distribution: marker = median request,
    thick bar p10-p90, whisker p1-p99 (p1/p10 are the slow-user tail, i.e. the latency P99/P90 users).
    Medians reproduce the committed CSV exactly. Same for the EDF rate sweeps:
    [`edf_ratesweep_20261015_userpct.png`](debug_kernel_server_perf/userpct_20261016/edf_ratesweep_20261015_userpct.png),
    [`edf_ratesweep2_20261015_userpct.png`](debug_kernel_server_perf/userpct_20261016/edf_ratesweep2_20261015_userpct.png)
    (their agent tenant ran on the retracted session-rate axis; the chat x-axis is a request rate).
    Requests whose tokens all arrive in one burst at the end (< 1 ms decode per token) are excluded
    from tok/s/user and counted in `burst_frac`: 0 here (one request in one row), but 17-29% of stock
    PD-disagg requests on H20 (`pd_colocation_vs_disagg.md`).
    Command, CPU only on the pod's JSONs: `eval_curves.py --root /dev/shm/w/<sweep> --slo
    '{"chat":[2000,50],"codex":[10000,80]}' --csv <sweep>_userpct.csv; plot_eval.py <csv> <png>`.
    Chat rates vs the reference list (per GPU): this sweep's 4-24 req/s on one H200 for a 7B model
    sits inside semi-PD's Llama3-8B ShareGPT range (11-16 req/s per A100) and above SLOs-Serve's
    1.5-4.5 req/s per A100 (OPT-7B/13B); DistServe / Sarathi-Serve use 0.5-10 req/s.
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
    uncached work instead.
    **Both variants were inert (2026-10-15 18:1x):** mine also fired ~once per run (TraceLab C=4,
    17:50-18:05). Cause, found in the server log: the EDF arms run the FCFS/priority policy, which
    prefix-matches waiting requests only when the radix cache `supports_fast_match_prefix()` (it does
    not here), so `num_matched_prefix_tokens` stays 0 and a 99%-cached 27K round is charged 27K
    tokens > the 2K budget -> the reservation loop breaks. Log: 88 s stretch with `#queue-req: 1`,
    every step `#new-seq: 1` (the cold chunked prompt), token usage <= 0.41 (not memory). The TTFT
    tail is a handful of cache-hit rounds (<600 uncached) waiting 60-96 s -- also in MTP (one at
    96 s). Single C=4 runs are too noisy to judge (two identical EDF controls: mean TTFT 3.9 vs
    4.6 s). Fix: match the first 16 waiting requests inside the reservation
    (`edf_reserve_matched.patch`, tree `up_edf_reserve2`); rerun with C=4 twice + same-tree control,
    C=8, cohort C=16 (`edf_reserve_matched_20261015.sh`), queued behind the A3 A/B and S3.
    **Still inert (23:13-23:20, TraceLab C=4):** <20 firings in 6 min with matching, so unmatched
    prefixes are not the whole cause; run stopped. One diagnostic arm queued
    (`edf_reserve_diag_20261015.sh`, tree `up_edf_reserve3`): logs the head waiting request's
    length, matched prefix, arrival and deadline against the chunked request's.
    **Root cause found (2026-10-16 01:3x, server-log trace of the 42-61 s outliers):** the outlier IS
    in the waiting queue (`#queue-req: 1` for 61 s) behind a cold ~390K prompt prefilled 2,048 tokens
    per step; it is a cache-hit round with 2,377 uncached tokens -- more than the whole step budget,
    and sglang allows one chunked request, so no budget reservation can admit it (the diagnostic arm:
    20 reservations in 15 min; MTP shows the same pattern, one 96 s outlier). Fix: EDF over the chunked
    request -- rescore waiting deadlines on uncached work and park the chunked prompt (prefix kept in
    the radix tree, row released as a retraction, re-queued next step) when a short earlier-deadline
    request waits (`edf_preempt_chunk.patch`, `SGLANG_EDF_PREEMPT_CHUNK`, tree `up_edf_preempt`).
    A/B with the EDF serving gate, interleaved in one queue: `edf_preempt_20261016.sh`.
    Gates (01:32-01:35): EDF control and EDF+preempt both PASS (long prefix 52) **but parked 0
    times** -- the standard gate sends short prompts first, so it cannot exercise parking. Forced
    gate queued: long prompts first, short ones while they are chunked (chunk 256), stock MTP in the
    same order as the order-only null case, park arm must park >= 1
    (`edf_park_gate_20261016.sh`, `drivers/serving/srv_correctness_edf_park_20261016.py`).
    **Serving, TraceLab (2026-10-16 01:35-03:40, all arms clean, peak = C, one tree/queue):**
    | C | arm | tput | mean TTFT | <1K-uncached TTFT mean / med | mean ITL | TPOT |
    |---|---|---|---|---|---|---|
    | 4 | MTP | 93.8 | 4178 ms | 1040 / 201 | 29.9 | 26.7 |
    | 4 | EDF control (x2) | 116.6 / 114.0 | 3941 / 4823 | 3107 / 2254 | 22.1 / 20.6 | 20.7 / 21.3 |
    | 4 | EDF + preempt (x2) | 119.4 / 119.1 | **2558 / 2518** | **326 / 284** | 24.0 / 24.1 | 23.3 / 24.4 |
    | 8 | MTP | 94.7 | 6068 | 2003 / 311 | 59.7 | 77.6 |
    | 8 | EDF control | 124.1 | 7771 | 6166 / 294 | 37.0 | 37.3 |
    | 8 | EDF + preempt | 125.2 | 5136 | 1762 / 405 | 42.2 | 49.6 |
    Parks: ~20-40 per run (urgent 197-546 uncached tokens behind 89-110K-token remainders). **C=4:
    preempt vs MTP TTFT -39/-40%, tput +27%, ITL -20% (control -28..-31%); repeat TTFT within 2%
    (control 22% apart).** C=8: TTFT only -15% vs MTP (-34% vs control), ITL -29%, tput +32%,
    TPOT -36%; mean TTFT is now set by the 17 cold 64K+ prompts (39 s vs MTP 45 s) and the 16-64K
    bin got worse (12.0 vs 9.7 s: over the 16K cap they cannot preempt a 390K prompt). Preemption
    raises ITL 2-5 ms over the control (hit rounds decode instead of queueing). Next: uncapped
    preemption = shortest-remaining-work-first (`edf_preempt_srpt_20261016.sh`), queued after the
    park gate. Correctness of parking pending the forced gate.
    **Bug found and fixed (04:05):** the cohort C=16 preempt arm hung after its first park (no server
    log for 18 min): parked requests were re-queued inside `_edf_rescore_and_park`, which runs after
    `get_new_batch_prefill`'s early returns, so a parked request that was the only remaining work was
    stranded. Re-queue moved to the top of `get_new_batch_prefill` (`edf_preempt_chunk_v2.patch`, tree
    `up_edf_preempt` patched in place before the park gate and the SRPT runs). The TraceLab preempt
    numbers above ran v1 (constant arrivals there always re-triggered the re-queue; every request
    completed), so they stand provisionally; the SRPT run re-measures the 16K-cap arm on v2. The
    cohort C=16 preempt point is void (rc 143, killed).
    **Forced park gate v1 (04:39-04:43, `edf_park_gate_20261016.sh`):** order-only null cases PASS (stock
    MTP long-first: long prefix 11; EDF control long-first: 96), but the park arm parked 0 times and
    failed its must-park check (rc 1; outputs still PASS, long 96): EDF already serves the short
    requests ahead of the WAITING long prompts, and the in-progress long had < 2x the short work left.
    v2 queued: one long, then the shorts 0.1 s later, then the other longs.
    **Preempt on the fixed code (v2), TraceLab C=8, first pair (04:48-05:16):** 16K cap: TTFT 4113 ms
    (**-32% vs MTP 6068**; v1 gave 5136 -- the stranded-request bug cost ~1 s), ITL 44.6 (-25%), tput
    125.8 (+33%); uncapped (SRPT): TTFT 3963 (**-35%**), ITL 44.8 (-25%), tput 127.3 (+34%), 16-64K bin
    6.1 -> 4.1 s. TTFT now clears at C=8; ITL does not (needs <= 41.8 ms). Mechanism: EDF runs ~1900
    mixed steps vs MTP's 670 (2048 vs 8192 prefill tokens per step at 100-390K context), and the EDF
    budget asks for less than the 2048 floor (`requested=8192 admitted=2050`); preemption adds decode
    concurrency (5.35-5.46 vs 4.98 running per decode step). One GPU sits on a TTFT/ITL frontier here
    (floor 1024: ITL -8%, TTFT +19%); the remaining lever is placement across nodes (PD point 1).
    **Uncapped vs 16K-cap preemption, complete (`edf_preempt_srpt_20261016.sh`, 04:48-06:32, all arms
    clean, peak = C, parks 20-60 per run; same-queue MTP at C=8: TTFT 6538 ms, ITL 58.8, tput 93.5):**
    | arm | TTFT | ITL | TPOT | tput | vs MTP (TTFT / ITL / tput) |
    |---|---|---|---|---|---|
    | C=8 16K cap (x2) | 4113 / 5016 | 44.6 / 42.9 | 54.4 / 48.8 | 125.8 / 125.4 | -37/-23% / -24/-27% / +35/+34% |
    | C=8 uncapped (x2) | **3963 / 3751** | 44.8 / 45.5 | 50.4 / 50.3 | 127.3 / 125.9 | **-39/-43%** / -24/-23% / +36/+35% |
    | C=4 uncapped (x2) | **2369 / 2378** | 24.9 / 25.0 | 32.1 / 32.1 | 120.7 / 119.8 | **-43%** / -17% / +28% (MTP from the 01:35 queue) |
    Uncapped preemption (shortest remaining work first among prompts) is adopted: better and steadier
    TTFT than the 16K cap. **Bar status, full-length TraceLab: TTFT and throughput clear at C=4 and
    C=8; ITL misses (-17% / -23..-24%), and C=4 TPOT is worse than MTP (+20%)** -- more requests decode
    beside 2048-token chunks at 100-390K context.
    **Forced park gate v2 PASS (06:32-06:36, `edf_park_gate_v2_20261016.sh` = `edf_park_gate_20261016.sh`
    with the v2 order):** the park arm parked (logged: chunked remainder 10,464, urgent 2,048 tokens,
    slack 1.96 s) and agrees with the stock MTP_LPT_Q4G reference on long prefix 11 -- the same as stock
    MTP in this order (11, the order-only noise floor; the EDF control without parking: 96); 0
    tracebacks / fallbacks. Parked prefills resume correctly; the preemption serving arms are gated.
    **Final single-GPU config vs MTP, mixed workload (2026-10-16 06:38-07:27, `mixed_final_20261016.sh`;
    split + EDF 50 ms + floor 2048 + uncapped preemption; frozen 32-64K cohort at program concurrency C
    + natural ShareGPT chat; all 8 arms clean, peak = C, parks 20-120; acceptance within 0.02):**
    | point | tenant | tput | mean TTFT | TPOT | mean ITL | P99 ITL |
    |---|---|---|---|---|---|---|
    | C=8, chat 4/s | agent | **+44%** | **-35%** | -38% | **-31%** | -85% |
    | C=8, chat 4/s | chat | **+11%** | **-83%** | -42% | **-39%** | -74% |
    | C=16, chat 4/s | agent | +40% | -23% | -58% | -46% | -86% |
    | C=16, chat 4/s | chat | +12% | -62% | -42% | -39% | -75% |
    | C=8, chat 8/s | agent | +22% | -19% | -43% | -28% | -83% |
    | C=8, chat 8/s | chat | +20% | -57% | -13% | -16% | -73% |
    | C=16, chat 8/s | agent | +27% | -27% | -52% | -34% | -81% |
    | C=16, chat 8/s | chat | +25% | -44% | -20% | -19% | -72% |
    **C=8 + chat 4/s clears the bar on every metric for both tenants** (chat throughput is near the
    offered load). Misses elsewhere: agent mean TTFT at C=16 / chat 8 (-19..-27%; MTP agent TTFT is
    18-44 s, i.e. overload, where the agent's own prefill backlog dominates) and chat mean ITL / TPOT
    under chat 8/s overload (-13..-20%). P99 ITL drops 72-86% everywhere. Data
    [`mixed_final_20261016.txt`](debug_kernel_server_perf/resume_20261012/mixed_final_20261016.txt),
    table `drivers/serving/mixtab.py mixed_final_20261016` (pod `/dev/shm/w`).
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
    - [x] **EDF earlier-deadline reservation (2026-10-15, done 17:06):** under EDF, cap the chunked
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
      **TraceLab C = 8 (16:55):** mean TTFT 7961 -> 5759 ms (-28%), short appends 6561 -> 3673 (-44%),
      ITL 36.6 -> 41.3 ms (+13%), throughput +2%; still above the sweep's MTP for short appends
      (2179). The reservation fired < 200 times: at C = 8 the long prompts are already past their own
      deadline, so EDF correctly ranks them first and nothing earlier waits. **Short-request reservation (mode 2)**
      (`edf_reserve_short_mode2.patch`, tree `up_edf_resv2`, `SGLANG_EDF_RESERVE=2`) also reserves
      for later-deadline requests whose whole prefill fits in half the step, and beside a chunked
      request admits only the reserved ones (an unreserved request ahead would end admission).
      Not smoke-tested separately; added as a fourth arm of the confirmation run.
      **Cohort C = 16 + chat 8 req/s (17:06):** no effect, within noise -- agent tput / TTFT / ITL
      138.6 / 32.0 s / 16.7 -> 139.9 / 31.6 s / 16.8, chat 1549 / 23.8 s / 24.2 -> 1559 / 23.1 s /
      23.8. Fired < 200 times; here TTFT is queueing past capacity (32 s), which reordering within a
      step cannot remove. Deadline reservation helps only where long prompts block short ones (TraceLab).
    - [x] **Reservation + lower floor (done 17:50):** chat mean ITL misses (-24% at cohort C=16)
      because the 2048 floor binds every step; floor 1024 and 512 with the reservation on, TraceLab
      C = 8 and cohort C = 16 (`edf_floor_resv_20261015.sh`). S3 moved behind it.
      **Done 17:50, all 4 arms clean.** TraceLab C = 8, floor 2048 / 1024 / 512 (all with
      reservation): mean TTFT 5759 / 9505 / 13220 ms, short appends 3673 / 6810 / 10486, mean ITL
      41.3 / 33.6 / 30.3, P99 ITL 282 / 178 / 136, tput 126.3 / 122.3 / --. Cohort C = 16 vs the
      sweep's MTP (cross-queue), agent and chat tput / TTFT / TPOT / ITL / P99 ITL:
      | floor | agent | chat |
      |---|---|---|
      | 2048 | +31% / -28% / -55% / -39% / -85% | +29% / -56% / -25% / -24% / -73% |
      | 1024 | +23% / -17% / -64% / -49% / -88% | +23% / -44% / **-30% / -30%** / -83% |
      | 512 | +20% / -11% / -68% / -54% / -90% | +20% / -39% / -33% / -33% / -84% |
      The floor is a straight TTFT-vs-ITL dial: lowering it buys chat mean ITL (the -30% bar is
      crossed at 1024) and pays in agent TTFT and throughput. No single floor clears both chat ITL
      and agent TTFT at this overloaded point. Floor 2048 stays the default; the confirmation run
      keeps it.
    - [x] **Same-queue confirmation (done 23:13, `confirm_resv_20261015.sh`):** MTP | split
      3072 | EDF + deadline reservation | EDF + short-request reservation, interleaved x2, TraceLab C = 4 / 8 and cohort
      C = 16. **Rep 1, TraceLab C = 4 (same queue, vs MTP):** split tput +23% / mean TTFT +28% / ITL
      -33%; EDF+reservation +28% / **-2%** / -28% (P99 ITL -70%); EDF+short-request reservation +22% / +2% / -25% (short
      appends 761 ms vs MTP 945, but 4-32K requests 4597 vs 2427 and cold 28.9 s). **Why mean TTFT
      stays near MTP:** the 19 cold >= 32K prompts (159K uncached on average, 190 PFLOP) are 45-69% of
      all TTFT; they reach 331 (MTP) / 396 (split) / 446 (EDF+reservation) TFLOP/s including queueing,
      vs a ~700 TFLOP/s FA3 floor of 14.3 s per prompt (`cold_bound.py`). -30% mean TTFT needs cold
      prefill near ~16 s, i.e. larger chunks, which the 50 ms ITL budget forbids (floor 2048 binds).
      **Rep 1 complete (20:42), all 12 arms clean**, vs same-queue MTP (tput / mean TTFT / mean ITL /
      P99 ITL):
      | point | split 3072 | EDF + deadline reservation | EDF + short-request reservation |
      |---|---|---|---|
      | TraceLab C=4 | +23% / +28% / -33% / -65% | +28% / -2% / -28% / -70% | +22% / +2% / -25% / -61% |
      | TraceLab C=8 | +32% / +15% / -33% / -78% | +33% / +4% / -33% / -79% | +35% / -8% / -31% / -79% |
      | cohort C=16 agent | +30% / -28% / -37% / -84% | +31% / -27% / -42% / -88% | +12% / -2% / -45% / -86% |
      | cohort C=16 chat | +29% / -37% / -25% / -66% | +30% / -56% / -24% / -73% | +17% / **-93%** / -9% / -76% |
      Short-request reservation reserved 4,200+ times on the cohort and serves chat almost at once (TTFT 51.8 -> 3.8 s)
      but starves agent prefill; at TraceLab C=8 it gives the best mean TTFT (-8%) yet short appends
      still wait 2780 vs MTP 1903 ms (its half-step cap excludes 1-4K tool-output appends). Deadline reservation is
      the balanced arm.
      **Final, both reps (23:13), all 24 arms clean, every peak = C.** Rep-to-rep spread is 0-2% on
      throughput / ITL and 0-13% on mean TTFT (`confirm_aggregate.py confirm_resv_20261015`). Mean of
      2 reps vs same-queue MTP (tput / mean TTFT / mean ITL / P99 ITL):
      | point | split 3072 | **EDF + deadline reservation** | EDF + short-request reservation |
      |---|---|---|---|
      | TraceLab C=4 | +21 / +37 / -33 / -67% | +26 / -4 / -27 / -68% | +24 / -10 / -24 / -62% |
      | TraceLab C=8 | +33 / +9 / -33 / -79% | +34 / -5 / -31 / -78% | +35 / -17 / -29 / -78% |
      | cohort C=16 agent | +30 / -28 / -37 / -84% | +32 / -28 / -43 / -88% | +13 / -3 / -46 / -86% |
      | cohort C=16 chat | +28 / -37 / -25 / -66% | +30 / -57 / -24 / -73% | +17 / -93 / -9 / -75% |
      **Verdict vs the bar (>=30% TTFT and ITL, >=10% throughput):** deadline reservation clears throughput
      (+26..+34%) and P99 ITL (-68..-88%) at every point, mean ITL at TraceLab C=8 and the cohort agent
      tenant, and chat TTFT; it misses mean TTFT on TraceLab (-4 / -5%; cold >=32K prefills are half
      of all TTFT and compute-bound, see above), cohort agent TTFT (-28%), and mean ITL for TraceLab C=4
      (-27%) and chat (-24%). **The bar is not met on all metrics.** Short-request reservation is not promoted: it trades
      agent throughput and TTFT for chat TTFT. Static split 3072 is dominated by deadline reservation.
    - [x] **Agent ITL SLO 80 ms (done 00:19, not promoted):** TraceLab is agent-only and the
      agent tenant's evaluation SLO is 80 ms; EDF + reservation with `ITL_SLO_MS=80`, TraceLab C = 4 /
      8, two reps (`edf_slo80_20261015.sh`). S3 behind it.
      Rep 1-2 TraceLab C=4: mean TTFT 3319 ms (-5% vs MTP), cold 27.2 s (368 TFLOP/s vs 446 at 50 ms),
      admitted chunk 2049-2654: the larger budget barely enlarges chunks and does not speed cold prefill.
      **Done (00:19), 4 arms clean.** Mean TTFT rep1 / rep2: C=4 3319 / 2570 ms (mean -16% vs MTP),
      C=8 5740 / 7605 ms (mean +9%); throughput 117-124 tok/s and ITL equal to the 50 ms arms. The rep
      spread (25-30%) exceeds any effect; **not promoted** -- chunk size does not limit TTFT, consistent
      with prefill being near compute-bound (below).
    - [x] **Is TraceLab TTFT compute-bound? (2026-10-16, traced 04:39):** the SM-cap hypothesis does not apply --
      the split launches upstream FA3 with `sm_margin` 0 (all SMs); `SGLANG_FUSED_SM_FA3` only sizes
      the fused single-launch path. In-server prefill rate from the existing cohort step traces
      (`steptrace_prefill_rate.py`, single-prefill steps incl. co-scheduled decode time): chunk 2048
      ~500 TFLOP/s, chunk 8192 ~570 TFLOP/s, near the H200 FA3 ceiling (~650-750). Prefill is close to
      compute-bound; small chunks cost ~12-15%; the rest of cold TTFT is cold prompts queueing behind
      each other. TraceLab C=4 step trace (MTP | EDF + deadline reservation) queued:
      `steptrace_tracelab_20261016.sh`, after the output gate; S3 behind it.
      **TraceLab C=4 traced (04:39), both arms clean.** In-server prefill rate on single-prefill
      steps, prefix <32K / 32-128K / 128-256K / >=256K (TFLOP/s, co-scheduled decode time included):
      MTP 401 / 373 / 359 / 351 (FA2 BatchPrefill, chunk 8192); EDF + deadline reservation 507 / 476 /
      484 / 494 (FA3 prefill rows, chunk 2048) -- **+30% prefill rate from the kernel, holding at
      >= 256K prefixes**. Total GPU-busy time 1029 -> 831 s (-19%); this traced pair's mean TTFT 3437
      -> 2843 ms (-17%; one rep, tracing on). The split reaches ~70% of the ~700 TFLOP/s FA3 ceiling;
      the rest is the 2048 chunk and the decode/verify rows sharing each step. Mean TTFT gains less
      than the prefill rate because TTFT also contains cold prompts queueing behind each other and
      the decode share of every step (which is what buys the ITL gain). ~~Single-GPU mean TTFT on
      full-length TraceLab is therefore bounded near the prefill-rate gain (~-20..-30%) at best; the
      30% bar there needs more prefill capacity (multi-node, PD points) rather than scheduling.~~
      **Retracted (2026-10-16 06:0x):** the parallel session's EDF chunk preemption (park the chunked
      long prompt, serve uncached-short waiting rounds; `edf_preempt_chunk_v2.patch`) reaches TraceLab
      mean TTFT -39/-40% (C=4) and -32/-35% (C=8, 16K cap / SRPT) vs MTP on one GPU. The prefill-rate
      measurement stands; the inference from it was wrong -- most of the remaining TTFT was blocking,
      not compute. The TraceLab miss is now mean ITL (-20..-25%).
    - [x] **Serving-output gate, reservation trees (2026-10-16 01:33, `resv_gate_20261015.sh`):**
      Codex's EDF gate driver (`drivers/srv_correctness_edf_20261014.py`, BENCH_CHUNK 512, 1200
      decode tokens, guard 0), all arms 0 tracebacks / fallbacks / width mismatches, graphs on.
      MTP repeat vs MTP (noise floor): min common prefix short 15 / long 39, PASS. Split + EDF +
      deadline reservation: short 1 / long 11, PASS by the rule (long >= 8); short-request
      reservation: identical 1 / 11. No reservation fired on the gate workload, so both test split +
      EDF. **Not a clean pass:** far below the floor, and earlier split-only gates kept long prefixes
      of 52-96. EDF changes batch composition, and greedy bf16 outputs depend on it, so the MTP repeat
      (same schedule) is an optimistic floor -- control queued: split-only, EDF-only (stock FA2 4-mode)
      and split + EDF, each vs a fresh MTP (`gate_control_20261016.sh`).
      **Control (04:48), all arms clean:** EDF-only (stock FA2 4-mode) short 18 / long 39 = the MTP floor;
      split-only 1 / 11; split + EDF 1 / 11. The early divergence is the split kernel path, not EDF.
      This matches the Codex role-isolation finding (`stock_fa3_dispatch_20261014.md`): the generation
      difference localizes to the FA3 primary, whose split-KV reduction differs from unsplit by up to
      4.2e-3 per-head relative error, while the attention checks against fp32 pass. Same-input check
      on this tree queued (`split_verify_20261016.sh`: `SGLANG_FUSED_SPLIT_VERIFY=1`, every split step
      recomputed with the stock FA2 wrapper, FAIL above 0.02 scale-normalised error).
      **Same-input check (06:36-06:38), both arms clean:** split-only and split + EDF, every split step
      recomputed with the stock FA2 wrapper on the same inputs: **0 FAIL lines**, sampled error 0.0059
      (FA3 rows 0.0059, FA2 rows 0.0029-0.0059), worst 0.0112 < 0.02. The split's attention agrees with
      stock within tolerance; the earlier greedy divergence (1 / 11 with this driver; the canonical
      `srv_correctness.py` driver gives long 52 on the same split code) is bf16 FA3-vs-FA2 numerics
      flipping near-tie tokens, not a wrong result. **Gate closed: PASS** for both reservation trees
      (no reservation fired on the gate workload; their only change is scheduling).
  - [ ] **Final-config arms (2026-10-15, queued):** MTP | split static 3072 | split + EDF 50 ms +
    floor 2048, tree `up_edf_floor`, guard 0, no barrier (`s3_standalone_final_20261015.sh`, root
    `s3_standalone_final_20261015`; yields to `codex_needs_gpu`). Replaces the never-run
    2026-10-13 arms. Re-queued 13:1x behind `concurrency_sweep_20261015.sh`; load axis is request
    rate only (Mooncake: recorded per-request timestamps compressed, achieved req/s reported).
    **ShareGPT (2026-10-16 07:27-08:08, 12 arms clean):** req/s achieved / out tok/s / mean TTFT /
    mean ITL / P99 ITL, acceptance 2.04-2.09 in every arm:
    | rate | MTP | split 3072 | split + EDF + floor |
    |---|---|---|---|
    | 4 | 3.99 / 847 / 43 ms / 4.17 / 13.5 | 3.99 / 847 / 49 / 4.16 / 15.8 | 3.99 / 847 / 48 / 4.16 / 16.9 |
    | 8 | 7.99 / 1633 / 48 / 4.68 / 18.5 | 7.99 / 1634 / 55 / 4.76 / 24.9 | 8.00 / 1635 / 57 / 4.80 / 25.0 |
    | 16 | 15.01 / 3074 / 95 / 6.18 / 31.2 | 15.02 / 3077 / 72 / 6.20 / 33.9 | 14.93 / 3059 / 76 / 6.21 / 34.0 |
    | 32 | 24.03 / 5049 / 17.3 s / 9.10 / 44.1 | 24.73 / 5198 / 14.7 s / 8.75 / 47.3 | 22.92 / 4818 / 20.2 s / 9.47 / 43.9 |
    **No gain on short-prompt ShareGPT:** below capacity all arms are equal within a few ms (the split
    adds ~6 ms TTFT and some P99 ITL); at saturation (32/s) the split gives +3% throughput and -15%
    TTFT, while EDF + floor loses (-5% throughput, +16% TTFT: the 2048 floor and 50 ms budget throttle
    prefill when every request is short). The 4-mode gains are specific to long-context mixing
    (agent / TraceLab / mixtures), as the B1 and step-trace evidence predicts.
    **GSM8K / MATH-500 / HumanEval / Code-Feedback (08:08-09:21, 36 arms, 0 tracebacks / fallbacks):**
    dataset size caps N (GSM8K 1319, MATH-500 500, HumanEval 164 prompts), so their higher-rate points
    exit rc=1 on the harness's "did not complete all N" check after serving every available prompt --
    all three arms see the same requests. MTP vs split vs EDF+floor, mean TTFT ms / mean ITL / P99 ITL:
    | point | MTP | split 3072 | EDF + floor |
    |---|---|---|---|
    | GSM8K 8 | 30 / 3.38 / 15.0 | 38 / 3.44 / 11.8 | 38 / 3.43 / 11.7 |
    | GSM8K 32 (31.0 req/s) | 36 / 4.99 / 18.4 | 51 / 5.34 / 25.0 | 50 / 5.33 / 24.8 |
    | MATH-500 32 (27-28 req/s) | 91 / 6.16 / 21.7 | 104 / 6.24 / 29.7 | 222 / 6.58 / 28.6 |
    | HumanEval 32 | 37 / 5.30 / 18.2 | 53 / 5.94 / 26.4 | 53 / 5.80 / 25.9 |
    | Code-Feedback 16 | 63 / 5.65 / 22.2 | 77 / 5.79 / 28.6 | 82 / 5.82 / 28.7 |
    | Code-Feedback 32 (saturated, 18 req/s) | 29.1 s / 6.75 / 25.1 | 27.5 s / 6.55 / 32.4 | 31.5 s / 6.90 / 31.9 |
    Throughput is equal (within 1-2%) everywhere below saturation. **On short prompts the 4-mode
    split is slightly worse:** +8..+16 ms TTFT, +1..+7% mean ITL, +30..+45% P99 ITL (prefill now shares
    the step with verify rows); at saturation (Code-Feedback 32) it gains +2% throughput and -5% TTFT.
    EDF + floor matches the split or is worse. The gains measured on the cohort / TraceLab come from
    long-context prefill mixed with decode; a production deployment should enable the 4-mode step only
    when such prefill is present (a dispatch decision per step, not a fallback).
    **Mooncake conversation / tool-agent (09:21-10:37, 18 arms clean; recorded per-request timestamps
    at 0.5 / 1 / 2x speed, 600 requests, ~3.7K-token prompts):** req/s achieved / mean TTFT / mean ITL /
    P99 ITL, MTP vs split vs EDF + floor:
    | trace, speed | MTP | split 3072 | EDF + floor |
    |---|---|---|---|
    | conversation 0.5x | 1.49 / 606 ms / 6.1 / 13 | 1.49 / 524 / 6.3 / 41 | 1.49 / **452** / 6.6 / 46 |
    | conversation 1x | 2.98 / 634 / 8.6 / 72 | 2.98 / 545 / 8.3 / 81 | 2.98 / **472** / 9.0 / 68 |
    | conversation 2x | 4.26 / 15.6 s / 28.2 / 236 | 4.63 / 9.5 s / 24.8 / 118 | **4.76 / 8.3 s / 24.6 / 96** |
    | tool-agent 0.5x | 2.73 / 589 / 6.6 / 13 | 2.73 / 505 / 6.7 / 45 | 2.73 / **422** / 7.0 / 64 |
    | tool-agent 1x | 5.36 / 624 / 9.0 / 147 | 5.35 / 523 / 8.7 / 83 | 5.36 / **430** / 9.5 / 70 |
    | tool-agent 2x | 6.70 / 7.5 s / 27.4 / 231 | 8.12 / 3.8 s / 23.0 / 113 | **8.13 / 3.1 s / 23.1 / 91** |
    **Medium-length prompts show the gain:** EDF + floor cuts TTFT 25-31% below capacity (ITL +5..+8%);
    at 2x, where MTP falls behind the trace, EDF gives throughput +12% (conversation) / +21% (tool-agent),
    TTFT -47% / -58%, P99 ITL -59% / -60%, mean ITL -13% / -16%. Throughput and TTFT clear the bar at
    2x; mean ITL does not.
    **LongBench v2 -- retracted (2026-10-16 12:11):** the rate grid 0.25 / 0.5 / 1 req/s was carried
    over unreviewed from the 2026-10-13 script. LongBench v2 prompts average ~57K tokens (11.39M input
    tokens / 200 requests) and one H200 prefills ~14.5K tok/s, so capacity is ~0.25 req/s: every point
    was at or past saturation (MTP mean TTFT 57 s at 0.25/s, 241 s at 0.5/s with 0.25 achieved). These
    points measure queue growth, not serving latency; not reported. Stopped after 0.25 and 0.5 (the
    1 req/s arms were killed). The references set load around measured capacity -- Crossflow by request
    rate (timestamp compression), ThunderAgent by concurrency; a long-document set without timestamps
    belongs on program concurrency (documents in flight), not req/s.
    **Scope (user, 2026-10-16):** S3 is ShareGPT, math reasoning (GSM8K, MATH-500), coding (HumanEval,
    Code-Feedback) and agentic web browsing only; LongBench and Mooncake leave S3 (the Mooncake rows above
    stay as recorded). **Web browsing (12:26-12:59):** THUDM/AgentInstruct WebShop, 351 multi-turn
    trajectories (4-10 turns, mean 4.7), converted by `drivers/serving/prep_webshop_agentic.py` and replayed
    round by round with the server's own replies (`sglang-oai-chat`, 64 output tokens per turn); load is
    program concurrency C = 8 / 16 / 32 / 64. Arms MTP | split 3072 | EDF + floor | EDF + floor + chunk
    preemption, tree `up_edf_preempt` (`s3_webshop_agentic_20261016.sh`, runner `run_agentic_standalone.py`).
    **WebShop done (12:26-12:59, 16 arms, 0 tracebacks / fallbacks, 0 parks; acceptance 1.78 in every
    arm).** Every arm completes all 1642 turns of the 351 conversations; rc=1 is only the runner's
    "requests == conversations" check (the bench counts turns). Out tok/s / mean TTFT ms / P99 TTFT /
    mean ITL / P99 ITL:
    | C | MTP | split 3072 | EDF + floor | EDF + preempt |
    |---|---|---|---|---|
    | 8 | 1246 / 40 / 70 / 5.93 / 24.9 | 1162 / 47 / 82 / 6.29 / 30.9 | 1156 / 48 / 86 / 6.31 / 31.5 | 1158 / 48 / 83 / 6.29 / 31.5 |
    | 16 | 1916 / 47 / 92 / 7.77 / 29.4 | 1732 / 59 / 120 / 8.50 / 36.0 | 1724 / 61 / 112 / 8.49 / 36.3 | 1714 / 60 / 105 / 8.57 / 36.3 |
    | 32 | 2577 / 61 / 201 / 11.4 / 41.5 | 2324 / 83 / 235 / 12.4 / 47.4 | 2390 / 81 / 328 / 12.1 / 46.9 | 2383 / 82 / 273 / 12.1 / 47.4 |
    | 64 | 2995 / 375 / 736 / 14.8 / 60.5 | 3148 / 380 / 821 / 13.6 / 64.0 | 2765 / 432 / 888 / 15.7 / 60.7 | 2697 / 447 / 908 / 16.0 / 64.4 |
    **Web browsing loses on one GPU at C <= 32:** throughput -7..-10%, TTFT +18..+36%, ITL +6..+9% for
    every 4-mode arm -- WebShop observations are short (no long prefill to co-schedule; preemption never
    parks), the same regime as the other short-prompt sets. At C=64 the split gains +5% throughput and -8%
    ITL; EDF arms lose there. Consistent across S3: the 4-mode step pays off only where long-context
    prefill shares the GPU with decode (cohort, TraceLab, Mooncake 2x).
    - [x] **4-mode context gate (2026-10-16 13:12-14:33, `mixed_gate_ab_20261016.sh`, 12 arms clean):**
      `SGLANG_MIXED_SPEC_MIN_CTX` (`mixed_spec_ctx_gate.patch`, tree `up_edf_gate` = `up_edf_preempt` +
      gate): a step mixes its prefill into the verify step only if some prefill row's prefix + chunk >=
      the gate, else the prefill runs alone as in MTP; `MIXEDSPEC_GATED` counts gated steps. MTP vs
      EDF + floor + preemption ungated vs gated at 8192 (out tok/s / mean TTFT / mean ITL / P99 ITL):
      | point | MTP | ungated | gate 8K | gated / mixed steps |
      |---|---|---|---|---|
      | WebShop C=16 | 1908 / 47 ms / 7.80 / 29.7 | 1696 / 60 / 8.66 / 36.4 | 1788 / 42 / 8.46 / 43.9 | 1201 / 0 |
      | ShareGPT 16 req/s | 3058 / 97 / 6.25 / 32.0 | 3078 / 68 / 6.26 / 34.5 | 3070 / **54** / 6.41 / 32.8 | 1601 / 1 |
      | cohort C=16 agent | 106.8 / 43.3 s / 28.0 / 503 | **137.3 / 31.6 s / 17.6 / 92** | 126.5 / 34.7 s / 19.6 / 104 | 1401 / 1151 |
      | cohort C=16 chat | 1219 / 51.8 s / 31.4 / 473 | **1528 / 28.4 s / 25.0 / 129** | 1404 / 34.4 s / 28.4 / 142 | |
      | TraceLab C=8 | 94.8 / 6461 / 58.6 / 1276 | 127.4 / 4214 / 43.8 / 310 | 126.8 / 4089 / 44.3 / 293 | 1 / 1801 |
      **Not promoted.** On short prompts it removes most of the mixed-step cost (ShareGPT: throughput
      equal to MTP, TTFT -44%; WebShop: throughput -6% instead of -11%), and TraceLab never gates.
      But on the overloaded cohort, gating the chat-only prefill steps stalls decode during them:
      agent throughput -8%, chat ITL 25.0 -> 28.4 ms vs ungated. Prompt length is the wrong signal --
      mixing pays when the server is loaded and costs at moderate load; a load-based gate (queue depth /
      running count) is the next candidate. Ungated EDF + preemption vs MTP on these runs: TraceLab C=8
      tput +34%, TTFT -35%, ITL -25%, P99 ITL -76%; cohort agent +29% / -27% / -37%, chat +25% / -45% /
      -20%.
    - [x] **Load + context gate -- promoted (2026-10-16 14:38-15:13, `mixed_gateq_ab_20261016.sh`, 9 arms
      clean):** `SGLANG_MIXED_SPEC_MIN_QUEUE` (`mixed_spec_ctx_queue_gate.patch`, tree `up_edf_gateq`)
      re-enables mixing whenever >= Q requests wait; gate = context 8192 OR queue 4. Same-queue, vs MTP
      and ungated EDF + preemption (out tok/s / mean TTFT / mean ITL / P99 ITL):
      | point | MTP | ungated | gate ctx 8K + queue 4 | gated / mixed steps |
      |---|---|---|---|---|
      | WebShop C=16 | 1919 / 46 ms / 7.76 / 29.5 | 1723 / 61 / 8.52 / 36.0 | 1786 / **43** / 8.48 / 43.4 | 1401 / 0 |
      | ShareGPT 16 req/s | 3091 / 95 / 6.21 / 30.9 | 3076 / 68 / 6.29 / 34.1 | 3087 / **52** / 6.28 / 33.5 | 1601 / 1 |
      | cohort C=16 agent | 107.8 / 42.9 s / 27.8 / 505 | 139.5 / 30.8 s / 17.6 / 91 | **139.2 / 30.9 s / 17.6 / 91** | 1 / 2651 |
      | cohort C=16 chat | 1228 / 51.2 s / 31.0 / 472 | 1542 / 27.0 s / 24.7 / 129 | **1549 / 27.1 s / 24.6 / 129** | |
      Matches or beats the ungated arm at every point: the overloaded cohort keeps the full gain (the
      queue clause keeps mixing on), ShareGPT keeps MTP throughput with TTFT -45%, WebShop's loss vs MTP
      shrinks from -10% to -7% throughput (ITL still +9%: short-prompt agent traffic is not fully
      recovered). **Default dispatch from here: mix iff prefill context >= 8192 or >= 4 requests wait.**
      **Output gate (15:27-15:31, `gateq_gate_20261016.sh`, 3 arms clean, graphs on):** MTP repeat vs MTP
      short 15 / long 39 (floor); split + EDF + preemption + gate short **37** / long 18, PASS (gate fired,
      FUSED_SPLIT_CALLS 1001). Short prompts now prefill alone as in MTP, so they agree longer than the
      floor; the long-prompt gap is the FA3-vs-FA2 bf16 difference already adjudicated by the same-input
      check above.

- [ ] **S4, EDF step model** re-fitted on the fused kernel and compared with the unfused schedule
  - [x] **Step model on the final schedule (PD multiplexing + EDF, uncached deadlines, divisor 30) vs the stock
    (unfused) schedules (2026-10-17 18:50-19:26, 6 traced arms rc=0, 0 tracebacks / 0 fallbacks):** STEPTRACE now
    covers pdmux's event loop (`apply_steptrace_pdmux.py`: decode and split-prefill forwards traced on their own
    streams, with the SM group `sg=` and split layers `lay=`). Lightweight linear models, fit on the first half of
    each trace, held-out MAPE on the second (`tasks/perf_model/fit_pdmux_step.py`):
    | what EDF predicts | pdmux + EDF | stock chunked LPT | stock MTP |
    |---|---|---|---|
    | decode interval a request sees (ITL), C=8 / 16 | **8.9% / 8.9%** (p90 15 / 21%) | all steps 117% / 74% | all steps 52% / 35% |
    | decode-only step GPU time, C=8 / 16 | **6.7% / 4.0%** | 17.5% / 7.5% | 66.8% / 32.3% |
    | prefill work (pdmux: one split forward; stock: mixed step), GPU, C=8 / 16 | 22.5% / 246% (p90 62 / 95%) | 56.7% / 52.5% | 43.4% / 46.8% |
    pdmux decode model: per SM group, c0 + c_bs*rows + c_kv*KV (21 coefficients); stock: fit_mtp_step.py's form
    (c0, prefill tokens, attention work, rows, KV). **With decode on its own SM group, the per-token interval is
    predictable to ~9%; on the unfused schedules prefill chunks interleave with decode and the same kind of model
    misses by 35-117%** -- the user's point-3 thesis (a schedule that is easy to model gives better SLO control)
    holds for ITL on this schedule, not for the fused-kernel step (whose co-run bound is 1.2%). TTFT-side prediction
    is the weak part: a split prefill shares the GPU with a varying decode load (p90 62-95%). The project fitter
    (`fit_mtp_step.py`, launch-to-launch target) gives >100% on the stock traces under the overlap scheduler, so the
    2026-10-14 10.1 / 10.7% figures (4-mode MTP server, other workload) are not comparable to these. Output:
    [`pdmux_step_model_20261017.txt`](perf_model/pdmux_step_model_20261017.txt); traces `REP=1 bash
    steptrace_pdmux_20261017.sh` (pod `/dev/shm/w`).
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
  - [x] **MuxWise baseline measured on the program-concurrency axis (Claude, 2026-10-17 08:51-09:40; H200 idle --
    Codex's owner PID992767 gone, lock free, noted in the coordination file).** Frozen 32-64K TraceLab cohort (48
    programs / 96 rounds) at agent C = 8 / 16 + natural ShareGPT 8 req/s, tree `up_edf_floor`, one session, one run per
    arm; all 8 arms rc=0, 0 tracebacks, 0 fallbacks; PDMUX server log `enable_pdmux: True`; ours FUSED_SPLIT_CALLS
    67 501 / 68 001; every agent arm logged `concurrency_limit` = C. Out tok/s / mean TTFT / mean ITL / P99 ITL, and
    tok/s/user p1 / p50 / p99 ([summary](debug_kernel_server_perf/baselines_20261017/pdmux_conc_20261017_summary.txt)):
    | C | arm | agent | chat | chat tok/s/user p1 / p50 / p99 |
    |---|---|---|---|---|
    | 8 | stock non-spec (LPT) | 123 / 13.2 s / 25.7 / 466 | 1 498 / 14.1 s / 26.3 / 465 | 7 / 52 / 156 |
    | 8 | **PD multiplexing (`--enable-pdmux`)** | 139 / 13.1 s / 15.7 / 39 | 1 612 / 11.3 s / **13.8 / 32** | **35 / 71** / 130 |
    | 8 | stock MTP | 110 / 18.8 s / 21.0 / 415 | 1 291 / 23.5 s / 30.3 / 476 | 8 / 42 / 162 |
    | 8 | ours (split + EDF 50 ms + floor 2048) | 139 / 14.7 s / 15.4 / 72 | 1 575 / **5.8 s** / 25.0 / 131 | 18 / 46 / 144 |
    | 16 | stock non-spec (LPT) | 127 / 29.8 s / 32.1 / 528 | 1 482 / 33.3 s / 24.6 / 456 | 7 / 59 / 155 |
    | 16 | **PD multiplexing** | 142 / 29.0 s / 16.9 / 41 | 1 609 / 25.3 s / **13.7 / 32** | **37 / 71** / 130 |
    | 16 | stock MTP | 107 / 43.4 s / 27.6 / 500 | 1 222 / 52.0 s / 31.3 / 472 | 8 / 40 / 158 |
    | 16 | ours | 139 / 31.8 s / 16.9 / 75 | 1 555 / **23.5 s** / 24.0 / 128 | 17 / 48 / 129 |
    **Measured negative result: PD multiplexing beats our final single-GPU config on ITL.** Ours vs PD multiplexing --
    C=8: throughput tie (agent 139 / 139, chat -2%), chat TTFT **-49%**, agent ITL tie, but chat ITL **+81%** (25.0 vs
    13.8 ms, P99 4x), agent TTFT +12%; C=16: agent / chat throughput -2% / -3%, chat TTFT -7%, agent TTFT +9%, agent
    ITL tie, chat ITL **+75%** (P99 4x). Its decode runs on its own SMs (GreenContext split) every step, so decode
    rows never wait behind a prefill chunk; ours mixes a 2048-token chunk into decode steps, and a mixed step costs the
    chunk plus the decode (measured on H20: decoders' ITL 12-34 -> 380-426 ms beside a cold 32K prefill). Its slow-user
    tail is far better too (chat p1 35-37 vs 17-18 tok/s/user). Against stock MTP ours still wins (C=16: agent
    throughput +30%, agent TTFT -27%, chat TTFT -55%, chat ITL -23%), but against this direct baseline the >=30% ITL
    bar is not met, and only chat TTFT clears it (C=8). PD multiplexing is non-speculative; the same SM split under
    MTP's verify rows is the open combination. **Tried directly (09:48-09:51, `pdmux_mtp_conc_20261017.sh`,
    `pdmux_mtp_eager_20261017.sh`): stock EAGLE3 chain MTP + `--enable-pdmux` does not run in this tree.** The only
    explicit rejection is for UNO tree mode, and the server does enable PD multiplexing (8 stream groups, prefill/decode
    SMs 132/0 ... 0/132), but (1) with CUDA graphs it dies capturing the EAGLE3 draft graphs (`AttributeError: 'NoneType'
    object has no attribute 'decode_wrappers'` -- the draft graph runner hard-codes `enable_pdmux = False` while the
    pdmux attention backends are per stream group), and (2) as a labelled eager diagnostic (`--disable-cuda-graph`) it
    reaches `event_loop_pdmux` and fails on the first draft step (`AssertionError` in
    `speculative/topk1.py:95 draft_topk1_postprocess`): pdmux's event loop runs split prefill batches and decode on
    separate stream groups, and EAGLE's draft/verify state assumes the standard loop. Combining them is scheduler
    integration work (drive the EAGLE worker's draft and verify from the decode stream group while prefill runs on the
    other), not a flag.
    - [ ] **PD multiplexing under EAGLE3 MTP -- scoped (2026-10-17, from `multiplex/multiplexing_mixin.py
      event_loop_pdmux`).** The loop runs the decode batch on the decode SM group with `run_batch`, advances a
      layer-split prefill (`split_index` / `split_forward_count`, a token budget's worth of layers per iteration) on
      the prefill SM group, and merges finished prefill requests into the running batch. Needed: (1) the EAGLE worker's
      `forward_batch_generation` must honour the split for extend batches (it runs the whole target forward today);
      (2) EAGLE3 reads hidden states from 3 target layers, so the split prefill must keep those across iterations and run
      the draft extend when the last layer finishes; (3) `merge_batch` must carry `EagleDraftInput` -- the eager
      `topk1.py` assert is consistent with a merged batch lacking draft state; (4) draft CUDA graphs per stream group
      (or eager draft, at a measured cost). Gates: srv_correctness against the stock-LPT floor, then the same cohort A/B
      vs `--enable-pdmux` and vs ours. This is where the SM-pre-assignment idea meets decode ITL: the measured gap to
      PD multiplexing is chat ITL +75-81%.
      **Pieces (1)-(3) implemented (2026-10-17, eager path; not yet measured):**
      `tasks/debug_kernel_server_perf/pdmux_mtp_20261017/apply_pdmux_mtp.py` on tree `up_pdmux_mtp` (copy of
      `up_edf_floor`): `TpModelWorker.forward_batch_split_prefill` takes a capture mode; `llama.forward_split_prefill`
      keeps the EAGLE3 aux hidden states on the ForwardBatch across splits and hands them to the logits processor at
      the last layer; `EAGLEWorkerV2.forward_batch_split_prefill` runs the target slice with `CaptureHiddenMode.FULL`
      and, after the last slice, `_draft_extend_for_prefill` exactly as `_forward_prefill_batch` does; the scheduler's
      pdmux branch routes split prefills through it when speculation is on. Compiles. Gate running:
      `smoke_pdmux_mtp.sh` (stock MTP vs MTP + pdmux, both eager, 24 concurrent requests incl. 8 x 12-24K prompts,
      greedy 128 tokens, per-request common prefix).
      **Eager path works end to end (2026-10-17 11:17, six fix rounds, all in `apply_pdmux_mtp.py`):** beyond (1)-(3) it
      needed (4b) the draft extend to run as a normal EXTEND (it inherited SPLIT_PREFILL and was routed to the draft
      model's split forward), (4c) the draft extend's `EagleDraftInput` installed on the batch before it joins decode,
      (5) the eager pdmux per-stream decode-backend override restricted to the target runner (the draft decode runs on
      the multi-step draft backends `draft()` planned; the override left `forward_metadata` None), (6) the same EAGLE3
      aux capture in `qwen2.forward_split_prefill` (the target is Qwen2), (7) target verify planned on and run under
      the per-stream decode backend (as an extend it re-planned the split prefill's `attn_backend` between slices:
      "q.shape[0] (19885) does not match qo_indptr[-1] (12)"), (8) `input_ids` restored from the split ForwardBatch for
      the draft extend. **Smoke gate (`bash /dev/shm/w/smoke_pdmux_mtp.sh`): MTP + pdmux served all 24 requests, 0
      tracebacks; 20 / 24 identical to stock MTP over 128 greedy tokens (median common prefix 128, worst 8);
      acceptance length 3.03-3.13, 17 split prefill batches beside decode.** The worst-case prefix is within the bf16
      batching noise measured before (split vs stock min common prefix 5 / 15) but this is not the formal gate:
      `srv_correctness.py` against the stock-LPT floor comes next, then (4) CUDA graphs, then the cohort A/B.
      **CUDA graphs work too (11:3x, `smoke_pdmux_mtp_graphs.sh`):** with graphs on in both servers MTP + pdmux served
      24 / 24, 0 tracebacks, decode on graphs (`cuda graph: True`); 18 / 24 identical to stock MTP over 128 tokens --
      five long prompts diverge at 48-99 tokens, one short at token 2 -- and acceptance length 3.07 vs 3.31 (-7%). Under
      greedy speculation the output is fixed by the target's verify, so divergence is target numerics (pdmux decodes on
      a different SM count, so flashinfer's split-KV partition differs); the acceptance drop is the open question.
      **Formal gate (`bash /dev/shm/w/gate_pdmux_mtp.sh`, srv_correctness, graphs on; the driver gained a PDMUX arm):
      MTP_PDMUX PASS by the rule** (long prompts agree on their first 8 tokens; 0 tracebacks, 0 fallbacks), but far
      from the floor: stock MTP repeat vs MTP is fully deterministic (min common prefix short 1 200 / long 96), MTP_PDMUX
      short 1 / long 11. Per request: every first token matches; 6 / 8 long prompts identical over 96 tokens (l1, l5
      diverge at 18 / 11); the 1 200-token background rows -- which decode on the decode SM group while pdmux resizes it
      as prefills come and go, changing the kernels' partition -- diverge at 1-723, one identical; several identical
      outputs show different acceptance, i.e. the draft saw differently rounded inputs. **Null case (`bash
      /dev/shm/w/gate_pdmux_nospec.sh`, stock LPT vs LPT + `--enable-pdmux`, no speculation, paths the patch does not
      touch): min common prefix short 1 / long 11 -- exactly MTP_PDMUX's.** The divergence is pdmux's own numerics (its
      decode runs on a resized SM group); the MTP integration adds none. **Gate cleared against the right floor.**
      **Payoff A/B, run 1 (11:33-11:58): MTP_PDMUX crashed at both C, two different bugs.** (a) OOM at C=8 and C=16
      (1.8 GiB MLP activation, 1.57 GiB free): pdmux prefills unchunked, and a whole 64K agent prompt does not fit
      beside the draft model at mem fraction 0.85; the arm now runs at 0.78 (KV pool ~1.6M tokens vs the cohort's
      <= 16 x 64K working set -- a handicap to our arm, not an advantage). (b) Rerun at 0.78: C=8 clean, C=16 **illegal
      address in the decode step** (47 running, 9 queued). Cause: the draft extend after a split prefill runs on the
      prefill stream while the decode step runs on the decode stream, and EAGLE had replaced the draft runner's pdmux
      prefill backend (private flashinfer workspace) with `draft_extend_attn_backend`, which shares the global
      workspace and eager kv-index buffers with the decode-stream backends. Fix (patch item 8): keep the pdmux
      prefill backend and run the prefill-side draft extend on it. **Output gate re-run with the fix: PASS, short 1 /
      long 11 -- unchanged, equal to the non-speculative pdmux null case** (`pdmux_mtp_fix8_20261017.sh`; the
      pre-fix C=8 run is discarded, since it ran with the race). PDMUX and EDFR50_F2048 from run 1 stand (0 tracebacks,
      0 fallbacks, FUSED_SPLIT_CALLS 67.5K / 68.5K); MTP_PDMUX rerun with the fix at C=8 / 16 (12:25-12:39): rc=0,
      0 tracebacks, 0 fallbacks, pdmux active.
      **Result: MTP + pdmux does not pay -- negative.** Agent / chat: tput, mean TTFT (s), mean ITL (P99) ms;
      chat tok/s/user p1/p50/p99 (`summarize_ab.py`, [summary](debug_kernel_server_perf/pdmux_mtp_20261017/pdmux_mtp_ab_20261017_summary.txt)):
      | C | arm | agent | chat | chat tok/s/user |
      |---|---|---|---|---|
      | 8 | PDMUX | 141 / 12.9 / 15.5 (37) | 1612 / 10.7 / 13.8 (33) | 36 / 68 / 132 |
      | 8 | MTP_PDMUX | 107 / 20.7 / 17.8 (99) | 1267 / 25.2 / 27.0 (116) | 16 / 40 / 113 |
      | 8 | EDFR50_F2048 (ours) | 139 / 14.8 / 14.3 (48) | 1576 / 5.8 / 25.1 (131) | 18 / 45 / 138 |
      | 16 | PDMUX | 142 / 29.0 / 16.9 (41) | 1610 / 25.3 / 13.7 (32) | 36 / 71 / 130 |
      | 16 | MTP_PDMUX | 106 / 48.0 / 18.1 (99) | 1213 / 55.0 / 28.2 (111) | 17 / 38 / 95 |
      | 16 | EDFR50_F2048 (ours) | 139 / 31.6 / 16.8 (73) | 1558 / 23.3 / 23.9 (127) | 16 / 48 / 126 |
      MTP_PDMUX vs PDMUX: throughput -24%, TTFT +61..+117%, chat ITL 2x. Acceptance is the same in both MTP arms
      (bench 2.12 tokens per verify for MTP_PDMUX and ours), so the smoke test's 3.07 vs 3.31 was small-sample noise,
      not a pdmux effect -- closed. The cost is the step: a 4-token verify plus 3 draft steps over 32-64K-context
      rows, on pdmux's shrunken decode SM group, takes ~4x a non-speculative decode step for 2.1x the tokens.
      Plain pdmux stays the strongest ITL baseline; ours keeps TTFT (chat -46% / -8% vs PDMUX) and ties throughput.
      - [x] **PD multiplexing + EDF (2026-10-17 14:32-15:42, 12 arms rc=0, 0 tracebacks / 0 fallbacks, EDF active in the
        EDF arms (EDF_RUNTIME, priority scheduling on), pdmux on in every pdmux arm; output gate queued):** pdmux keeps decode on its own SM group, which is where its
        ITL comes from; our EDF wins chat TTFT by deadline-ordered admission, which is independent of chunking.
        pdmux's batch builder goes through `get_new_batch_prefill`, so worker-side EDF (deadline order, no latency
        model, no chunk cap; server-assigned deadlines) applies unchanged. Arms PDMUX | PDMUX_EDF | stock MTP (the
        bar's baseline), C = 8 / 16, two repeats, same cohort / tree / session. `pdmux_edf_20261017.sh` (REP=1, 2).
        Two-repeat means (rep 1 ; rep 2 in [summary](debug_kernel_server_perf/pdmux_mtp_20261017/pdmux_edf_20261017_summary.txt)):
        tput / mean TTFT (s) / mean ITL (P99) ms, chat tok/s/user p1 / p50 / p99:
        | C | arm | agent | chat | chat tok/s/user |
        |---|---|---|---|---|
        | 8 | stock MTP | 110.5 / 18.69 / 21.3 (423) | 1285 / 23.19 / 30.5 (478) | 7 / 42 / 168 |
        | 8 | PDMUX | 140.9 / 12.91 / 15.4 (38) | 1613 / 10.67 / 13.7 (33) | 38 / 69 / 133 |
        | 8 | **PDMUX + EDF** | 143.0 / 12.97 / 15.2 (36) | 1611 / **3.80** / 14.0 (33) | 33 / 68 / 132 |
        | 16 | stock MTP | 107.7 / 43.14 / 27.2 (495) | 1223 / 51.36 / 31.1 (471) | 7 / 41 / 162 |
        | 16 | PDMUX | 142.5 / 28.87 / 17.0 (41) | 1609 / 25.06 / 13.7 (32) | 37 / 72 / 130 |
        | 16 | **PDMUX + EDF** | 142.2 / 29.37 / 16.1 (38) | 1610 / **17.45** / 14.0 (34) | 34 / 69 / 131 |
        Repeat spread <=2% on every mean (chat TTFT 1.3%). **PDMUX + EDF vs stock MTP: C=8 agent tput +29.4%, TTFT
        -30.6%, ITL -28.6%; chat tput +25.4%, TTFT -83.6%, ITL -54.2%. C=16 agent +32.1% / -31.9% / -40.8%; chat
        +31.6% / -66.0% / -54.9%.** 11 of 12 metrics clear the bar (>=30% TTFT and ITL, >=10% throughput); the miss is
        agent ITL at C=8 (-28.6%). EDF adds to pdmux exactly what it was meant to: chat TTFT -64% (C=8) / -30% (C=16)
        with ITL and throughput unchanged. Against stock *non-speculative* LPT (2026-10-17 baseline run, same cohort,
        other session) agent TTFT does not move (12.97 vs 13.20 s at C=8): cold 32-64K agent prefills are compute-
        bound and EDF serves chat's earlier deadlines first. Agent-ITL lever next: pdmux's decode SM share
        (`decode_bs_divisor` 36 -> 18 / 12, `pdmux_edf_div_20261017.sh`), after the output gate (`gate_pdmux_edf.sh`).
        sglang `29842f52f`+ (tree `up_pdmux_mtp`).
        - [x] **Output gate (2026-10-17 15:4x, `gate_pdmux_edf.sh`): PASS.** LPT_PDMUX_EDF vs LPT: min common prefix
          short 1 / long 11, identical to the null case LPT_PDMUX vs LPT (pdmux's own numerics: decode on a resized SM
          group); stock repeat floor short 163 / long 96; 0 tracebacks / 0 fallbacks, priority scheduling and pdmux on.
          EDF changes order, not numerics.
        - [x] **Decode SM share (2026-10-17 15:48-16:10, rep 1, all rc=0):** pdmux gives decode `decode_bs * 6 //
          decode_bs_divisor` of its 8 groups (default 36). vs stock MTP, agent tput / TTFT / ITL: C=8 divisor 36
          +29% / -31% / -29%, 18 +25% / -24% / -37%, 12 +22% / -20% / -39%; C=16 36 +32% / -32% / -41%, 18 +27% / -26%
          / -45%, 12 +28% / -26% / -48%. More decode SMs buy agent ITL with agent TTFT -- a trade-off curve, not a free
          lever; at C=8 no divisor clears both. Default 36 stays. Chat keeps -59..-84% TTFT and -55..-58% ITL
          throughout ([summary](debug_kernel_server_perf/pdmux_mtp_20261017/pdmux_edf_div_20261017_summary.txt),
          `pdmux_edf_div_20261017.sh`). Next: the two-tier TTFT SLO (agent prompts >= 16K get a flat SLO instead of a
          length-scaled deadline that queues them behind all chat), `pdmux_edf_t2_20261017.sh`.
        - [x] **Two-tier TTFT SLO (2026-10-17 16:10-16:44, rep 1, 6 arms rc=0, EDF + pdmux on, 0 tracebacks / 0
          fallbacks): negative.** `SGLANG_EDF_LONG_PROMPT_TOKENS=16384` with a flat 10 s / 5 s agent SLO, vs stock MTP:
          C=8 10 s (divisor 36) agent +28% / -30% / -28%, chat TTFT -80%; 10 s + divisor 18 agent -23% TTFT / -35% ITL;
          5 s + divisor 18 agent -25% / -32%, chat TTFT only -58%. C=16 10 s agent +31% / -32% / -39%. Agent TTFT
          does not move (13.09 vs 12.97 s at C=8) while chat TTFT degrades, so agent TTFT is not chat queueing: it is
          agent-vs-agent queueing on compute-bound cold 32-64K prefills. Length-scaled deadlines stay.
          ([summary](debug_kernel_server_perf/pdmux_mtp_20261017/pdmux_edf_t2_20261017_summary.txt))
        **Where the bar stands for PD multiplexing + EDF (default divisor, length-scaled SLO), vs stock MTP:** 11 of 12
        metrics clear at C=8 and C=16; the one miss is C=8 agent ITL (-28.6%, two repeats), and every knob that buys it
        (decode SM share, agent SLO tier) costs agent TTFT below -30% -- the C=8 agent pair sits on a trade-off frontier
        at about (-30%, -29%).
        - [ ] **TTFT deadline on uncached tokens (2026-10-17, `apply_edf_uncached.py`, `SGLANG_EDF_UNCACHED_DEADLINE=1`):**
          the length-scaled deadline used the full prompt, so a follow-up agent turn whose 32-64K context is already in
          the radix tree got a cold prompt's late deadline. The cached length is now read at enqueue by a read-only
          match and the deadline scales with the uncached remainder. First arm (C=8, divisor 36, rep 1, rc=0, EDF + pdmux
          on, 0 tracebacks / 0 fallbacks) vs stock MTP: agent tput +29.8%, **TTFT -34.1%** (12.31 vs 12.97 s with
          full-length deadlines), ITL -29.1%; chat +25.4% / -74.0% / -54.4%. The TTFT margin it buys is what the decode
          SM share can spend on agent ITL. **Divisor sweep (16:51-17:30, rep 1, 8 arms rc=0, 0 tracebacks / 0
          fallbacks, EDF + pdmux on)**, vs stock MTP, agent tput / TTFT / ITL ; chat:
          | C | divisor | agent | chat |
          |---|---|---|---|
          | 8 | 36 | +29.8% / -34.1% / -29.1% | +25.4% / -74.0% / -54.4% |
          | 8 | **27** | **+28.3% / -31.1% / -31.9%** | +25.5% / -71.1% / -55.3% |
          | 8 | 24 | +26.8% / -27.5% / -33.8% | +25.5% / -69.2% / -55.7% |
          | 8 | 18 | +24.3% / -23.8% / -36.2% | +25.5% / -66.1% / -57.3% |
          | 16 | 36 | +34.8% / -34.3% / -38.6% | +31.9% / -59.4% / -54.6% |
          | 16 | **27** | **+32.1% / -31.1% / -41.5%** | +31.6% / -55.8% / -56.2% |
          | 16 | 24 | +32.2% / -31.2% / -43.0% | +31.5% / -56.1% / -55.9% |
          | 16 | 18 | +30.2% / -28.6% / -45.6% | +31.9% / -52.6% / -56.5% |
          **Divisor 27, three repeats (17:35-17:58, all rc=0, 0 tracebacks / 0 fallbacks) + output gate PASS**
          (LPT_PDMUX_EDF with uncached deadlines and divisor 27 vs LPT: short 1 / long 11, identical to the pdmux null
          case; stock repeat floor short 132 / long 96). Means vs stock MTP (per-rep tput / TTFT s / ITL ms in
          [3-rep summary](debug_kernel_server_perf/pdmux_mtp_20261017/pdmux_unc27_3rep_summary.txt)):
          C=8 agent +27.8% / -30.1% / -31.6%, chat +25.4% / -70.8% / -55.3%; C=16 agent +32.7% / -32.2% / -40.2%, chat
          +31.6% / -57.1% / -54.7%; repeat spread <=3.4% (C=8), <=5.2% (C=16). **All 12 means clear the bar, but C=8
          agent TTFT by 0.1 point (per rep -31.1 / -29.4 / -29.7%) -- not robust.** Divisor 30 (between 36's -34.1 /
          -29.1 and 27's -30.1 / -31.6) for three repeats next, `pdmux_edf_unc2_20261017.sh` with UNC_D30.
        - [x] **Final single-GPU config -- PD multiplexing + EDF with uncached-token deadlines, decode_bs_divisor 30
          (2026-10-17 18:05-18:50; three repeats, all rc=0, 0 tracebacks / 0 fallbacks, EDF + pdmux on; output gate
          PASS -- LPT_PDMUX_EDF vs LPT short 1 / long 11, identical to the pdmux null case, stock repeat floor short 545
          / long 90).** Three-repeat means vs stock MTP (same cohort, ShareGPT 8 req/s, frozen 32-64K TraceLab):
          | C | agent tput / TTFT / ITL | chat tput / TTFT / ITL |
          |---|---|---|
          | 8 | **+28.9% / -31.7% / -31.1%** | +25.3% / -72.8% / -54.9% |
          | 16 | +32.7% / -32.3% / -40.4% | +31.6% / -57.3% / -54.7% |
          **All 12 means clear the bar (>=30% TTFT and ITL, >=10% throughput).** Per repeat, C=8 agent TTFT / ITL:
          -30.4 / -31.9, -33.0 / -31.9, -31.6 / -29.6 (5 of 6 values clear; rep 3 ITL is 0.4 point short); C=16 all
          clear in every repeat. Chat tok/s/user p1/p50/p99 34 / 69 / 132 (C=8) vs stock MTP 7 / 42 / 168; agent P99
          ITL 34-35 ms vs 417-498 ms. **Rep 4 (2026-10-17 22:45-22:57, rc=0, 0 tracebacks / 0 fallbacks): C=8 agent
          141 / 12.95 s / 14.6 ms (-30.7% / -31.5% vs stock MTP); four-repeat C=8 agent means -31.4% TTFT / -31.2% ITL,
          7 of 8 per-repeat values clear; C=16 clears in every repeat.** Per-rep tput / TTFT / ITL in
          [summary](debug_kernel_server_perf/pdmux_mtp_20261017/pdmux_unc30_3rep_summary.txt). Commands (pod
          `/dev/shm/w`, tree `up_pdmux_mtp` = `up_edf_floor` + `apply_pdmux_mtp.py` + `apply_edf_uncached.py`):
          `for r in 1 2 3; do REP=$r D27=0 D24=0 D30=1 bash pdmux_edf_unc2_20261017.sh; done`, `bash
          gate_pdmux_edf_unc30.sh`. **Attribution, stated plainly:** the SM split that buys ITL is sglang's PD
          multiplexing; ours is the scheduling on top -- deadline-ordered admission with uncached-token deadlines and
          the decode-share setting -- which turns pdmux's 9-of-12 into 12-of-12 against stock MTP. Per-request spread in
          [summary](debug_kernel_server_perf/pdmux_mtp_20261017/pdmux_edf_unc_20261017_summary.txt). Script:
      `pdmux_mtp_ab_20261017.sh` (frozen cohort, C = 8 / 16 + ShareGPT 8 req/s; PDMUX | MTP_PDMUX |
      EDFR50_F2048, one tree, one session, graphs on). Command: `bash /dev/shm/w/pdmux_conc_20261017.sh`
    (`tasks/debug_kernel_server_perf/baselines_20261017/pdmux_conc_20261017.sh`).
  - [ ] **MuxWise = sglang `--enable-pdmux` (2026-10-13; superseded by the 2026-10-17 run above):** no public MuxWise repo; upstream
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
