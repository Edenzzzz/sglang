# Changelog

## v0.5.1 — Task-doc cleanup: invalidated results removed, story-order restructure (2026-08-13)

- `tasks/debug_kernel_server_perf.md`: deleted the sections marked superseded/
  invalidated (2026-07-24 serving results; Historical H800 WL1–6/EDF tables that
  compared contaminated DMC baselines). Reordered: current story (sync-free plan +
  gate + cost model + conclusion) and Open TODOs first; honest H100-era kernel/server
  results and closed investigations behind them as labeled history. Open TODOs
  rebuilt: EDF-analytic A/B marked done (multiturn_slo_clean), remaining suite
  scenarios added.
- `tasks/realistic_workload_analysis.md`: removed the superseded headline table,
  contaminated offered-load sweep, and pre-fix sweet-spot table; rewrote Scope
  (gate + LPT_EDF methods) and Final conclusions to the current story; added an
  evaluation-hygiene section (baseline contamination, warmup, queue-vs-service TTFT
  decomposition, rate=inf confound, episode defense) and mixed-workload repro
  commands.
- `tasks/edf_scheduling.md`: validity note (pre-fix DMC arms; policy conclusions
  stand, kernel deltas do not); task #117 closed by the measured gate.
- `tasks/server_perf_model.md`: status note — analytic model supersedes the MLP for
  chunk control; MLP remains for the router `/get_load` signal only.
- `tasks/README.md`: updated to the current document map.
- Profiling note added to the overhead diagnosis, mined from the mixed-tenancy nsys
  captures: `non_blocking=True` is not honored for D2H into pageable memory, and the
  cost is a pipeline drain rather than a copy. Within back-to-back groups of pageable
  D2H copies the **first** costs 5–8 ms (median) and the rest ~14 µs — 560–800× apart
  at identical size and destination — so the old plan() stall was paid by the first
  `.to("cpu")`, not by the following `torch.cuda.synchronize()`. Steady-state window
  analysis shows the host blocked 74.6% of wall in both arms while the GPU stays ~94%
  busy (GPU-idle-while-host-blocked 0.64% of wall), so such readbacks hurt only when
  they remove run-ahead the GPU needs. Two live optimizations filed: flashinfer's stock
  `plan()` still does pageable D2H for the BatchPrefill path (baseline pays it every
  mixed step), and sglang's `fast_decode_plan` remains shadowed by the class-level
  `begin_forward = plan` alias.

## v0.5.0 — Mixed-Tenancy Win, Overlap Attribution, Analytic Cost Model (2026-08-04 → 2026-08-06)

### Mixed-tenancy workloads (`run_mixed_workloads.py`)
- New multi-tenant suite (coding agent + chat on one node, plus long-CoT / RAG / control
  scenarios). `agent_plus_shortchat` vs honest BatchPrefill baseline: coding tenant
  **+36.5% throughput**, chat tenant **−42% mean ITL / −23% P99 / −37% TTFT**;
  `shortchat_only` control at parity with the gate declining persistent (6/2,566).
- One −19% LPT trial archived as `run1_regressed/` (storm-phase retraction-cascade
  signature); unreproduced across repeat + nsys trials with a stable baseline.

### Gate v2 (KV concentration + out-of-support guard)
- Root cause of the win region: few long-context decode rows starve BatchPrefill of
  query tiles (8×64K = 8.2x vs 128×4K = 1.02x at equal aggregate KV); one chunk flips
  its CTA tile and exhausts the KV-split budget (3.125 ms plateau for any chunk ≥1K).
- Two k-NN blind spots fixed (fresh-prefix chunks; blended pools) via 837 new sampled
  compositions and a KV-concentration feature (Σkv²/Σkv). Merged table 1,316 points,
  k=5 unanimous, label ≥1.10: leave-one-out worst regression 1.000, capture 81%.

### Overlap attribution + analytic cost model
- Same-kernel ablation: temporal PD overlap contributes up to **1.26×** in the balanced
  band (T_p ≈ T_d); heterogeneity-robust scheduling supplies the rest.
- `SGLANG_EDF_LATENCY_MODEL=analytic`: `T_attn = max + 0.53·min` (3.1% MAPE / 9% max,
  7 physical constants, no training) drops into the EDF chunk controller; BP's best
  parts-based fit is 13.3% / 43% with fitted β=2.19 (super-additive) — the fused kernel
  is what makes mixed-step cost predictable enough for SLO-aware chunking.

### Docs
- Superseded/contaminated 2026-07-24 serving tables flagged in place; conclusions and
  TODOs consolidated to current results.

## v0.4.0 — Sync-Free Persistent Plan + Per-Batch Mixed Kernel Gate (2026-07-30)

### Validation (TraceLab decode-heavy N=100 screens + WL1 Qwen3-8B, honest baselines)
- 10 sessions (old +14.5% pooled-P99 regression point): parity on every axis across paired
  runs at the matched cap; pooled P99 ITL −0.7% to −6.2% better than true DMC.
- 16 sessions: +0.8% mean throughput across paired runs, pooled P99 ITL −1.3% to −4.1%,
  request-max P99 up to −23%; cap 6656 option reaches −14.6% pooled P99 ITL / +0.77% tput.
- WL1 (Qwen3-8B, rates 8 and 2): parity or slightly better on all metrics; 100%
  retokenization on every accepted run.
- Startup prewarm of both persistent module variants added after a mid-replay JIT build
  froze all nine sessions for 18.3 s in the first g10 run (flashinfer `c6d2a00`,
  sglang `313b97611`).

### Sync-free BatchAttention planning (flashinfer `f4d2a80` + backend host metadata)
- `BatchAttention.plan()` no longer calls `torch.cuda.synchronize()`. The backend now
  builds `qo_indptr`/`kv_indptr`/`kv_len` on the host from `seq_lens_cpu`/`extend_prefix_lens_cpu`
  (NumPy cumsum) and passes CPU tensors straight to `plan()`; flashinfer double-buffers its
  pinned int-workspace staging with CUDA-event fencing so removing the sync is race-free.
  Measured: with ~40ms of queued GPU work, plan wall time drops 56.5ms → 0.47ms. This was the
  main serving overhead of the persistent kernel (it broke overlap-scheduler CPU run-ahead on
  every mixed step); previous naive removal attempts hung because of the pinned-buffer race.
- `persistent_attn_forward` reuses grown `out`/`lse` buffers instead of allocating two tensors
  per layer per step.
- Removed a dead `call_begin_forward` branch that would have routed BatchPrefill wrappers
  into a BatchAttention-signature `plan()` call when `use_persistent=1`.

### Per-batch mixed-attention kernel gate (scaffolding)
- New env `SGLANG_MIXED_ATTENTION_KERNEL` = `auto` (default) / `persistent` / `batch_prefill`.
  For each mixed batch, `select_mixed_attention_kernel()` chooses persistent BatchAttention or
  BatchPrefill from CPU-side batch composition; forward dispatch now keys on the planned
  wrapper type instead of the global env flag. Cost-model constants pending the H800 gate
  sweep (see tasks/debug_kernel_server_perf.md).

### DMC baselines were silently persistent
- `SGLANG_FLASHINFER_USE_PERSISTENT` defaults to 1 and no DMC launcher overrode it, so every
  H800 "DMC_CG" baseline since 2025-11-14 ran the persistent kernel for mixed batches. July
  TraceLab LPT-vs-DMC deltas compared chunk caps/EDF guards on the same kernel. Both
  `run_gain_search.sh` and `run_edf_sweep.sh` now set `SGLANG_FLASHINFER_USE_PERSISTENT=0`
  for baseline methods.

### H800 kernel measurements (serving-matched non-causal prefix composition, Qwen2.5-7B heads)
- Persistent kernel run() CPU dispatch is cheaper than BatchPrefill (16µs vs 26µs) —
  cooperative launch is not a CPU-side cost.
- 479-case sweep: persistent wins up to 20× at decode-dominant compositions but loses up to
  0.55× when decode work is tiny and the chunk is compute-bound; 87/186 raw losses are
  recovered by choosing the right flipped schedule (big chunks + long prefix want non-flipped).

## v0.3.3 — Nebius-SWE Workload + Chunk-Control Fall-back Fix (2026-04-09)

### Heavier Benchmark Workload
- Added `nebius-swe-traj` and `swe-bench-pro` datasets to `bench_serving_3rdparty.py` and the bench_serving CLI choices.
- `nebius-swe-traj` loads `nebius/SWE-agent-trajectories` (multi-turn SWE agent traces) via streaming, samples a random AI-turn boundary per trajectory. Median input ~9.7K tokens — the only prefill-heavy real-world dataset surveyed in `tasks/realistic_workload_analysis.md`. Code-Feedback (200-token median input) was too lightweight for chunk control to matter; nebius is the right stress workload.
- `swe-bench-pro` loads `ScaleAI/SWE-bench_Pro` test split — 731 single-step Q/A pairs, longer than swe-gym but still single-shot, kept available but not the headline workload.

### Chunk-Control Fall-back Fix (`edf_scheduler.py`)
- **Bug**: when any decode req's slack went negative (`min_slack_ms ≤ 0`), `edf_compute_max_chunk_size` fell back to the **full** `default_chunk_size` (8192). At saturation this triggered immediately, so chunk control was effectively disabled — first nebius EDF run gave P99 ITL = 297ms vs 297ms baseline (zero improvement).
- **Fix**: throttle to `MIN_CHUNK = 512` instead of falling back to default. Already-violated requests can't be saved by relaxing the chunk; the not-yet-violated ones can be, by capping added prefill latency to ~10-20ms.
- **Impact** on nebius-swe-traj 1k sweep: DMC/LPT_CG P99 ITL drops from ~290ms → ~135ms across all rates; per-req P99 ITL ≤ 150ms attainment goes from 0.6-22% to **85-100%**. Throughput rises 1212→1290 tok/s (faster KV drainage = higher effective concurrency).

### New SLO Metric (`analyze_results.py`)
- Added `itl_p99_attain_pct`: per-request P99 ITL ≤ ITL_SLO (Sarathi-Serve-style). Strict max-ITL gives ≈0% on long-decode workloads (any single outlier kills the score, masking real differences); per-req P99 filters single-step noise but still penalizes consistent prefill-blocking. TPOT (mean ITL) is too lenient (>95% on every run).
- Added `nebius_swe` to the per-workload SLO table (5s TTFT / 150ms ITL — same as SWE-Gym).

### New Sweep Scripts + Pareto Plot
- `tasks/edf_benchmark/run_nebius_baseline_1k.sh` and `run_nebius_edf_1k.sh`: 1000 prompts, rates 1/2/4/8/12/inf (lower than other workloads because 9K-token prefills are 30× heavier than ShareGPT), output fixed to 1024 tok via `--sharegpt-output-len`. Both scripts toggle `gpu_keepalive` per benchmark per project rules.
- `tasks/edf_benchmark/plot_pareto_nebius.py`: side-by-side baseline median ITL + EDF SLO attainment plot. Output: `tasks/edf_benchmark/pareto_nebius.png`.

### Findings
- Chunk control eliminates the historical PP advantage on decode tail latency on prefill-heavy workloads. DMC/LPT_CG now match PP at 91-100% vs 95-100% per-req P99 ITL attainment.
- TTFT remains the binding constraint above rate 2 — saturation queues are fundamental and no scheduling policy can hide them. EDF + chunk control protects the decode SLO at the cost of TTFT for late requests, which is the correct trade-off for interactive serving.

## v0.3.2 — 1k-prompt EDF Sweep + schedule_policy Admission Fix (2026-04-07)

### schedule_policy Bug Fix (overlap scheduler crash root cause)
- `add_chunked_req` and `add_one_req` could admit a chunked-prefill req with `extend_input_len ≤ 0` (full prefix hit on a freshly chunked req). In MIXED mode the sampler then produced N-1 sampled tokens for a batch of N, crashing the overlap scheduler with `RuntimeError: expanded size (N) must match existing size (N-1)`.
- Fix: when `extend_input_len ≤ 0`, drop one prefix block (`prefix_indices[:-1]`), recompute `extend_input_len`, and refuse admission if still ≤ 0. This is a real root-cause fix — no `--disable-overlap-schedule` workaround.
- Files: `python/sglang/srt/managers/schedule_policy.py` (`add_chunked_req`, `add_one_req`).

### 1000-Prompt EDF Sweep with 6 Pareto Curves
- Re-ran the full 3 methods × 3 workloads × 6 rates (8/16/24/32/48/inf) sweep at **1000 prompts** per workload, both baseline (FCFS) and EDF + SLO-aware chunk control.
- Baseline tables now report median TTFT and median ITL alongside P99 (per user request).
- Switched the EDF Pareto y-axis from TPOT (mean ITL ≤ SLO) to **strict max-ITL ≤ SLO**: TPOT stays > 99% across every run because median ITL is 25-30ms vs 100-150ms SLO, masking real differentiation. Strict max-ITL is what chunk control actually protects.
- 6 Pareto curves: `tasks/edf_benchmark/pareto_baseline_median_itl.png`, `tasks/edf_benchmark/pareto_edf_slo_attainment.png`.
- Key result: EDF + chunk control gives 2-10× tail-ITL SLO attainment improvement for DMC/LPT_CG vs PP at saturation on ShareGPT and SWE-Gym.
- Scripts: `tasks/edf_benchmark/run_baseline_1k.sh`, `run_edf_1k.sh`, `analyze_results.py`, `plot_pareto_6.py`.
- Results: `tasks/edf_benchmark/results_baseline_1k/`, `results_edf_1k/`.

### Project Rule
- Added to `CLAUDE.md`: **Never compromise by falling back silently** — if a feature crashes, find the root cause; don't disable it to work around the problem.

## v0.3.1 — EDF Benchmark Sweep + Chunk Control Fix (2026-04-07)

### EDF Chunk Control Bug Fix
- `edf_compute_max_chunk_size` was using `req.last_tic` as previous decode time, but `last_tic` is only updated when `enable_metrics=True`. With metrics off it equals request creation time, slack went negative, prefills starved (smoke test: 579 vs 2135 tok/s).
- Fix: track `_edf_last_step_time` per req, updated in `process_batch_result_decode`. New reqs fall back to `now` (full ITL_SLO budget).

### Full EDF Benchmark Sweep (3 methods × 3 workloads × 5 rates)
- Ran PP / DMC / LPT_CG with EDF priority + SLO-aware chunk control on ShareGPT, Code-Feedback, SWE-Gym at rates 4/8/12/16/24
- Per-workload SLOs: ShareGPT (2s/100ms), Code-Feedback (3s/150ms), SWE-Gym (5s/150ms)
- TTFT attainment is 100% across all 45 runs — ITL is the binding SLO
- PP dominates Code-Feedback (100% at rate ≥ 12); mixed batches (DMC/LPT_CG) win on ShareGPT at saturation
- Generated two Pareto curves: throughput vs ITL P99, and SLO attainment vs request rate
- Scripts: `tasks/edf_benchmark/run_edf_sweep.sh`, `analyze_results.py`, `plot_pareto.py`
- Results in `tasks/edf_benchmark/results_edf/`

## v0.3.0 — EDF Scheduling + SLO-Aware Chunk Control (2026-04-06)

### EDF Scheduling
- Implemented Earliest Deadline First scheduling: priority = TTFT deadline (arrival + SLO)
- TTFT SLO scales with input length (base + extra for >4K tokens)
- Enable via `SGLANG_EDF_SCHEDULING=1` + `--enable-priority-scheduling --schedule-low-priority-values-first`

### SLO-Aware Chunk Size Control
- Binary search for max prefill chunk size that fits within ITL SLO slack
- Uses trained MLP latency model to predict server step latency
- Configured via `SGLANG_EDF_LATENCY_MODEL=path/to/model.pt`
- Fixed clock mismatch: all EDF code uses `time.monotonic()` consistently

### Latency Prediction Models
- Retrained both LPT and DMC server-level latency models (25K+ configs each)
- LPT model: 2.4% MAPE, 3.26ms MAE (tasks/perf_model/lpt_model.pt)
- DMC model: 2.3% MAPE, 3.04ms MAE (tasks/perf_model/dmc_model.pt)

### Benchmark Status
- Created benchmark scripts for PP vs LPT_CG vs DMC with EDF on 3 workloads
- Blocked by pre-existing `req_to_token_pool memory leak` bug (available_size > total_size)

## v0.2.1 — CG25: data-driven mixed CG size optimization (2026-04-01)

### Mixed CG Size Optimization
- Instrumented mixed CG replay to collect (raw_tokens, padded_tokens) per workload
- Greedy selection of 6 additional capture sizes based on real padding waste: +3584 (28% of waste), +2816 (7%), +2304 (5%), +640, +1408, +1920
- CG25 (25 sizes, ~1.79GB) reduces padding from 11% to 6.5%, improves all 6 workloads +0.3-1.4% over CG19
- LPT+CG25 beats trueDMC on all workloads: +18.9% (WL1), +3.6-8.0% (WL2-4), +1.9-2.1% (WL5/6)
- Updated task MD: replaced v3 with LPT in tables, updated stale CG12-era results with CG25 data

## v0.2.0 — Mixed Batch CUDA Graphs + LPT + Server Benchmarks (2026-03-27 → 2026-03-29)

### Mixed Batch CUDA Graph Support
- Implemented mixed batch (prefill+decode) CUDA graph capture and replay for persistent kernel (BatchAttention)
- LPT (dynamic scheduler) compatible with CUDA graphs — activates at runtime when plan() detects mixed batch
- 6 bugs fixed: missing kv_indices copy, plan_info[-1] wrong field, module auto-switch breaking CG, padding overflow in replay, shared indptr buffer too small, LPT pointer initialization for CG

### Server Benchmarks (H100 80GB, Qwen3-8B)
- v3+LPT+CG beats prefill-prio by +41-87% throughput on long prefill (12k-16k input)
- HOL blocking crossover at ~8k input tokens for prefill-prio (max ITL 21-25s vs v3's 230ms)
- v3 matches default-mc within ±0.1% — plan() overhead offsets CG savings

### LPT Load-Balance Profiling
- Grid.sync() wait reduction: -15% to -39% max wait on KV-splitting workloads
- Kernel speedup: 7-16% on mixed workloads

### Overlap Profiler
- Fixed shape mismatch for persistent v3 compatibility (disable CG during profiling)

## v0.1.0 — Persistent Kernel v3 + Decode Fix (2026-03-26)

### Decode Overhead Fix
- Decode steps use regular BatchDecodeWithPagedKVCacheWrapper instead of BatchAttention
- Eliminated 0.5-4% throughput regression vs default-mc
