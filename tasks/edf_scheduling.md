# EDF (Earliest Deadline First) Scheduling for SLO-Aware Serving

## Goal
Implement deadline-aware scheduling in sglang to maximize SLO attainment under load. Compare PP vs LPT+CG vs DMC on real agent workloads with realistic SLOs and request rates.

## Design

### 1. SLO Definition

Per-request SLOs based on literature:
- **TTFT SLO**: Base 2s + length-dependent scaling (longer inputs get more time)
- **ITL SLO**: Workload-specific (see Tempo paper, DistServe, Sarathi-Serve)

Details and paper references in `tasks/realistic_workload_analysis.md`.

### 2. Priority = Earliest Deadline

sglang already supports `--schedule-policy priority` with integer priorities. Modify to:
- Set priority = deadline timestamp (lower = earlier deadline = higher priority)
- For prefill admission: priority = arrival_time + TTFT_SLO
- For decode scheduling: priority = last_token_time + ITL_SLO
- Scheduler picks requests with earliest deadlines first (EDF)

Key files:
- `python/sglang/srt/managers/schedule_policy.py` — priority scheduling logic
- `python/sglang/srt/managers/scheduler.py` — request admission, batch formation
- `python/sglang/srt/managers/io_struct.py` — request metadata (add deadline fields)

### 3. SLO-Aware Chunk Size Control (DMC + LPT only)

Use a **server-level** latency prediction model to estimate: **how many prefill tokens can we admit in this step without violating the ITL SLO of any running decode request?**

- Retrain `tasks/server_perf_model.md` MLP model for BOTH LPT and DMC **server configurations**
- Uses the existing `overlap_profiler.py` to profile full forward pass (MLP + attention + all layers)
- Note: kernel-level latency is too noisy (L2 eviction effects) — server-level profiling smooths this out, which is why LPT exists
- Input: (n_p, r_p, r_d, decode_bs)
- Output: predicted server step latency (ms)
- At batch formation time:
  1. For each candidate prefill chunk size C:
     - Predict step_latency(C, prefix_len, decode_kv_avg, decode_bs)
     - Check: for ALL running decode requests, does `now + step_latency - last_token_time <= ITL_SLO`?
  2. Use largest C that satisfies all decode SLOs
  3. If no prefill can be admitted without violating SLOs, run pure decode

This applies to DMC and LPT (mixed batches). PP doesn't mix, so it only uses EDF for prefill admission order.

### 4. Benchmarking

**Workloads** (must have traces for realistic arrival patterns):
1. ShareGPT (general chat) — rate TBD from trace analysis
2. Code-Feedback (code generation with feedback loops) — rate TBD
3. Nebius-SWE (SWE-agent trajectories) — may need lower rate (compute-bound at rate=2+)

**Metrics**:
- SLO attainment rate (% requests meeting both TTFT and ITL SLOs)
- P50/P95/P99 TTFT and ITL
- Total throughput (tok/s)
- Pareto curves: vary request rate, plot ITL P99 vs throughput for all 3 methods

**Configs**: PP+PCG, LPT+CG29, DMC (all with EDF scheduling + SLO-aware chunk size where applicable)

---

## Implementation Plan

### Phase 1: SLO Research + Performance Model
- Research TTFT/ITL SLO standards from Tempo, DistServe, Sarathi-Serve, vLLM papers
- Define per-workload SLOs
- Profile server-level latency for LPT and DMC configs using `overlap_profiler.py`
- Retrain MLP to predict server step latency for both configs

### Phase 2: EDF Scheduler
- [x] Modify priority scheduling to use deadline timestamps
- [x] Add TTFT/ITL deadline fields to request metadata
- [x] Implement SLO-aware chunk size control using latency model
- [x] Fix clock mismatch: `last_tic` uses `time.monotonic()`, EDF must use monotonic consistently (not perf_counter)
- [x] Add `SGLANG_EDF_LATENCY_MODEL` env var to specify per-config model path
- [x] Wire predictor into scheduler init and `get_new_batch_prefill`

### Phase 3: Benchmarking
- [x] Run 3 workloads × 3 methods × 5 request rates (commit c68bfcefe, sglang 85b129fd3)
  - Memory leak workaround applied: log error + reset instead of crash
  - Command: `python3 -m sglang.bench_serving --backend sglang --dataset-name {sharegpt,code-feedback,swe-gym} --num-prompts {200,200,100} --request-rate {1,2,4,6,8}`
  - PP: `--enable-piecewise-cuda-graph` (no `--enable-mixed-chunk`)
  - DMC: `--enable-mixed-chunk --enable-piecewise-cuda-graph`
  - LPT_CG: `SGLANG_FLASHINFER_USE_PERSISTENT=1 --enable-mixed-chunk --enable-piecewise-cuda-graph`

#### ShareGPT (Qwen3-8B, H100)
| Rate | Method | Output tok/s | TTFT P99 (ms) | ITL P99 (ms) |
|------|--------|-------------|---------------|--------------|
| 1 | PP | 883 | 82 | 9.5 |
| 1 | DMC | 883 | 906 | 9.6 |
| 1 | LPT_CG | 883 | 913 | 9.8 |
| 2 | PP | 1593 | 95 | 20.3 |
| 2 | DMC | 1593 | 499 | 14.6 |
| 2 | LPT_CG | 1592 | 455 | 14.7 |
| 4 | PP | 2027 | 96 | 32.3 |
| 4 | DMC | 2032 | 452 | 25.6 |
| 4 | LPT_CG | 2031 | 445 | 26.3 |
| 6 | PP | 2209 | 324 | 36.0 |
| 6 | DMC | 2225 | 166 | 28.8 |
| 6 | LPT_CG | 2225 | 107 | 28.5 |
| 8 | PP | 2306 | 95 | 38.4 |
| 8 | DMC | 2309 | 556 | 33.3 |
| 8 | LPT_CG | 2316 | 475 | 33.1 |

#### Code-Feedback (Qwen3-8B, H100)
| Rate | Method | Output tok/s | TTFT P99 (ms) | ITL P99 (ms) |
|------|--------|-------------|---------------|--------------|
| 1 | PP | 987 | 45 | 8.8 |
| 1 | DMC | 987 | 46 | 8.9 |
| 1 | LPT_CG | 987 | 44 | 8.9 |
| 2 | PP | 1776 | 46 | 16.5 |
| 2 | DMC | 1776 | 48 | 10.2 |
| 2 | LPT_CG | 1776 | 51 | 10.0 |
| 4 | PP | 2718 | 48 | 19.9 |
| 4 | DMC | 2729 | 54 | 14.5 |
| 4 | LPT_CG | 2728 | 51 | 14.3 |
| 6 | PP | 3172 | 53 | 22.3 |
| 6 | DMC | 3192 | 59 | 18.2 |
| 6 | LPT_CG | 3191 | 58 | 18.9 |
| 8 | PP | 3432 | 360 | 25.9 |
| 8 | DMC | 3461 | 68 | 20.5 |
| 8 | LPT_CG | 3459 | 66 | 21.3 |

#### SWE-Gym (Qwen3-8B, H100)
| Rate | Method | Output tok/s | TTFT P99 (ms) | ITL P99 (ms) |
|------|--------|-------------|---------------|--------------|
| 1 | PP | 821 | 108 | 10.6 |
| 1 | DMC | 821 | 198 | 9.5 |
| 1 | LPT_CG | 821 | 464 | 9.5 |
| 2 | PP | 1179 | 126 | 19.6 |
| 2 | DMC | 1181 | 476 | 13.7 |
| 2 | LPT_CG | 1180 | 198 | 13.5 |
| 4 | PP | 1472 | 152 | 33.0 |
| 4 | DMC | 1473 | 290 | 27.7 |
| 4 | LPT_CG | 1473 | 461 | 24.5 |
| 6 | PP | 1602 | 144 | 38.9 |
| 6 | DMC | 1605 | 150 | 32.3 |
| 6 | LPT_CG | 1605 | 153 | 32.4 |
| 8 | PP | 1673 | 149 | 40.3 |
| 8 | DMC | 1675 | 463 | 34.8 |
| 8 | LPT_CG | 1677 | 224 | 34.7 |

#### Key Findings
1. **Throughput**: All 3 methods achieve nearly identical throughput at all rates — the bottleneck is the model, not the scheduling policy
2. **ITL P99**: DMC and LPT_CG consistently beat PP by 15-30% on ITL P99 at high rates (mixed batches reduce decode starvation)
3. **TTFT P99**: PP has the lowest TTFT at low rates (no prefill-decode interference), but DMC/LPT_CG are competitive at high rates
4. **LPT vs DMC**: Nearly identical performance — the persistent kernel adds no overhead but also no benefit at this scale
5. **No EDF enabled in these runs** — these are baseline comparisons without priority scheduling. EDF comparison requires re-running with `--enable-priority-scheduling`

- [ ] Generate Pareto curves (throughput vs ITL P99)
- [ ] Re-run with EDF scheduling enabled and compare SLO attainment

### Blockers
1. **req_to_token_pool memory leak**: Fixed upstream in commit `027f314` on `public/main` — "[Fix] data race in req_to_token pool (#17850)". The fix changes `alloc()`/`free()` to be request-based, preventing double-frees. Need to cherry-pick into `attn_fuse_`. Touches 13 files including `memory_pool.py`, `radix_cache.py`, `scheduler.py`.
2. **LPT cooperative kernel hang**: Persistent kernel hangs under load (~30+ concurrent requests). **DMC does NOT hang** under identical load — confirmed by side-by-side test (DMC completed all 50 requests in 180s, LPT hung at 163 requests). The hang happens when only 1-2 requests remain running (last decode batch shows `#running-req: 1`). No error messages — the scheduler loop simply freezes. Possible causes:
   - Cooperative kernel deadlock during CG replay at a specific batch size
   - State corruption from mixed CG replay that only manifests on the final request
   - Stream interference from `non_blocking=True` operations (identified in `forward_batch_info.py:771` and `flashinfer_backend.py:1358,1381,1892`)
