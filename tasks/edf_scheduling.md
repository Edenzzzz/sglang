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
- [ ] Run 3 workloads × 3 methods × multiple request rates — **BLOCKED** by `req_to_token_pool memory leak` bug
  - PP with EDF ran 1 rate (ShareGPT rate=1): TTFT P99=258s, ITL P99=8.6ms, throughput=406 tok/s
  - Server hangs after ~200-1000 requests with `ValueError: req_to_token_pool memory leak detected! available_size=85260, total_size=4096`
  - This is a pre-existing sglang bug (available_size > total_size), not EDF-related
  - LPT_CG also hangs under load (cooperative kernel hang, separate from memory leak)
  - DMC with EDF tested successfully on small load (5 concurrent requests), chunk control working
- [ ] Generate Pareto curves (throughput vs ITL P99) — needs stable server first
- [ ] Analyze SLO attainment breakdown

### Blockers
1. **req_to_token_pool memory leak**: Fixed upstream in commit `027f314` on `public/main` — "[Fix] data race in req_to_token pool (#17850)". The fix changes `alloc()`/`free()` to be request-based, preventing double-frees. Need to cherry-pick into `attn_fuse_`. Touches 13 files including `memory_pool.py`, `radix_cache.py`, `scheduler.py`.
2. **LPT cooperative kernel hang**: Persistent kernel hangs under load (~30+ concurrent requests). **DMC does NOT hang** under identical load — confirmed by side-by-side test (DMC completed all 50 requests in 180s, LPT hung at 163 requests). The hang happens when only 1-2 requests remain running (last decode batch shows `#running-req: 1`). No error messages — the scheduler loop simply freezes. Possible causes:
   - Cooperative kernel deadlock during CG replay at a specific batch size
   - State corruption from mixed CG replay that only manifests on the final request
   - Stream interference from `non_blocking=True` operations (identified in `forward_batch_info.py:771` and `flashinfer_backend.py:1358,1381,1892`)
