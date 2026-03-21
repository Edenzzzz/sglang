# Debug: Persistent/POD Kernel Server Performance

## Goal
Investigate and fix performance overhead of persistent kernel and POD (Prefill-On-Decode) attention in the SGLang flashinfer backend, compared to the default (non-persistent) path.

## Setup
- Model: `Qwen/Qwen3-8B`
- Server: `--enable-mixed-chunk --attention-backend flashinfer`
- Env vars: `SGLANG_FLASHINFER_USE_PERSISTENT=1`, `SGLANG_FLASHINFER_USE_POD=1`
- Benchmark: `bench_serving --dataset-name random --random-input 4096 --random-output 1024 --request-rate 8 --num-prompt 100`
- Profile logs: `tasks/profile_logs/`

## Profiling Configurations
| Config | Server log | nsys trace | sqlite |
|--------|-----------|------------|--------|
| Default (no persistent) | `default_server.log` | `default_bt.nsys-rep`, `default_kernel.nsys-rep` | `default_kernel.sqlite` |
| Persistent (no POD) | `persistent_bt_server.log` | `persistent_bt.nsys-rep`, `persistent_kernel.nsys-rep` | `persistent_kernel.sqlite` |
| POD | `pod_bt_server.log` | `pod_bt.nsys-rep`, `pod_kernel.nsys-rep` | `pod_kernel.sqlite` |
| POD optimized | `pod_opt_server.log` | `pod_opt_bt.nsys-rep` | `pod_opt_bt.sqlite` |

## Findings & Fixes

### 1. Decode path using persistent kernel unnecessarily (commit 637d3889)
**Problem:** When `use_persistent=True`, the decode path was using `BatchAttention` (persistent kernel), which is slower than the default decode wrapper for pure decode batches.

**Root cause:** The code was using persistent wrappers for all metadata types including `DecodeMetadata`, but the persistent kernel is only beneficial for mixed batches (prefill+decode).

**Fix:** Only use persistent kernel for mixed batch attention; fall back to `self.decode_wrappers` for `DecodeMetadata`.

**Impact:** 3846 tok/s (fixed) vs 3586 tok/s (before) = **+7.2% throughput**

### 2. `torch.arange` allocation in decode persistent path (commit 34d5c728)
**Problem:** `torch.arange(bs)` was being allocated every call in the decode persistent kernel path.

**Fix:** Cache the `torch.arange` buffer and reuse it.

### 3. POD attention launch overhead (commit 6244007a)
**Problem:** POD attention had significant CPU-side overhead from:
- Per-call `torch.empty` allocation for `kv_indices` buffer in `update_pod`
- Recomputing `extend_lens = seq_lens - prefix_lens` for ragged wrapper plan
- Unnecessary try/except around `pod_wrapper.plan()`

**Fixes:**
- Cache `kv_indices` buffer in `FlashInferIndicesUpdaterPrefill` and reuse across calls
- Reuse `extend_lens` (already computed as `seq_lens - prefix_lens`) for the ragged wrapper plan
- Remove try/except for cleaner control flow

**Impact (POD):** 3804 tok/s (+3.3%), 182ms TTFT (-31%), 12.0ms ITL (-19%)

### 4. Benchmark CSV logging (commit 5c268789)
Added `use_persistent` and `use_pod` fields to `bench_serving.py` CSV output and `http_server.py` `/get_server_info` endpoint for tracking kernel configuration in benchmark results.

## Remaining TODOs
From `flashinfer_backend.py`:

1. **Remove `torch.cat` for POD outputs** (line ~857):
   ```python
   # TODO: remove cat
   o = torch.cat([o_p, o_d], dim=0)
   lse = torch.cat([lse_p, lse_d], dim=0)
   ```
   Could use `join_outputs=False` in `pod_wrapper.plan()` or pre-allocate output buffer.

2. **Remove device sync in prefill indices updater** (line ~1414):
   ```python
   # TODO: remove this device sync, we can use forward_batch.extend_prefix_lens_cpu
   ```
3. Look into `class BatchPODWithPagedKVCacheWrapper` in `../flashinfer/flashinfer/pod.py`
to see if some indices operations can be optimized.

4. Verify valid output using send_reqs.sh 

## Remaining Overhead
The main remaining overhead in POD attention is inside flashinfer's `pod_wrapper.plan()` itself, which contains GPU→CPU synchronizations (`.to("cpu")` calls at lines 849, 859-862, 897, 907-910 of pod.py). This is upstream in flashinfer and not easily fixable from the SGLang side.

## GPU Launch Gap Analysis (from nsys bt traces)
| Config | Num gaps | Avg gap (us) | Total gap (ms) | Trace duration (ms) | Gap % |
|--------|----------|-------------|---------------|---------------------|-------|
| Default (before) | 58,253 | 147.8 | 8,612 | 13,106 | 65.7% |
| Persistent (before) | 55,631 | 149.4 | 8,312 | 12,525 | 66.4% |
| Persistent (after fix) | 60,419 | 126.9 | 7,666 | 12,244 | 62.6% |
| POD (before, kernel trace*) | 53,250 | 3,202.7 | 170,545 | 171,665 | 99.3% |
| POD (after fix) | 57,827 | 159.9 | 9,249 | 10,836 | 85.4% |

*POD before-fix used kernel-level trace (higher overhead), but the 99.3% gap still indicates extreme CPU-side bottleneck.

**Key findings:**
- POD fix reduced gap from ~99% to ~85% — massive improvement but still ~20% more gap than default/persistent
- Persistent fix reduced avg gap from 149→127 us (-15%) and total gap% from 66%→63%
- Default baseline gap% is ~66%, so persistent is now on par

## Benchmark Summary

### flashinfer 0.4.1 (original profiling run)
| Config | Throughput (tok/s) | Mean TTFT (ms) | Median ITL (ms) |
|--------|-------------------|----------------|-----------------|
| Default (no persistent) | 3572 | 2519 | 18.3 |
| Default (profiled) | 3640 | 1148 | 16.9 |
| Persistent (before fix) | 3586 | 1033 | 17.8 |
| POD (before fix) | 3684 | 356 | 14.8 |
| POD (optimized, commit 6244007) | 3804 | 182 | 12.0 |

### flashinfer 0.5.2 (post-fix, current)
| Config | Throughput (tok/s) | Mean TTFT (ms) | Median ITL (ms) | Notes |
|--------|-------------------|----------------|-----------------|-------|
| Default | 2836 | 145 | 8.7 | warmup=1 |
| Persistent | 2838 | 102 | 8.6 | warmup=1 |
| POD (join_outputs) | 2938 | 129 | 8.1 | warmup=25 (JIT compile) |

**NOTE:** POD requires `--warmup-requests 25` (or flush cache + re-run) because flashinfer JIT-compiles POD kernels on first use, which is very slow. The earlier 601 tok/s result was caused by JIT overhead during the benchmark. flashinfer 0.5.2 shows ~20% throughput regression for all configs vs 0.4.1.
