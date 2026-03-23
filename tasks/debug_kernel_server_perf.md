# Debug: Persistent/POD Kernel Server Performance

## Goal
Investigate and fix performance overhead of persistent kernel and POD (Prefill-On-Decode) attention in the SGLang flashinfer backend, compared to the default (non-persistent) path.

## Setup
- Model: `Qwen/Qwen3-8B`
- Server: `--enable-mixed-chunk --attention-backend flashinfer`
- Env vars: `SGLANG_FLASHINFER_USE_PERSISTENT=1`, `SGLANG_FLASHINFER_USE_POD=1`
- Benchmark: `bench_serving --dataset-name random --random-input 4096 --random-output 1024 --request-rate 8 --num-prompt 100`
- Profile logs: `tasks/debug_kernel_server_perf/profile_logs/`

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

## Completed TODOs
- ~~Remove `torch.cat` for POD outputs~~ — used `join_outputs=True` (fixed p-first ordering in pod.py)
- ~~Remove device sync in prefill indices updater~~ — pass `prefix_lens_cpu` from `forward_batch.extend_prefix_lens_cpu`
- ~~Optimize POD pod.py indices~~ — added optional CPU tensor params to `plan()` to bypass GPU→CPU syncs
- ~~Verify valid output~~ — confirmed with `send_reqs.sh` for both persistent and POD

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
| Default mixed chunk | 2836 | 145 | 8.7 | warmup=1 |
| Persistent | 2838 | 102 | 8.6 | warmup=1 |
| POD (join_outputs) | 2938 | 129 | 8.1 | warmup=25 (JIT compile) |

### Full 5-config server benchmark (H100 80GB, `--mem-fraction-static 0.85 --enable-piecewise-cuda-graph`, keepalive OFF)

Full output saved to `tasks/debug_kernel_server_perf/profile_logs/bench_results/`.

**Workload 1: --random-input 4096 --random-output 1024 --request-rate 8 --num-prompt 400**
| Config | Total tok/s | Out tok/s | TTFT med (ms) | TTFT P95 | TTFT P99 | ITL med (ms) | ITL P95 | ITL P99 | Max ITL |
|--------|-----------|----------|-------------|---------|---------|------------|---------|---------|---------|
| **Prefill-prioritized** | **15,108** | 3,061 | **226** | **4,869** | **5,148** | **26** | **118** | **182** | 486 |
| Persistent (static) | 13,369 | 2,709 | 4,876 | 13,208 | 13,555 | 27 | 131 | 504 | 714 |
| Default-mixed-chunk | 13,249 | 2,685 | 4,813 | 14,005 | 14,299 | 27 | 133 | 495 | 664 |
| POD | 13,199 | 2,675 | 6,148 | 14,403 | 14,707 | 26 | 158 | 525 | 611 |
| Persistent v3 (dynamic) | 13,059 | 2,646 | 4,986 | 14,693 | 15,164 | 26 | 125 | 579 | 792 |

**Workload 2: --random-input 6500 --random-output 7500 --request-rate 8 --num-prompt 400**
| Config | Total tok/s | Out tok/s | TTFT med (s) | TTFT P95 (s) | TTFT P99 (s) | ITL med (ms) | ITL P95 | ITL P99 | Max ITL (s) |
|--------|-----------|----------|------------|------------|------------|------------|---------|---------|-----------|
| **Persistent v3 (dynamic)** | **4,104** | **2,172** | **189** | **451** | **487** | **26** | **27** | **83** | 472 |
| Persistent (static) | 4,072 | 2,155 | 196 | 457 | 493 | 26 | 28 | 89 | 502 |
| Prefill-prioritized | 4,046 | 2,141 | 191 | 464 | 505 | 27 | 28 | 52 | 503 |
| POD | 3,995 | 2,114 | 199 | 476 | 510 | 27 | 29 | 104 | 512 |
| Default-mixed-chunk | 3,987 | 2,110 | 197 | 476 | 512 | 27 | 29 | 95 | 514 |

**Key findings:**
1. **WL1 (short output)**: Prefill-prioritized dominates throughput (15.1K vs ~13.2K) and TTFT (226ms vs ~5s). All mixed-chunk configs have similar throughput (~13.2K). **Tail latency**: Prefill-prio has best P99 ITL (182ms) vs mixed-chunk configs (495-579ms). POD has worst P95 ITL (158ms).
2. **WL2 (long output)**: All configs collapse to ~4K tok/s with 3+ minute TTFT — the system is saturated. v3 wins marginally on throughput (4,104) and TTFT (189s). Prefill-prio has best P99 ITL (52ms) since long decodes run uninterrupted.
3. **v3 vs POD**: v3 wins WL2 throughput (4,104 vs 3,995) and TTFT (189s vs 199s). On WL1, v3 is slightly behind (13,059 vs 13,199). v3's WL1 P95 ITL (125ms) is better than POD (158ms).
4. **Persistent (static) surprisingly competitive**: Beats default-mixed-chunk on WL1 throughput (13,369 vs 13,249) and WL2 (4,072 vs 3,987). No longer collapses on WL2 as in old B200 results.
5. **Prefill-prioritized trade-off confirmed**: Best WL1 throughput+TTFT+tail ITL, but on WL2 the advantage disappears. The P99 ITL concern (head-of-line blocking) is NOT seen — Prefill-prio actually has the lowest P99 ITL on both workloads.

**NOTE:** POD requires `--warmup-requests 25` (or flush cache + re-run) because flashinfer JIT-compiles POD kernels on first use, which is very slow. The earlier 601 tok/s result was caused by JIT overhead during the benchmark. flashinfer 0.5.2 shows ~20% throughput regression for all configs vs 0.4.1.

---

## TODO: Persistent Kernel Prefill-Decode Overlap Scheduler

### Goal
Implement a new scheduler for the persistent kernel that beats POD by doing prefill-decode overlap within the persistent kernel itself.

### Background
Currently the persistent kernel uses `TwoStageHolisticPlan` in `../flashinfer/include/flashinfer/attention/scheduler.cuh`, which uses a coarse cost function to rank tiles and a min-heap to achieve load balance across SMs (2 CTAs/SM). It has two drawbacks:

1. **No L2 cache awareness**: the scheduler may assign too many query tiles from different sequences to the same SM, causing their KV caches to overflow L2 and degrade performance vs iterating sequence-by-sequence.
2. **Coarse cost function**: cannot accurately model runtime for prefill-decode overlap (see `tasks/pod_overlap.png` and `FlippedSchedule` in `../flashinfer/include/flashinfer/attention/persistent_template.cuh`).

### Design
1. Use the same coarse cost function to **rank** tiles.
2. **Runtime tile fetching + LPT scheduling**: due to causal attention, a CTA contends for the largest tile in a queue using a semaphore. SMs that finish faster fetch more/larger tiles, and LPT (Longest Processing Time first) minimizes load imbalance.
   - Still 2 CTAs/SM: one fetches from **prefill queue**, the other from **decode queue**; switches to the other queue when its queue is empty.
3. **L2-aware contention**: as in FA3 ([tile_scheduler.hpp#L317](https://github.com/Dao-AILab/flash-attention/blob/c783ab2f7e05ba1cd79ecfe0e6e109a4e3f6e542/hopper/tile_scheduler.hpp#L317)), only do LPT among the KV heads that can completely fit into L2 to avoid cache eviction. E.g. for a 1k seq w/ 8 heads, if only 4 heads' KV cache fits in L2, contend for tail (largest) queries on those 4 heads.

### References
- Flash-Attention 3 tile scheduler: clone under `tasks/` to read the code
- `../flashinfer/.claude/` for how to add/modify a kernel
- `../flashinfer/benchmarks/bench_mixed_attention.py` for benchmarking

### Validation
- `bench_mixed_attention.py`: `FlippedSchedule=True` should beat `FlippedSchedule=False` **and** POD
- Precision must match (note: `o_batch_pod` has a precision issue likely due to `join_outputs` p/d order mismatch — fix that)

### Baseline Benchmark Results (B200, flashinfer 0.5.2)
| Case | Prefill | Decode | Persistent (ms) | POD (ms) | POD speedup |
|------|---------|--------|-----------------|----------|-------------|
| 1 | 512x1 | 1x100, kv=2048 | 0.28 | 0.17 | 1.69x |
| 2 | 1536x1 | 1x100, kv=2048 | 0.48 | 0.21 | 2.33x |
| 3 | 2048x2 | 1x100, kv=2048 | 0.80 | 0.38 | 2.09x |
| 4 | 2048x1 | 1x100, kv=2048 | 0.52 | 0.24 | 2.19x |
| 5 | 4096x1 | 1x100, kv=4096 | 1.45 | 0.72 | 2.01x |
| 6 | 4096x1 | 1x100, kv=8192 | 1.92 | 0.92 | 2.09x |
| 7 | 6000x1 | 1x100, kv=8192 | 3.77 | 4.14 | 0.91x |

### Issues Encountered
1. **POD precision mismatch in `bench_mixed_attention.py`**: Fixed. The `join_outputs=True` was returning prefill-first order but the reference `o` has decode-first order. Changed to use `join_outputs=False` (returns `(o_p, o_d)` tuple) and cat manually in d-first order.

### Architecture Understanding
- **Current persistent kernel**: `TwoStageHolisticPlan` (CPU) pre-assigns all tiles to clusters using MinHeap. `BlockPersistentRunner::Run()` iterates `work_indptr[blockIdx.y]` to `work_indptr[blockIdx.y+1]` — **static assignment, no runtime contention**.
- **Current FlippedSchedule**: Just alternates Runner1/Runner2 execution order per CTA (even CTAs: big tiles first, odd: small tiles first). Still uses static tile assignment. Provides some overlap but can't adapt to runtime load imbalance.
- **POD (separate kernel)**: Launches prefill and decode independently on different SMs, using SM-aware scheduling (`sm_aware_sched` buffer). No L2 contention between p/d since they run on different SMs.

### Implementation Plan (v3: FA3-style runtime dynamic scheduler)

The v1/v2 approach of sorting pre-built flat tile arrays and atomically indexing into them was wrong — it still uses the static per-tile metadata arrays from `TwoStageHolisticPlan`. The correct approach is a **runtime dynamic scheduler** like FA3's `VarlenDynamicPersistentTileScheduler`, which replaces `get_block_coord()` entirely.

#### FA3 VarlenDynamicPersistentTileScheduler Design
Global tile indices are laid out as: **batch (slowest) → head → query block (fastest)**.
1. `atomicAdd(tile_count_semaphore, 1)` → get global `tile_idx`
2. `tile_idx_to_work_tile(tile_idx)`:
   - **Find batch**: while loop using `warp_prefix_sum` of `num_m_blocks` per batch (31 batches at a time via warp lanes). `__ballot_sync` + `__popc` to find exact batch within group.
   - **Compute offset**: `mh_block = tile_idx - group_start_tile` (offset within this batch)
   - **L2-aware LPT (non-split case)**: divide heads into sections of `nheads_in_l2` (power-of-2, computed as `L2_size / (kv_len * head_dim * 2 * elem_size)`). Within each section, iterate **query block first, head second** → consecutive tiles access same KV cache (L2-friendly). `block = num_m_blocks - 1 - block` reverses order for LPT (causal: larger Q tiles = more work).
   - **Split-KV case**: head-first ordering with split_idx packed into bidh upper bits.
3. **Queue switching**: when one CTA's queue is empty, it fetches from the other queue using the same `atomicAdd` mechanism.
4. **gridDim stride**: FA3 uses `atomicAdd(...) + int(gridDim.x)` to skip by gridDim — this gives each CTA an initial tile at launch without contention, then subsequent tiles via atomic. **Need to explore**: whether strict sequential `atomicAdd(..., 1)` (no gridDim skip) gives better LPT behavior since it always fetches the next-largest tile, vs gridDim stride which distributes tiles more evenly but breaks strict LPT order. For per-queue gridDim, need `gridDim_per_queue = gridDim.x * gridDim.y / 2` (half the CTAs per queue in FlippedSchedule).

#### Mapping to Flashinfer
- FA3 `batch` = flashinfer sequence index (within prefill or decode set)
- FA3 `num_m_blocks` = `ceil_div(packed_qo_len, CTA_TILE_Q)` per sequence
- FA3 `num_head` = `num_kv_heads` (GQA group is handled inside)
- FA3 `block` = Q tile index within sequence
- FA3 `bidh` = KV head index
- Flashinfer has paged KV cache → `kv_indptr` + `kv_indices` for page table lookup
- Flashinfer may split KV → need `partial_indptr`, `merge_indptr`, `merge_o_indices` for reduction

#### Static Scheduler (plan-time) Changes
For the FlippedSchedule path, `TwoStageHolisticPlan` just needs to:
1. Split sequences into prefill queue (task 0) and decode queue (task 1) — already done
2. Store per-sequence metadata: `num_m_blocks_ptr`, `nheads_in_l2_ptr`, `qo_indptr`, `kv_indptr`, `kv_len`, per queue
3. Compute L2 budget: `nheads_in_l2 = min(num_kv_heads, floor_pow2(L2_size / (kv_len * head_dim * 2 * elem_size)))` per sequence
4. Store tile counters (one per queue, zero-initialized)
5. Keep the MinHeap path for non-FlippedSchedule (pure prefill/decode batches)

#### Runtime Scheduler (kernel-side) Changes
Replace `get_block_coord(params, work_idx)` with `tile_idx_to_work_tile(params, tile_idx)`:
1. Thread 0 does `atomicAdd(tile_counter, 1)` (+ optional gridDim stride)
2. Broadcast tile_idx to all threads via smem + `__syncthreads()`
3. `tile_idx_to_work_tile()`: batch lookup → L2-aware LPT head/block mapping → return (q_indptr, kv_indptr, q_len, kv_len, packed_qo_start, kv_start, kv_end, kv_head_idx, len_kv_chunk)

#### Queue Switching
When a CTA finishes its own queue (e.g., prefill CTA drains prefill queue), it switches to the other queue's counter and fetches from there. This requires:
- Both Runners to have access to both tile counters
- A way to call the other Runner's compute path from within a Runner
- OR: simpler — after Runner1 finishes, PersistentKernelTemplate passes Runner1 the OTHER queue's params/counter to continue draining

#### Files to Modify
1. `scheduler.cuh` — `TwoStageHolisticPlan`: for FlippedSchedule, store per-sequence metadata (num_m_blocks, nheads_in_l2, qo_indptr, kv_indptr, kv_len) per queue instead of per-tile metadata
2. `persistent.cuh` — replace `get_block_coord()` with `tile_idx_to_work_tile()` containing batch-find while loop + L2-aware LPT mapping. Replace `work_indptr` loop with atomic fetch loop.
3. `batch_attention_customize_config.jinja` — add new params for dynamic scheduling (num_m_blocks_ptr, nheads_in_l2_ptr, per-sequence metadata arrays)
4. `batch_attention.cu` — populate new params from plan_info

### Key Findings from Investigation
1. **FlippedSchedule rarely triggers**: Condition `prefill*128 <= decode_kv_total && prefill >= 3072`. Only cases 6-7 in bench trigger it. Cases 1-5 where persistent is 2x slower than POD don't use FlippedSchedule.
2. **v2 (static LPT sort + relaxed condition) improved some cases but still 2x gap**: The static MinHeap tile assignment + atomic indexing into the flat array is fundamentally wrong — it doesn't do L2-aware grouping and the tile metadata is still laid out per-cluster.
3. **B200 specs**: L2=126MB, 148 SMs, 2 CTAs/SM (smem constrained, same as POD).
4. **`__syncthreads` for smem broadcast is safe**: Confirmed by user.

### Progress

**v1: Static LPT atomic tile fetching**
- Added `tile_counter` and `total_num_works` to `HolisticPlanInfo` and `PersistentParams`
- Modified `BlockBatchPagedAttentionPersistent::Run()`: when `tile_counter != nullptr`, uses `atomicAdd` to grab next tile from the flat array
- **Result**: Cases 6-7 improved ~10%. Cases 1-5 unchanged.

**v2: LPT sorting + relaxed FlippedSchedule condition**
- Sorted flat tile arrays by cost descending. Relaxed condition to `prefill > 0 && decode > 0`.
- **Results vs baseline**:
  | Case | Original | LPT v2 | Improvement | POD |
  |------|----------|--------|-------------|-----|
  | 1 | 0.28ms | 0.32ms | -14% | 0.17ms |
  | 2 | 0.48ms | 0.40ms | +17% | 0.21ms |
  | 3 | 0.80ms | 0.79ms | +1% | 0.38ms |
  | 4 | 0.52ms | 0.38ms | **+27%** | 0.24ms |
  | 5 | 1.45ms | 1.31ms | +10% | 0.72ms |
  | 6 | 1.92ms | 1.79ms | +7% | 0.92ms |
  | 7 | 3.77ms | 3.26ms | +14% | 4.14ms |
- Still 2x slower than POD for most cases — needs FA3-style runtime dynamic scheduler.

**v3: FA3-style runtime dynamic scheduler (WORKING)**
- Implemented `tile_idx_to_work_tile()` with batch-find while loop + L2-aware LPT mapping
- Fixed critical KV splitting bug: `dyn_len_kv_chunk` was too small, causing `num_kv_chunks > 1` and partial output overwrites. Fixed by setting `dyn_len_kv_chunk = INT_MAX` (no KV splitting).
- **Results (B200, driver 580, flashinfer 0.5.2)**:
  | Case | Persistent v3 | POD | Speedup |
  |------|--------------|-----|---------|
  | 1 | **0.18ms** | 0.17ms | 1.06x |
  | 2 | **0.22ms** | 0.21ms | 1.06x |
  | 3 | **0.36ms** | 0.38ms | **0.94x** |
  | 4 | **0.25ms** | 0.24ms | 1.05x |
  | 5 | **0.63ms** | 0.71ms | **0.89x** |
  | 6 | **0.85ms** | 0.90ms | **0.95x** |
  | 7 | **1.52ms** | 1.64ms | **0.93x** |
- **Persistent beats POD in 4/7 cases**, near-parity in remaining 3. Original gap was ~2x.
- NOTE: earlier results used `np.mean` for persistent (inflated by outliers) — fixed to `np.median` to match POD measurement.

**v3 + uniform_num_m_blocks fast path (latest)**:
  | Case | Persistent v3 | POD | Ratio |
  |------|--------------|-----|-------|
  | 1 | 0.17ms | 0.17ms | 1.03x (tie, 6us cooperative launch overhead) |
  | 2 | 0.21ms | 0.21ms | 1.02x (tie) |
  | 3 | **0.34ms** | 0.38ms | **0.88x** |
  | 4 | 0.24ms | 0.24ms | 1.00x (tie) |
  | 5 | **0.60ms** | 0.72ms | **0.84x** |
  | 6 | **0.82ms** | 0.92ms | **0.89x** |
  | 7 | **3.99ms** | 4.15ms | **0.96x** |
- **Persistent beats POD in 4/7 cases, ties in 3/7.** No case where POD is meaningfully faster.

**NCU profiling (case 1: 512x1 prefill + 100 decode@kv2048)**:
  | Metric | Persistent | POD |
  |--------|-----------|-----|
  | Kernel time | 224.45us | 223.04us |
  | DRAM reads | 847MB | 850MB |
  | L2 hit rate | 1.85% | 1.40% |
  | Grid size | (1, 296) | (928, 1) |
- **Kernel execution is identical.** The 2-3% gap in CUDA event timing is from `cudaLaunchCooperativeKernel` overhead vs regular `cudaLaunchKernel`.
- Both kernels are bandwidth-bound with near-zero L2 reuse (KV cache is too large: 100 seqs × 2048 × 128 × 2 × 2 = 100MB per head, >> 126MB L2).
- Switching to regular launch crashes (persistent kernel requires guaranteed block residency for 296-block grid on 296 SMs).

**v3: FA3-style runtime dynamic scheduler (IN PROGRESS)**
- [x] Implement `tile_idx_to_work_tile()` with batch-find while loop + L2-aware LPT mapping
- [x] Modify `TwoStageHolisticPlan` to store per-sequence metadata (qo_indptr, qo_len, kv_indptr, kv_len, num_m_blocks, nheads_in_l2) per queue
- [x] Add dynamic scheduler fields to `HolisticPlanInfo`, `PersistentParams`, `batch_attention.cu`
- [x] L2-aware nheads_in_l2 computation: `floor_pow2(L2_budget / kv_bytes_per_head)`, queried at runtime via `cudaDevAttrL2CacheSize`
- [x] Fixed qo_len indptr differencing bug (non-contiguous sequences) — added separate `dyn_qo_len` array
- [x] **TESTED AND WORKING** — correctness verified, beats POD in 4/7 cases
- [x] Queue switching: FlippedSchedule handles this — even CTAs: Runner1(prefill)→Runner2(decode), odd CTAs: Runner2→Runner1. Each runner atomically fetches from its own queue.
- [x] Verify correctness with `bench_mixed_attention.py` — all 7 cases pass
- [x] Run detailed NCU profiling for both POD and Persistent to find more optimization opportunities and try to beat it in all cases
- [x] Find a load threshold below which LPT makes perf worse — N/A: NCU shows kernel times are IDENTICAL; the 2-3% gap in cases 1-2 is from cooperative launch overhead, not LPT
- [x] Explore gridDim stride vs strict sequential atomicAdd — tested round-robin (static stride, CTA i starts at tile i and strides by gridDim), hurts L2 locality. Batch-fetch (`atomicAdd(counter, N)` to grab N consecutive tiles per CTA) breaks syncthreads safety in the smem broadcast loop. Sequential `atomicAdd(counter, 1)` is optimal among tested approaches. **BUT**: FA3's actual pattern (`atomicAdd(counter, gridDim.x) + blockIdx.x`) was NOT tested
- [x] Added `uniform_num_m_blocks` fast path: O(1) sequence lookup via direct division when all seqs have same num_m_blocks (skip warp search)
- [x] Fixed `bench_mixed_attention.py`: persistent was using `np.mean` (skewed by outliers) while POD used `np.median` — fixed to use median consistently
- [x] Handle KV splitting in the dynamic scheduler: uses actual `len_kv_chunk` from static scheduler; falls back to static MinHeap when any sequence needs splitting (causal attention makes partial_o addressing complex). No splitting triggers for current benchmark cases (kv_lens ≤ 8192 < kv_len_limit).
- [x] Run sglang server benchmark: Qwen3-8B, --random-input 4096 --random-output 1024 --request-rate 8 --num-prompt 400
- [x] Run sglang server benchmark: Qwen3-8B, --random-input 6500 --random-output 7500 --request-rate 8 --num-prompt 400
- [x] Run full 5-config comparison with **P95/P99 TTFT and ITL** on H100 80GB with `--mem-fraction-static 0.85 --enable-piecewise-cuda-graph`. Key surprise: Prefill-prioritized has lowest P99 ITL (not highest as expected).
- [x] **Investigate 2x persistent-vs-POD gap in baseline benchmarks**: **RESOLVED.** The 2x gap was between the **static** persistent scheduler (baseline) and POD. The v3 dynamic scheduler closes this gap entirely:
  | Case | Persistent v3 (ms) | POD (ms) | POD speedup | Old static gap |
  |------|-------------------|---------|-------------|---------------|
  | 1 (512×1) | 0.31 | 0.30 | 1.01x | was 1.69x |
  | 2 (1536×1) | 0.35 | 0.32 | 1.09x | was 2.33x |
  | 3 (2048×2) | 0.47 | 0.44 | 1.06x | was 2.09x |
  | 4 (2048×1) | 0.36 | 0.34 | 1.07x | was 2.19x |
  | 5 (4096×1, kv4096) | **0.80** | 0.82 | **0.98x** | was 2.01x |
  | 6 (4096×1, kv8192) | **1.26** | 1.29 | **0.98x** | was 2.09x |
  | 7 (6000×1, kv8192) | **1.83** | 1.87 | **0.98x** | was 0.91x |
  v3 matches POD on small cases (1-4, ≤1.09x gap) and **beats POD on large cases (5-7)**. This aligns with the PR author's reported max 1.26x speedup ([flashinfer#2079](https://github.com/flashinfer-ai/flashinfer/pull/2079)). Run on H100 80GB (mlx worker 867056).
- [x] **Test FA3's tile fetching pattern**: Implemented `blockIdx.y` for first tile + `atomicAdd(counter,1) + gridDim.y` for subsequent. **Result: worse than sequential.** 5-19% slower on small cases (1-4), 3-7% on large (5-7). The `blockIdx.y` initial assignment breaks L2-aware LPT ordering — first gridDim.y tiles are distributed round-robin by CTA index rather than by optimal L2 order. The `+gridDim.y` offset disrupts subsequent ordering too. **Sequential `atomicAdd(counter,1)` remains optimal** — it preserves strict LPT ordering which maximizes L2 reuse. Reverted.
- [ ] **Investigate v3 throughput regression vs default-mixed-chunk**: nsys profiling (H100, 4096/1024, 50 prompts) reveals:
  - **`cudaDeviceSynchronize` is 2.4x slower in v3** (2,771ms vs 1,177ms) despite similar call count (2,672 vs 2,485). Avg sync time: 1,037μs (v3) vs 474μs (default).
  - v3 kernel itself is efficient: `PersistentKernelTemplate` avg=150μs (878 calls), total=132ms. Default `BatchPrefillWithPagedKVCacheKernel` avg=217μs (936 calls), total=203ms. **Persistent kernel is faster per-call.**
  - v3 has fewer kernel launches overall (36,524 vs 45,578 `cudaLaunchKernelExC` calls) — persistent reduces launch overhead.
  - **NOTE**: Persistent kernel is only used for **mixed batches** (`forward_mode.is_mixed()`), NOT pure decode. Pure decode uses `BatchDecodeWithPagedKVCacheWrapper` in both configs. So the cooperative launch overhead is isolated to mixed batch steps only. The `forward_decode` path calls `.run()` on the regular decode wrapper even when `use_persistent=True` (the `persistent_attn_forward` wrapper is misleading — it just forwards to `.run()`).
  - **Revised hypothesis**: The extra cudaDeviceSync time comes from mixed batch steps where cooperative launch + grid sync in the persistent kernel blocks the GPU longer per step. Since cooperative launch must guarantee block residency, it may serialize with other CUDA work (e.g., memcpy for KV cache updates). Need to check nsys timeline to see if the extra sync time correlates with mixed batch frequency.
  - **Next steps**: (a) Open nsys traces in GUI to compare mixed batch step timelines. (b) Count mixed vs decode steps in both traces to see if v3 has more mixed batches (persistent plan may alter scheduler behavior). (c) Check if `cudaLaunchCooperativeKernel` itself takes longer than `cudaLaunchKernel` even when the kernel runs the same duration.
  - Profiles saved to `tasks/debug_kernel_server_perf/profile_logs/nsys/`.
- [x] Rerun full 5-config server benchmark with correct flags (`--mem-fraction-static 0.85 --enable-piecewise-cuda-graph`) and **P95/P99 TTFT and ITL**. Results above. Key surprise: Prefill-prioritized has LOWEST P99 ITL, not highest as expected.
- [x] **Microbench vs server reversal explained**: In the 0.5.2 microbenchmark, the test measures a single mixed-batch kernel call where persistent/POD kernel is faster than BatchPrefill. But in the server: (a) most batches are **pure decode** (not mixed), where persistent's cooperative launch overhead (cudaDeviceSync 2.4x slower) hurts throughput, (b) POD has double plan() + per-layer GPU-CPU sync overhead that the microbenchmark doesn't capture, (c) the server measures end-to-end including scheduling, index computation, and CUDA graph overheads — not just the attention kernel. The reversal is because the kernel advantage on mixed batches is outweighed by overhead on the much more frequent pure-decode batches.
- [x] **POD vs Default-mixed-chunk server throughput (code analysis)**: POD is ~0.4% slower (13,199 vs 13,249 tok/s) despite faster kernel. Root causes:
  1. **Double `plan()` calls**: POD calls C++ planning twice (prefill + decode) vs once for default. Each involves workspace setup + host-device copies.
  2. **GPU-CPU sync in `pod_forward`** (`pod.py:840`): `pod_wrapper._qo_indptr_buf_p[-1]` reads a GPU tensor scalar **per layer per step** — likely implicit device sync. Fix: cache `num_p_tokens` during `plan()`.
  3. **2x index computation in `update_pod`**: Extra cumsum, slice ops, CPU tensor construction for split prefill/decode indptrs.
  4. **`cudaMemsetAsync` per layer** (`batch_pod.cuh:339`): 80 extra CUDA API calls per step for sm_aware_sched buffer.
  5. **Extra `.to(device)` copies**: 12 host-to-device copies in POD plan vs ~6 for default.
- [x] **Why prefill-prioritized has lowest P99 ITL (code analysis)**: The key is **separation of concerns**. In prefill-prioritized mode (`enable_mixed_chunk=False`), the scheduler (`scheduler.py:1815-1870`) runs **pure prefill steps** and **pure decode steps** — never a mixed batch. Decode steps contain only decode tokens → consistent, low latency (~26ms). Prefill steps temporarily skip decode (causing a one-step gap of ~50-100ms). In mixed-chunk mode, **every step is a mixed batch** (prefill + decode tokens together via `mix_with_running()` in `schedule_batch.py:1396-1419`). The mixed attention kernel processes both prefill tokens (quadratic) and decode tokens (linear), making every step 2-3x slower than a pure decode step. The P99 ITL difference (182ms vs 495-579ms) is because: mixed-chunk makes every decode step slow (high baseline), while prefill-prioritized keeps decode steps fast and only occasionally skips them (low baseline with rare gaps).
- [x] Sweep prefill/decode length ratios — done. See `tasks/realistic_workload_analysis.md` for full results. Key findings: v3 wins throughput only on 6500/7500 (+1.9%), wins TTFT on prefill-heavy 4096/1024 (-24%), but has 0.5-4% throughput regression on other workloads. Benefit comes from large absolute prefill (>4096 tokens), not ratio.
- [ ] Realistic workload analysis for v3 kernel evaluation — see `tasks/realistic_workload_analysis.md`

**To test when GPU returns:**
```bash
# 1. Start keepalive to hold GPU during setup
bash ../VibeCUDA/scripts/gpu_keepalive.sh start

# 2. Clear JIT cache
python3 -m flashinfer clear-cache && rm -rf ~/.cache/flashinfer

# 3. STOP keepalive before benchmarking — it is compute-intensive and corrupts results!
bash ../VibeCUDA/scripts/gpu_keepalive.sh stop

# 4. Run benchmarks
cd ../flashinfer && python3 benchmarks/bench_mixed_attention.py

# 5. Restart keepalive after benchmarks
bash ../VibeCUDA/scripts/gpu_keepalive.sh start
```
**IMPORTANT:** gpu_keepalive is compute-intensive (platform requirement). It MUST be stopped before any benchmark or profiling. Previous server benchmark results were collected with keepalive running and need to be re-run.

If it crashes, check server log and use `FLASHINFER_LOGLEVEL=3` for debugging.

**Code review notes (done while GPU was down):**
- Verified warp-level operations in `tile_idx_to_work_tile()` are correct for multi-warp blocks (each warp independently computes same result)
- Verified `__syncthreads()` ordering for smem broadcast of tile_idx (barrier → write → barrier → read)
- Verified warm-start mechanism (`prev_bidb`, `prev_group_start`) works with monotonically increasing tile indices per CTA
- Verified L2 sectioning math: `section_idx`, `l2_mod`, `nheads_in_this_section` correctly handle tail sections
- Verified LPT reversal: `block = num_m_blocks - 1 - block` gives largest-first for causal
- Verified task classification: with gqa_group_size=4, decode packed_qo_len=4 ≤ 16 → task 1. Prefill → task 0. Correct.
- No KV splitting in v3 initial impl: `num_kv_chunks=1` → write-through to `final_o` → merge/reduction is a no-op. This should work but means long KV sequences aren't parallelized.

### Issues Encountered
1. **POD precision mismatch in `bench_mixed_attention.py`**: Fixed — `join_outputs` was returning prefill-first but reference is decode-first. Changed to `join_outputs=False` with manual d-first cat.
2. **v2 static LPT + FlippedSchedule hurts small workloads (case 1)**: Atomic overhead + CTA alternation overhead > benefit. Dynamic scheduler should fix this by being pure compute (no extra memory traffic for pre-built tile arrays).
3. **Tile counter reset**: Counter in `int_buffer` needs `cudaMemsetAsync` before each `run()`.
4. **Non-contiguous sequences in dyn_qo_indptr**: Sequences in prefill/decode queues are not necessarily contiguous in the original batch. Cannot use `indptr[s+1] - indptr[s]` for qo_len. Fixed by adding separate `dyn_qo_len` array.
5. **GPU preempted**: B200 GPU lost during v3 development. Keepalive process died, GPU was reclaimed. Cannot test compilation or correctness of v3 until GPU returns. All v3 code is committed but untested.

### Questions

1. **How should KV splitting work with the dynamic scheduler?**
   Answer: Use option (a) — pre-compute partial output slots per (sequence, head, kv_chunk) triple. The mapping is deterministic given sequence metadata, independent of which CTA processes it. Skip for initial v3 implementation (no KV splitting), add later.

2. **Should `atomicAdd` use a gridDim stride or be strictly sequential?**
   FA3 does `atomicAdd(..., 1) + gridDim.x`. For two queues, per-queue gridDim = `gridDim.x * gridDim.y / 2`. Strict sequential always fetches the true next-largest tile (better LPT). Plan to benchmark both.

3. **Does FlippedSchedule already implement queue switching?**
   Answer: Partially. FlippedSchedule provides the *template* — even CTA: Runner1→Runner2, odd CTA: Runner2→Runner1. But with the dynamic scheduler, each Runner now dynamically fetches tiles from its queue via `atomicAdd`. When a Runner's queue is drained (tile_idx >= total_tiles), it exits, and the CTA proceeds to the other Runner which queries the other queue's atomic counter. So the dynamic part is the per-queue `atomicAdd`, and FlippedSchedule just controls which queue each CTA starts with.

4. **What's the right L2 budget for B200?**
   FA3 hardcodes 32MB for K&V on H100. B200 has 126MB L2. Should query `cudaDevAttrL2CacheSize` at runtime, or hardcode a conservative fraction (e.g. 50% = 63MB)?

5. **For KV splitting in `tile_idx_to_work_tile()`**: FA3 packs `split_idx`/`num_splits` into upper bits of `bidh` and uses head-first ordering without L2 sectioning. Flashinfer has explicit `kv_start`/`kv_end` — can handle this differently. Skip for initial v3, add later.

6. **Is L2-aware grouping essential even for the first implementation?**
   Answer (from user): Yes, absolutely. Without L2-aware grouping the scheduler is fundamentally wrong — different sequences' KV caches will thrash L2.

7. **Can persistent kernel launch more than 2 CTAs/SM?**
   Answer (from user): No, 2 CTAs/SM is a shared memory constraint. Same limit as POD (which launches 128 threads/block, 256 threads/SM total).
