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
| Default | 2836 | 145 | 8.7 | warmup=1 |
| Persistent | 2838 | 102 | 8.6 | warmup=1 |
| POD (join_outputs) | 2938 | 129 | 8.1 | warmup=25 (JIT compile) |

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

### Implementation Plan
1. Add a new scheduler mode (e.g., `LPTScheduler`) alongside `TwoStageHolisticPlan`
2. In `plan()`: sort tiles by cost descending (LPT), store as two sorted arrays (prefill queue, decode queue) + global atomic counters
3. In `PersistentKernelTemplate`: replace static `work_indptr` iteration with atomic fetch from queues
4. L2-aware grouping: compute `nheads_in_l2 = L2_size / (seq_len * head_dim * 2 * elem_size)`, only contend among those heads
5. Two CTAs per SM: CTA0 fetches from prefill queue, CTA1 from decode queue; switch when own queue is empty

### Key Findings from Investigation
1. **FlippedSchedule rarely triggers**: Condition `prefill*128 <= decode_kv_total && prefill >= 3072`. Only cases 6-7 in bench trigger it. Cases 1-5 where persistent is 2x slower than POD don't use FlippedSchedule.

2. **Enabling FlippedSchedule universally makes things worse**: Tested with `total_prefill > 0 && total_decode > 0` — persistent got slower (case 1: 0.28→0.38ms, case 4: 0.52→0.61ms). The CTA alternation overhead isn't worth it without real overlap benefit from LPT.

3. **B200 specs**: L2=126MB, 148 SMs. L2 can hold ~15 decode sequences (kv=2048, 8 heads, hd=128, bf16).

4. **Root cause of persistent slowness**: Static tile assignment via MinHeap. When a cluster gets a large prefill tile, its decode tiles must wait. No runtime load balancing.

### Files to Modify
1. `include/flashinfer/attention/scheduler.cuh` — `TwoStageHolisticPlan`: add LPT-sorted flat tile arrays + atomic counter offsets to `HolisticPlanInfo`
2. `include/flashinfer/attention/persistent.cuh` — `BlockBatchPagedAttentionPersistent::Run()`: add atomic tile fetching code path
3. `include/flashinfer/attention/persistent_template.cuh` — `PersistentKernelTemplate`: allocate/zero tile counters, pass to runners
4. `include/flashinfer/attention/persistent.cuh` — `BatchPagedAttentionPersistent()`: allocate global tile counter buffer

### Questions
1. **Grid sync with runtime fetching**: `grid.sync()` is needed before reduction. With runtime fetching, all CTAs naturally drain both queues before hitting the barrier. No issue — CTAs loop until queue empty, then fall through to `grid.sync()`.

2. **Partial output with runtime fetching**: The `partial_indptr` and `merge_o_indices` are pre-computed per work item. Since the tile *content* (q_indptr, kv ranges, etc.) is still pre-computed in plan — only the *assignment* (which CTA gets which tile) changes. The tile metadata arrays stay the same, just not grouped by cluster. Each tile still has its own `partial_indptr` slot. This should work without changes to the merge logic.

3. **L2 cache awareness**: For the initial implementation, skip L2 grouping and just do plain LPT. Add L2 awareness as a second step if LPT alone shows improvement.

4. **Two queues vs one**: The design says CTA0 fetches from prefill queue, CTA1 from decode. But with the current `PersistentKernelTemplate`, Runner1 (tile=128) runs prefill and Runner2 (tile=16) runs decode. With LPT, we want CTA0 to grab from Runner1's queue and CTA1 from Runner2's queue. The FlippedSchedule alternation in the template already does this — CTA0 runs Runner1 first, CTA1 runs Runner2 first. The key change is making each Runner's `Run()` use atomic fetching instead of static iteration.

5. **Cost function**: Keep `2*qo + kv` for now. LPT (descending sort) naturally gives larger tiles to faster CTAs.

### Progress

**v1: LPT atomic tile fetching (scheduler.cuh + persistent.cuh + batch_attention.cu)**
- Added `tile_counter` and `total_num_works` to `HolisticPlanInfo` and `PersistentParams`
- Modified `BlockBatchPagedAttentionPersistent::Run()`: when `tile_counter != nullptr`, uses `atomicAdd` to grab next tile instead of iterating `work_indptr`
- Tile counters zeroed via `cudaMemsetAsync` before each kernel launch
- **Result**: Cases 6-7 (where FlippedSchedule triggers) improved 10% (1.92→1.73ms, 3.77→3.68ms). Cases 1-5 unchanged (FlippedSchedule condition not met).
- **Next step**: The FlippedSchedule condition needs to be relaxed, AND the tile arrays need to be sorted in LPT order (descending cost). Currently tiles are still in the MinHeap-assigned order which is roughly balanced but not LPT-sorted. Need to sort the flat tile arrays by cost before writing to device.

**v2: LPT sorting + relaxed FlippedSchedule condition**
- Added `std::sort` by cost descending after `flatten()` in `TwoStageHolisticPlan`
- Changed FlippedSchedule condition from `prefill*128 <= decode && prefill >= 3072` to `prefill > 0 && decode > 0`
- **Results vs original baseline**:
  | Case | Original | LPT v2 | Improvement | POD |
  |------|----------|--------|-------------|-----|
  | 1 | 0.28ms | 0.32ms | -14% | 0.17ms |
  | 2 | 0.48ms | 0.40ms | +17% | 0.21ms |
  | 3 | 0.80ms | 0.79ms | +1% | 0.38ms |
  | 4 | 0.52ms | 0.38ms | **+27%** | 0.24ms |
  | 5 | 1.45ms | 1.31ms | +10% | 0.72ms |
  | 6 | 1.92ms | 1.79ms | +7% | 0.92ms |
  | 7 | 3.77ms | 3.26ms | +14% | 4.14ms |
- Case 7 now beats POD (3.26 vs 4.14ms). Cases 2,4 significantly improved.
- Case 1 regressed slightly — atomic overhead for small workloads.
- Still ~2x slower than POD for most cases. Root cause: the persistent kernel processes ALL tiles sequentially (prefill then decode for each CTA), while POD runs them on different SMs concurrently.

### Issues Encountered (implementation)
1. **POD precision mismatch in `bench_mixed_attention.py`**: Fixed — changed to `join_outputs=False` with manual d-first cat.
2. **`__shfl_sync` only works within a warp**: For multi-warp blocks, needed shared memory broadcast of atomic result. Used `volatile` cast of `smem_storage[0]`.
3. **Tile counter lifetime**: Counter is in `int_buffer` (set once in `plan()`). Need `cudaMemsetAsync` before each `run()` call to reset.
4. **Enabling FlippedSchedule universally hurts**: CTA alternation overhead without LPT is worse than static scheduling. Need both LPT + FlippedSchedule together.

### Questions
1. **Tile ordering**: The current MinHeap produces tiles grouped by cluster. When using atomic fetching, the tile order in the flat array matters — LPT needs descending cost order. But the `flatten()` function just concatenates per-cluster arrays. Need to either (a) sort the flat array after flatten, or (b) build the flat array directly in sorted order without the per-cluster grouping.

2. **Separate queues for prefill/decode**: The design says CTA0 fetches from prefill queue (task 0, tile=128), CTA1 from decode queue (task 1, tile=16). Currently FlippedSchedule alternates which Runner goes first per CTA, and each Runner has its own `tile_counter`. This should naturally give overlap — but only if both tasks have tiles. In cases 1-5, task 0 (tile=128) may have very few tiles while task 1 (tile=16) has many decode tiles. Need to verify the tile classification is correct.

3. **`smem_storage` reinterpretation**: Using `volatile IdType*` cast of the SharedStorage union for atomic broadcast is sketchy. The SharedStorage is a union of q_smem, k_smem, v_smem — writing to byte 0 could corrupt data if the kernel reads q_smem later. However, we do `__syncthreads()` and the atomic fetch happens at the loop top before any smem data loading, so it should be safe. The q_smem loading happens after `get_block_coord()` which is after the broadcast. Verify this doesn't cause issues.
