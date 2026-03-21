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

**v3: FA3-style runtime dynamic scheduler (TODO)**
- [ ] Implement `tile_idx_to_work_tile()` with batch-find while loop + L2-aware LPT mapping
- [ ] Modify `TwoStageHolisticPlan` to store per-sequence metadata for dynamic scheduler
- [ ] Add per-sequence `num_m_blocks_ptr`, `nheads_in_l2_ptr` to PersistentParams
- [ ] Queue switching: CTA drains own queue, then switches to other queue
- [ ] Explore gridDim stride vs strict sequential atomicAdd
- [ ] Handle KV splitting in the dynamic scheduler (partial output slots)

### Issues Encountered
1. **POD precision mismatch in `bench_mixed_attention.py`**: Fixed — `join_outputs` was returning prefill-first but reference is decode-first. Changed to `join_outputs=False` with manual d-first cat.
2. **v2 static LPT + FlippedSchedule hurts small workloads (case 1)**: Atomic overhead + CTA alternation overhead > benefit. Dynamic scheduler should fix this by being pure compute (no extra memory traffic for pre-built tile arrays).
3. **Tile counter reset**: Counter in `int_buffer` needs `cudaMemsetAsync` before each `run()`.

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
