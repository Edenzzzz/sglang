# Fused 4-mode attention kernel: interface, workloads, and the design points

Written to be handed off: either to iterate kernel performance against, or to a
different agent asked to write the kernel from scratch. Everything here is
measured on an H200 (132 SMs) with the GPU exclusively held, Qwen2.5-7B head
shape 28/4/128 (gqa 7), bf16, page_size 1, unless stated otherwise.

Answers to the two prerequisite questions, up front and honestly:

| prerequisite | status |
|---|---|
| stable, CUDA-graph-compatible kernel interface | **No.** The launch interface is stable enough to iterate on (§3) but is *not* CUDA-graph capturable, and nothing captures it today (§4). |
| all 4-mode mixed workloads from real online serving | **Partly.** Real online traces exist for 2 modes (chunked prefill + decode), per-step and complete (§5.1). The 4-mode sets that include draft/verify are *constructed*, not captured from a live MTP server (§5.2). |

## 1. What a kernel win is worth here (read this before optimizing anything)

Measured decomposition of one median MIXED serving step
(`tasks/fused_specdec_workload/attn_share.py`):

| component | per layer | x28 layers | share of the 18.96 ms step |
|---|---|---|---|
| prefill row, 500 q over 2595 prefix (kv 3095) | 200.2 us | 5.605 ms | **29.56%** |
| 11 decode rows, 4357 kv each | 48.5 us | 1.359 ms | 7.17% |
| whole mixed batch | 215.1 us | 6.022 ms | **31.76%** |

So attention is ~32% of a mixed step, and **93% of that attention is one prefill
row**. Consequences, which are arithmetic, not opinion:

- 1.5x on attention -> ~+11% end-to-end throughput.
- 2.6x on attention -> ~+21%.
- The published bar (>=30% TTFT/ITL, >=10% throughput) needs >=1.5x on attention
  for throughput and ~2.5-3x for the latency axes, *on this workload*. Workloads
  with longer contexts move the 32% up and make the bar easier; short-context
  ones make it unreachable by any attention change.

## 2. The baseline to beat, per row class

`tasks/fused_specdec_workload/row_cmp.py`, one row, both mainloops planned from
the same rows, full device each:

| row (qo x kv) | persistent (FA2 lineage) | FA3 wgmma | FA3 wins by | pers TFLOP/s | FA3 TFLOP/s |
|---|---|---|---|---|---|
| 500 x 3095 (**the modal row**) | 194.5 us | 74.0 us | **2.63x** | 114 | 300 |
| 500 x 8192 | 304.5 us | 148.1 us | 2.06x | 193 | 397 |
| 500 x 32768 | 939.2 us | 499.8 us | 1.88x | 250 | 470 |
| 128 x 3095 | 236.0 us | 73.2 us | 3.22x | 24 | 78 |
| 2048 x 3095 | 362.9 us | 215.0 us | 1.69x | 250 | 423 |

Two things to take from this. The wgmma mainloop wins at **every** kv length, by
*more* at short kv. And both mainloops are far from the machine: 300 TFLOP/s is
the best cell here, so there is headroom beyond picking the better of the two.

Decode-shaped rows (qo_len 1-4) are the one place the persistent kernel wins,
because it splits KV and packs GQA and the wgmma path as planned today does
neither (§6.2).

## 3. The kernel interface as it stands

One launch, `fused_run` in `flashinfer/tests/fused_fa3_persistent_ext.cu`,
consuming two already-planned workspaces. Argument contract:

```
fused_run(q, k_cache, v_cache, kv_indices, o,
          fa3_int_ws, fa3_plan,            # PrefillPlanSM90Info ToVector()
          pers_int_ws, pers_float_ws, pers_plan,   # HolisticPlanInfo ToVector()
          num_qo_heads, num_kv_heads, page_size, sm_scale,
          sm_split,                 # blockIdx.x < sm_split start on the FA3 role
          pers_row_offset,          # FA2 rows begin here in q/o
          fa3_num_ctas,             # MUST equal the CTA count the FA3 plan was built for
          pers_kv_page_offset,      # FA2 pages begin here in kv_indices
          fa3_kv_indices,           # vector-sparse OFFSETS, not page ids
          pers_o)
```

Invariants that are load-bearing (each was a real bug):

1. `fa3_num_ctas == sm_split` in asymmetric mode; `== num_sm` in symmetric mode,
   where every CTA owns an FA3 slice and `sm_split` only decides who *starts*
   there. Violating this silently mis-schedules.
2. The FA2 side's pages start after FA3's, expressed in its **indptr** (global
   `ki_r`), never by slicing the index array: a sliced base is 16-byte aligned
   only when the page count is a multiple of 4, and the mainloop gathers indices
   with vectorised loads (`misaligned address`, server down).
3. FA3 takes **vector-sparse offsets**, not page ids; pass the output of
   `block_sparse_indices_to_vector_sparse_offsets`. Passing raw page ids leaves
   FA3 rows wrong by ~60% of output magnitude while the same kernel is exact
   standalone.
4. KV cache rank is not fixed: rank-4 `[pages, page_size, H, D]` (harness) and
   rank-3 `[tokens, H, D]` (sglang, page_size 1) both occur. Derive strides with
   `kv_strides()`; assuming rank 4 makes `cudaLaunchKernel` itself fail.
5. Named barriers: `kFusedQueueBarrier = FirstUserBarrier (8)`,
   `kPersistentMainloopBarrier = 9`, per-warpgroup queue barriers at `9 + 3 + wg`,
   symmetric rendezvous at `9 + 6`. `__syncthreads()` is `barrier 0`, which
   collides with FA3's `kQueryEmpty` -- that collision was the symmetric-mode bug.
6. Shared memory: FA3's `SharedStorage` at one K/V stage (~96 KB of a 163920-byte
   budget) plus the persistent mainloop's, sized as the max over all runners.
   `FUSED_FA3_STAGES=2` costs 128 KB and leaves no room for a disjoint FA2 region.

## 4. CUDA graphs: what is true today

- sglang graph-captures **pure decode** batches only. Mixed batches -- the ones
  this kernel exists for -- are not captured, so the fused path has never run
  under a graph.
- The path is **not** capturable as written. Per step it runs numpy routing on
  the host, two plans with two H2D copies, and mutates
  `FLASHINFER_PERSISTENT_MAX_CLUSTERS` / `FLASHINFER_FA3_MAX_CTAS` in
  `os.environ` around the plan calls. Host work and env mutation cannot be
  captured, and re-planning is the dominant host cost of the path.
- What a capturable interface requires: the plan must be device-resident and
  produced by a device kernel or reused across steps (the existing `st["key"]`
  cache is the seed of this); no `os.environ` writes in the launch path; fixed
  workspace pointers; one graph per (shape class, split) with the split chosen
  from the graph's own shape rather than per step; and
  `init_forward_metadata_capture_cuda_graph` / `..._replay` implemented on the
  backend for mixed batches, which currently exist only for decode.
- The grid-shrink design does not block capture: a task's cluster range is a
  plan-time number, so one graph per shape suffices rather than one per ratio.

## 5. The workloads

### 5.1 Real online serving, 2 modes (complete, use this)

`SGLANG_STEP_TRACE=1` on the scheduler emits one line per step, with the row
classes separated the way sglang actually carries them -- a mixed batch holds its
decode rows as extend rows of length 1, so the split is on that length, not on
the batch's mode:

```
STEPTRACE MIXED et=512 bs=12 np=1 ptok=502 pf=2595 nd=11 dkv=47935 kv=51032 mx=502 gpu=18.9612
   et   extend_num_tokens      np    prefill rows (extend len > 1)
   bs   batch size             ptok  their new tokens      pf  their prefix total
   nd   decode rows (len 1)    dkv   their KV total        kv  total KV
   mx   longest chunk          gpu   the step's own GPU span, CUDA events read a
                                     step later so nothing synchronises
```

Cost of the instrumentation: 0.02% (24,668 vs 24,431 tok/s traced/untraced).

Captured corpus: 2121 steps per arm (1901 MIXED, 197 DECODE, 23 EXTEND) at
`--random-input-len 8192 --random-output-len 192 --num-prompts 200
--request-rate 12 --max-concurrency 32`, chunked-prefill 512, 64 max running,
radix cache off. Distribution of the MIXED steps:

| feature | p10 | p50 | p90 |
|---|---|---|---|
| batch size | 6 | 12 | 15 |
| prefill rows | 1 | 1 | 2 |
| chunk tokens | 498 | 501 | 507 |
| prefix behind them | 426 | 2595 | 6221 |
| decode rows | 5 | 11 | 14 |
| decode KV total | 16818 | 47935 | 58786 |

Fit targets and six model classes (linear through gated MoE) are in
`tasks/perf_model/fit_steptrace.py`.

### 5.2 4-mode sets including draft/verify (constructed, not captured)

`tasks/fused_specdec_workload/sets/*.jsonl` and `fused_batches_k{3,8}.jsonl`
carry per-step `{draft, verify, prefill, running, kv_used}` for several targets
(`mtp_h200`, `qwen_small_h800`, `qwen_large_h200_tp8`, `qwen_large_gb200_tp16`).
They are generated from measured length distributions, not dumped from a live
MTP server, and `WORKLOAD_SETS.md` states the k-per-model constraint that makes
the verify class occupy its own tile: verify owns tile 64 only while
`16 < (k+1)*r <= 64`, so r=7 needs k=3 and r=16 needs k=1.

**The gap to close:** a real online MTP trace. Nothing in this repo yet contains
draft/verify/prefill/decode co-resident in one step as a live server produced it.

## 6. The design points, with what each is worth

### 6.1 Mixed tiles

Four scheduling classes over three tile shapes: `large_prefill` (128),
`spec_verify` (64), `mixed_decode` (128), `spec_draft` (16). Draft shares tile 16
with decode deliberately -- draft-only work saturates HBM at 85-91% of peak from
~8 rows, so a dedicated low-SM dispatch throttles it, and one shared queue is
4-7% cheaper than separate launches. The third class (64) is a real win only once
it carries work: below ~100K total KV in the class, giving it its own queue costs
every CTA an extra pass, so it folds into 128 (`FLASHINFER_TRI_TILE_MIN_KV`).

### 6.2 No CPU-GPU sync for spec metadata -- keep it on the GPU

This is the e2e lever for MTP, and most of it is already true in the v2 path:
`eagle_worker_v2.draft()` writes the tree mask and positions **directly into the
verify CUDA-graph buffers** on device (`build_tree_kernel_efficient`), returns
`seq_lens_sum=None, seq_lens_cpu=None`, and plans verify on a side stream that
the compute stream waits on; there is no `.item()` / `.cpu()` / `tolist()` in
that file at all. The v1 path still synchronises:
`eagle_worker.py:310` (`global_need_forward[0].item()`), `:417`, `:454`,
`:643-647` (`retrive_next_token.cpu()`, `retrive_next_sibling.cpu()`),
`:812-813`, `:926-928`.

The structural reason it *can* be sync-free: with `--speculative-eagle-topk 1`
the draft is a chain, so the verify metadata is length-deterministic -- every row
grows by exactly `k+1` regardless of what the draft produced. Only the token ids
depend on draft output, and those stay on device through the future map. So a
4-mode schedule can build the whole verify plan before the draft finishes.

### 6.3 Overlapping tile classes for FA2, instead of partitioning

Two mechanisms, both in `TwoStageHolisticPlan`:
- `FLASHINFER_SM_PARTITION` gives each task its own contiguous cluster range,
  sized by its share of the work; `=0` restores one shared heap over all
  clusters, i.e. classes overlap and time-share SMs.
- A device-side `cluster_cursor` lets any block, in either role, pull more
  persistent clusters once its own queue is empty. This is what lets an SM that
  finished FA3 backfill the tile-16 queue instead of idling, and it is the
  property a static split gives up. Measured end-to-end at `sm_fa3=8`:
  symmetric backfill 9,990 vs asymmetric 6,279 tok/s (**1.59x**), mean ITL 46.08
  vs 73.05 ms, p99 ITL 63.44 vs 170.84 ms, on identical batch shapes.

JIT reduction is what makes this possible without a grid barrier: a split-KV
group is registered per (q tile, kv head), and the last work item to arrive
reduces the group in place (`group_counter`, `merge_indptr`, `partial_o/lse`),
so no CTA waits for the whole grid and the merge overlaps with other SMs still
in their mainloops.

### 6.4 Load balancing

Per-task `MinHeap` over that task's cluster range, work counted in
(q tile x kv tile) pairs, plus `kv_len_limit` chunking sized by the clusters the
task actually owns -- sizing it from all clusters makes a task with a quarter of
the SMs under-split its KV by 4x. Work in MACs is the wrong currency across
classes (large-tile tasks are compute-bound, small-tile bandwidth-bound), which
is what `FLASHINFER_SM_TASK_RATE` scales and `FLASHINFER_SM_SPLIT` overrides so
the optimum can be measured instead of assumed. Measured: the analytic balance
point returns 115-131 SMs for FA3 on real batch shapes, which is one of the two
*worst* settings end-to-end -- that cost model was fit for the overlap kernel and
does not transfer.

## 7. Claims in this repo that measurement has killed (do not inherit them)

- `mixed_dispatch.py`'s docstring said the FA3 role "runs the wgmma mainloop with
  GQA packing and split-KV". **False for the path the server uses.**
  `PrefillSM90Plan` tiles `qo_len` (not `qo_len * gqa`), emits one work item per
  qo head, carries the full `kv_len`, and has no partial-O/LSE or merge map.
- The gate's `MIN_KV_TILES = 32` was never measured and was backwards: it
  excluded the modal prefill row (kv 3095 = 25 tiles), i.e. 93% of a step's
  attention, from a mainloop that runs it 2.63x faster. Removing the KV floor
  moved the fused dispatch share from 90.6% to 96.2%.
- "The fused kernel is harder to perf-model, so EDF gets worse": an artefact of
  fitting a 3-parameter line to a cost containing `max(FA3, FA2)`, on a target
  contaminated by arrival gaps. With the step's own GPU span and six model
  classes, the arms are at parity (~1.5% MAPE; grad boosting 1.49% fused vs
  1.58% baseline).
- Any absolute throughput number measured on 2026-10-05 before 06:48 is void: a
  detached `compute-sanitizer synccheck` held the GPU at 100% for 2h54m and cost
  every arm 2.45x while staying consistent with its own control. Check
  `nvidia-smi --query-compute-apps` before measuring; the driver now refuses to
  start if it is non-empty.

## 8. Which class belongs on which mainloop, measured with packing

The question "why not give tile 64 and tile 128 to the wgmma mainloop" had no
honest answer while FA3 ran unpacked, because unpacked FA3 loses the short-q
classes for a reason that is an artefact of the missing feature. Packing emulated
at the batch level (`pack_class_boundary.py`: each KV head's group presented as
one head with `qo_len * gqa` rows, legal because the cached-prefix part is
non-causal), same head shape, full device per arm:

| class | qo x kv, rows | persistent | FA3 unpacked | FA3 packed | packing worth | winner |
|---|---|---|---|---|---|---|
| decode / draft | 1 x 4357, 11 | **48.6 us** | 224.7 us | 93.6 us | 2.40x | **persistent, 1.9x** |
| verify k=3 | 4 x 4357, 8 | 96.4 us | 159.6 us | 93.4 us | 1.71x | tie (1.03x) |
| verify k=7 | 8 x 4357, 8 | 137.7 us | 159.4 us | **93.6 us** | 1.70x | **wgmma, 1.47x** |
| small chunk | 128 x 3095, 1 | 236.1 us | 73.3 us | 73.1 us | 1.00x | **wgmma, 3.23x** |
| modal chunk | 500 x 3095, 1 | 195.0 us | 74.6 us | 74.0 us | 1.01x | **wgmma, 2.63x** |
| long-ctx chunk | 500 x 32768, 1 | 940.1 us | 499.2 us | 496.2 us | 1.01x | **wgmma, 1.89x** |
| big chunk | 2048 x 3095, 1 | 362.6 us | 215.6 us | 214.5 us | 1.01x | **wgmma, 1.69x** |

Read off it:

- Packing is worth **1.70-2.40x on the short-q classes and nothing (1.00-1.01x)
  on the chunks**, which is exactly what a tile-occupancy fix should look like.
- **Tile 64 (verify) belongs on the wgmma mainloop once packed**: unpacked it
  loses (96.4 vs 159.6 = 0.60x), packed it wins at k=7 (1.47x) and ties at k=3.
  The previous "keep 64 on FA2" conclusion was an artefact.
- **Tile 128 (all chunk classes) belongs on the wgmma mainloop** by 1.69-3.23x,
  with no packing needed -- those tiles are already full.
- **Tile 16 (decode and draft) stays on FA2**, 1.9x, even packed. Those rows are
  bandwidth-bound, and FA2's split-KV + LPT is the right engine for them. So the
  FA2 overlap machinery is still wanted -- scoped to the 1-token class.

## 9. The static SM split is the wrong structure (measured, not argued)

With the gate corrected so the modal chunk reaches the wgmma role (fused dispatch
share 90.6% -> 96.2%), the serving A/B got **worse**, at every split:

| arm | tok/s | vs base | mean ITL | vs base | mean TTFT | vs base |
|---|---|---|---|---|---|---|
| baseline, fused off | 24,861.9 | — | 18.53 ms | — | 2,684 ms | — |
| fused, `sm_fa3=32` | 23,245.9 | -6.50% | 19.91 ms | +7.44% | 2,964 ms | +10.40% |
| fused, `sm_fa3=64` | 23,399.6 | -5.88% | 19.78 ms | +6.73% | 2,927 ms | +9.06% |
| fused, `sm_fa3=96` | 22,949.7 | -7.69% | 20.18 ms | +8.91% | 3,017 ms | +12.38% |

The arithmetic of why, which generalises past this workload: the modal mixed step
is 93% chunk attention and 7% decode attention. A row's wgmma work is
`h_qo * ceil(qo_len/128)` work items -- 112 for the modal chunk -- so on 64 CTAs
it takes two waves and the isolated 74 us becomes ~150 us, while the decode rows
run on the remaining 68 CTAs instead of the device that made them cost 48 us.
**No fixed fraction can be right when the mix is 93/7 and varies per step.**

So SM pre-assignment is not the design that collects the 2.6x. The design that
can is one queue over the whole device with the mainloop chosen per tile class --
which is what §6.3's `cluster_cursor` backfill approximates and what the
4-task `TwoStageHolisticPlan` (flashinfer `ca31e08`) exists to feed: the wgmma
class becomes task 3 of the same plan, inheriting packed-GQA tiling, split-KV and
the JIT-reduction group, and every CTA pulls from the shared queue rather than
owning a slice.
