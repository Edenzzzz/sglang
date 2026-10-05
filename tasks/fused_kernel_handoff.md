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

## 9. The static SM split is the wrong structure — CONFIRMED on exact numerics

> **The original table here was measured in symmetric mode, which computes FA3
> wrong, so it was withdrawn. The sweep was re-run in asymmetric mode, which is
> exact (2.3e-4), and it reaches the same verdict with valid data:**
>
> | arm | tok/s | vs base | mean ITL | mean TTFT |
> |---|---|---|---|---|
> | base, fused off | 24,892.7 | — | 18.50 ms | 2,684 ms |
> | asymmetric, `sm_fa3=64` | 21,305.8 | **-14.4%** | 21.69 ms (+17%) | 3,336 ms (+24%) |
> | asymmetric, `sm_fa3=96` | 21,411.2 | **-14.0%** | 21.58 ms | 3,317 ms |
> | asymmetric, `sm_fa3=116` | 22,034.3 | **-11.5%** | 21.13 ms | 3,176 ms |
>
> **The loss shrinks monotonically as FA3 takes more of the device** -- 64 -> 96
> -> 116 gives -14.4% -> -14.0% -> -11.5% -- which is what a 93/7 chunk/decode
> work mix predicts, and it extrapolates to an optimum of "give FA3 everything",
> i.e. do not split. A static fraction has no winning setting here, with the
> correct kernel, with the corrected gate, at any split tested.


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

## 10. Serial dispatch: measured, and it is a dead end twice over

§9 showed SM pre-assignment loses, and the per-kernel times suggested the two
mainloops run back-to-back on the whole device would give 1.74x on attention
(123.7 vs 215.4 us/layer, launch cost only +1.0 us). That was a **projection from
isolated kernels, and the server measurement killed it**: two repeats of serial
dispatch, same workload, exclusive GPU, gate as corrected:

| arm | tok/s | mean ITL | mean TTFT |
|---|---|---|---|
| baseline, fused off | 24,861.9 | 18.53 ms | 2,684 ms |
| serial dispatch, run 1 | **12,923.9 (-48.0%)** | 18.73 ms | 7,255 ms |
| serial dispatch, run 2 | **12,905.5 (-48.1%)** | 18.75 ms | 7,264 ms |

Mean ITL barely moves while throughput halves and TTFT nearly triples, which is
the signature of a host-bound path, not a slower kernel: serial dispatch plans
*both* wrappers for the full device every step and issues two paged runs per
layer, and this path was already the host-bound one. The projection ignored all
of that because it timed launches with the plans held fixed.

The code is reverted rather than left behind a flag. Beyond the measurement, it
is the wrong structure for this work on principle: **spec draft and verify cannot
be fused if the dispatch is serial.** One launch holding prefill chunk, draft,
verify and decode is the claim; serialising the roles gives it up to collect a
projected gain that does not exist.

So the remaining path is the one the measurements point at from both sides: a
single launch, one queue over the whole device, the wgmma mainloop as task 3 of
`TwoStageHolisticPlan` (flashinfer `ca31e08`) with packed-GQA tiling (worth
1.70-2.40x on the short-q classes, §8), split-KV from the plan's
`kv_start`/`kv_end`, and the JIT reduction the FA2 runners already perform.

## 11. Next step, specified: the wgmma role driven by task 3 of the unified plan

The plan side is done and compiles for both task counts (flashinfer `ca31e08`).
What remains is device-side, and it is three separable pieces. Order matters:
each is measurable on its own with `test_fused_server_batch.py` (numerics) and
`pack_class_boundary.py` / `row_cmp.py` (speed).

**(a) A tile scheduler over task 3.** The FA3 mainloop's scheduler contract is
small (`hopper/tile_scheduler.cuh`): `Params` of device arrays,
`get_initial_work` / `get_next_work`, and a `WorkTileInfo` whose
`get_block_coord()` returns `{q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr,
kv_indptr, qo_len, kv_len, batch_idx}`. Build one reading task 3 of
`HolisticPlanInfo<4>`, i.e. `tasks[3].{work_indptr, q_indptr, kv_indptr, q_len,
kv_len, q_start, kv_start, kv_end, kv_head_idx, partial_indptr, group_id}`.

**Split-KV needs no mainloop change**: set `kv_indptr += kv_start` and
`kv_len = kv_end - kv_start`. The mainloop walks `[0, kv_len)` of the pointer it
is given, and this path is non-causal (`causal=false`), so a chunk is just a
shifted base with a shorter length. This is the piece to land first -- it is the
one the user is right about costing nothing to try.

**(b) Packed-GQA addressing.** The plan tiles *packed* rows
(`qo_len * gqa_group_size`), so a task-3 work item covers packed rows
`[q_start, q_start + tile)` of one kv head, spanning several qo heads. The
mainloop must map packed row `p` -> `(token = p / G, head = kv_head * G + p % G)`
for both the Q load and the O store. Q therefore cannot come through TMA as a
dense `[CTA_TILE_Q, D]` block: use the existing `FLASHINFER_SKIP_TMAQ` path
(`cp.async`), which exists precisely as this diagnostic and is how the sparse KV
path already loads. Worth 1.70-2.40x on the short-q classes (§8) and ~1.00x on
chunks, so it is what moves the verify class over -- do not expect it to help the
chunk class.

**(c) JIT reduction, shared with the FA2 runners.** When `group_id >= 0` the
epilogue writes `partial_o` / `partial_lse` at `partial_indptr` instead of `o`,
then `atomicAdd` on `group_counter[group_id]`; the work item that observes the
last arrival reduces the group's `merge_indptr` rows in place. `persistent.cuh`
around lines 240-620 is the reference, including the two bugs already paid for
there: the merge must use the *real linear tid* (not `threadIdx`) and needs a
fence before reading partials, or it intermittently merges stale data. The fused
kernel already threads an `s_merge` flag through `fused_drain_queue`, so the FA3
role joins an existing mechanism rather than adding one.

Then, and only then, the two things this exists for: every CTA pulls from one
queue over the whole device (no `sm_split` fraction, which §9 measured as the
wrong structure), and a single launch holds prefill chunk + draft + verify +
decode -- which §10 is the reason not to serialise.

Multicast (§6 of `debug_kernel_server_perf_4_mode.md`) layers on top of (c): the
DSMEM handoff measured 1.33-1.59x in isolation is the same handoff a split-KV
group needs when its chunks sit in one cluster, so the plan should prefer
assigning a group's chunks to cluster-mates. `FLASHINFER_FUSED_CLUSTER=2` still
fails on the fused kernel and the open route is incremental reconstruction from
`dsmem_kv_share.cu`, which clusters fine at 384 threads with the fused kernel's
`__launch_bounds__`; note that `compute-sanitizer synccheck` is not viable here
(3 hours without reaching a launch result, and it silently cost a day of
measurements, §7).

## 12. Why pre-assignment could not win: launch geometry, not the split

The FA2 role is crippled inside the fused launch at **every** split, for a reason
independent of how the SMs are divided:

| | FA2 native (`BatchAttention`) | FA2 role inside the fused kernel |
|---|---|---|
| CTAs per SM | 2 (`num_sm *= 2`, "two cta per sm", `scheduler.cuh:1189`) | **1** (`__launch_bounds__(384, 1)`, and ~164 KB smem forces it) |
| threads per CTA for the mainloop | ~128 | 128 of 384 (WG0-only drain) |
| **FA2 threads per SM** | **~256** | **128** |

mma.sync saturates by occupancy -- many warps issuing to hide latency -- while
warp-specialised wgmma is one CTA per SM by construction (384 threads, 3
warpgroups, ~164 KB). A single launch has one block shape and one occupancy for
the whole grid, so the two roles cannot both have native geometry in one kernel.
That is the real reason SM pre-assignment was chosen, and it is correct about the
constraint.

But pre-assignment also freezes the *work fraction*, and §9 measured that a fixed
fraction cannot fit a 93/7 chunk/decode mix that moves every step. Both
constraints are satisfiable at once, and this is the design to build:

1. Keep the uniform geometry: 1 CTA/SM, 384 threads.
2. Make the FA2 role **warpgroup-reentrant** so an SM doing FA2 work uses all
   three warpgroups -- 384 threads/SM against a native 256, which more than
   recovers the lost CTA.
3. Every SM pulls from one queue (`cluster_cursor`), so the chunk/decode split is
   dynamic per step instead of a constant.

The blocker is known and was paid for once already: the persistent mainloop
synchronises block-wide, so running it on three warpgroups deadlocked (exit 124,
GPU at 99%) and the drain was reverted to WG0-only. It needs per-warpgroup named
barriers, and the groundwork exists -- `kPersistentMainloopBarrier + 3 + kWg` are
already allocated for exactly this. Converting the mainloop's `__syncthreads()`
to those barriers is the next piece of device work, ahead of §11(b) and (c),
because without it the wgmma role's 2.63x is collected on one third of the FA2
role's threads.

### 12.1 First obstacle for warpgroup-reentrant FA2: the named-barrier budget is full

Hopper has 16 named barriers. Current allocation in the fused kernel:

| ids | owner |
|---|---|
| 0-7 | FA3 / CUTLASS reserved (`kQueryEmpty` is 0, which is also `__syncthreads()`) |
| 8 | `kFusedQueueBarrier` (`FirstUserBarrier`) |
| 9-11 | `kPersistentMainloopBarrier + 0..2`, the mdo barriers |
| 12-14 | `kPersistentMainloopBarrier + 3 + kWg`, per-warpgroup queue barriers |
| 15 | `kPersistentMainloopBarrier + 6`, symmetric-mode rendezvous |

So there is **no free id** for three per-warpgroup mainloop syncs, and
`persistent_block_sync` currently hard-codes one id with width 128. Three options,
in the order they should be tried:

1. Reuse each warpgroup's queue barrier (12-14) for its mainloop sync, after
   proving the two are never pending simultaneously inside one warpgroup -- the
   queue barrier is held while pulling work, the mainloop sync while computing a
   tile. If that holds it costs nothing.
2. Make the mdo barriers (9-11) per-warpgroup instead of per-stage, since their
   three ids are already a per-stage split that a reentrant mainloop would have
   to rework anyway.
3. Drop the symmetric rendezvous (15) and re-derive it from the queue barriers,
   freeing one id.

Do not add a fourth warpgroup's worth of ids without re-auditing 0-7: the
`__syncthreads()` == barrier 0 == `kQueryEmpty` collision is what broke symmetric
mode once already, and it is silent -- it shows up as a hang, not an error.

### 12.2 Is warpgroup-reentrant FA2 even possible? Measured: yes, by aliasing

Two things had to be checked before building it.

**Instruction scope.** `mma.sync` is a *warp*-level instruction -- its fragments
are defined across a warp's 32 lanes -- so three warpgroups can issue it
independently. This is unlike `wgmma`, which is warpgroup-scoped and needs its
own fence / commit_group / wait_group across 128 threads in lockstep. Nothing in
the ISA prevents three warpgroups each running an FA2 mainloop on its own work
item.

**Shared memory**, which is the binding constraint (`/tmp/smem_budget.cu`,
compiled for sm_90a against the JIT-generated config):

| region | bytes |
|---|---|
| FA3 `SharedStorage` (1 K/V stage) | 98,352 |
| persistent tile128 | 49,152 |
| persistent tile64 | 32,768 |
| persistent **tile16** | **69,632** (the largest class) |
| H200 per-CTA budget | 232,448 |
| 3 warpgroups x largest class | **208,896 -> fits, 89.9% of budget** |
| FA3 + 3 x largest, disjoint | 307,248 -> **does not fit** |

So three concurrent mainloop instances fit only if they **alias FA3's region**,
which is what the kernel already does by default (`FLASHINFER_FUSED_DISJOINT_SMEM`
is the opt-out). That is sound because the roles are time-multiplexed within a
CTA: an SM runs FA3 work or FA2 work, never both at once. Note tile16 is the
largest class, not tile128, so the worst case is three tile-16 instances and the
sizing must come from the max over classes rather than from tile128.

What remains for (2) in §12 is therefore mechanical rather than structural:
warpgroup-relative warp and thread indices inside the mainloop (the `ltid`
distinction the merge path already had to make), the three barrier ids from
§12.1, and three smem slots carved from the aliased region.

### 12.3 Correction to 12.1: the barrier budget was not the blocker

12.1 said three per-warpgroup mainloop sync ids were needed and none were free.
Reading the code rather than the id table: `threadblock_sync_mdo_states` is
**already** per-warpgroup -- `kPersistentMainloopBarrier + int(tid_base / 128u)`,
i.e. 9/10/11 -- and `prefill.cuh:1120` documents the exact deadlock (exit 124,
GPU at 99%) that forced it. The only syncs still hard-coded to one shared id
were `persistent_block_sync()` (8 call sites) and the JIT merge's last-arrival
broadcast.

Both now take the warpgroup's own id, reusing its mdo id rather than asking for
a new one. That is sound for the same reason `__syncthreads()` can reuse barrier
0 throughout a kernel: both are full-warpgroup rendezvous reached in program
order, so they cannot interleave within a warpgroup. No new barrier ids, option 2
of 12.1 for free, and option 3 (dropping the symmetric rendezvous) is not needed.

What actually had to change was shared memory: all three warpgroups' mainloop
storage pointed at one base, so concurrent drains overwrote each other's Q/K/V
tiles and mdo scratch while the cursor and merge flag had already been split per
warpgroup. The drain now carves one `kPersSlotBytes` slot per warpgroup (sized
from the largest tile class, tile 16 at 69632 B), moves the cursor/merge tail
past all three slots, and the launch's dynamic smem request covers
`base + 3 * slot + 48` instead of one slot's worth.

## 13. WITHDRAWN: there is no second header copy — the file was never pushed

> **This section's conclusion was wrong and is kept only because the reasoning
> error is instructive.** The 4-task plan failed to build and the diagnostic
> named the pre-refactor `TwoStageHolisticPlan` signature under
> `~/.cache/flashinfer/90a/generated/...`, which I read as flashinfer's JIT
> compiling a stale copy of the headers. It was not: that path was just the
> include chain in the error context. `grep -c "NUM_TASKS, typename IdType"` on
> the pod's own `include/flashinfer/attention/scheduler.cuh` returned **0**, and
> `find` over the venv and the JIT cache turned up **no** second copy of that
> header. The refactor had only ever existed on the local machine -- I pushed
> `persistent_fa3_fused.cuh`, `unified_wgmma_scheduler.cuh`,
> `batch_attention.cu` and the harness, and never the one header the build
> actually needed.
>
> The lesson is narrower than the one I drew, and more useful: when editing
> across two machines, verify the remote file contains the change before reading
> anything into a build error. One `grep -c` of a distinctive token would have
> replaced the whole investigation. The genuine inert-flag mechanisms recorded
> elsewhere -- env vars standing in for compile-time switches, and torch JIT
> reuse of a cached `.so` -- are unaffected and still stand.


The 4-task plan refuses to build from `BatchAttention.plan()`, and the diagnostic
names the old signature inside
`~/.cache/flashinfer/90a/generated/batch_attention_.../`:

```
RuntimeError: Ninja build failed.
  .../generated/batch_attention_.../...
  inline cudaError_t TwoStageHolisticPlan(void* float_buffer, size_t ...
```

That declaration is the pre-refactor one, without the `NUM_TASKS` template
parameter. So flashinfer's JIT does not compile `TwoStageHolisticPlan` from the
source tree on `PYTHONPATH`; it compiles against a generated/cached include tree
of its own. Two consequences, and the second is the bigger one:

1. Emitting a 4-task plan from the server path needs that copy refreshed -- purge
   `~/.cache/flashinfer/90a/{cached_ops,generated}` for the affected module, or
   point the JIT's include path at the edited tree -- before the unified queue can
   run from sglang at all.
2. **It explains the inert compile flags.** The standalone harness passes
   `-I $FI/include` explicitly in its `load()` call, so header edits reach it --
   which is why every numerics result in this document is trustworthy. The sglang
   path goes through the JIT, so a header edit there may compile against the
   cached copy instead. Any sglang-side measurement of a *header* change needs
   the cache purged first, and `grep -c` of the generated `build.ninja` for the
   expected `-D` or a sentinel is the only way to know it took.

This is the fourth distinct way a change here has silently failed to take effect,
after plain env vars for compile-time switches, torch JIT reuse of a cached `.so`,
and a stale `compute-sanitizer` process skewing every arm. The pre-flight rule in
memory covers all four: prove the change is in the binary before believing a
number.

## 14. Host side is complete; the unified path changes who the plan covers

Verified on the pod: `BatchAttention.plan()` with `FLASHINFER_FA3_TASK=1` emits a
**60-entry** 4-task plan (49 for 3 tasks), and the default asymmetric path still
computes the dumped serving batch exactly (FA3 3.4e-4, task0 2.2e-3, task2
2.9e-3) with the scheduler switch, the params wrapper and both plan-length guards
in place.

Before the unified path can be exercised, one structural consequence has to be
handled, and it is not a detail:

**Today the two plans cover disjoint row ranges.** `_plan_fused_split_inner`
builds FA3's plan over rows `[0, n_fa3_rows)` and the persistent plan over
`[n_fa3_rows, B)`, which is why the launch takes `pers_row_offset` and why the
FA2 side's q/o pointers are advanced. Under the unified plan there is **one** plan
over **all** rows, with the wgmma class and the FA2 classes as tasks inside it.
So:

- the persistent plan must be built over the whole batch, not the tail;
- `pers_row_offset` becomes 0, and the FA2 params must stop advancing q/o;
- `n_fa3_rows` stops being a row split and becomes purely informational -- row
  membership is the plan's `idx_qo_kv_len_vec[3]`, decided by
  `packed_qo_len >= FLASHINFER_FA3_MIN_PACKED_QO`;
- FA3's own `PrefillSM90Plan` call disappears from the serving path, which also
  removes one of the two per-step H2D copies that made this path host-bound.

Note what the admission threshold does to the regression batch: rows have
`packed_qo = [2968, 609, 7]`, so at the default 128 **rows 0 and 1 both join the
wgmma class** and only row 2 stays on FA2 (tile 16). That is the intended
behaviour given the measured boundary -- tile 128 and tile 64 both belong on
wgmma once packed -- but it means `test_fused_server_batch.py`'s `task[]` labels
and its `N_FA3_ROWS` assumption need updating in the same change, or the test
will mislabel correct output.

## 15. First unified run: the two work-index spaces do not line up

The unified path builds, runs, and the FA2 side of it is correct. With
`UNIFIED=1` (one plan over all rows, `pers_row_offset=0`, `-D` confirmed present
in the generated `build.ninja`), asymmetric, `sm_split=8`:

| row | packed_qo | plan task | result |
|---|---|---|---|
| 0 (qo 424, kv 4001) | 2968 | 3 (wgmma) | **nan, unwritten** |
| 1 (qo 87, kv 87) | 609 | 3 (wgmma) | **nan, unwritten** |
| 2 (qo 1, kv 14) | 7 | 2 (tile 16) | **rel err 2.8e-3, exact** |

511 of 512 q rows unwritten. So task-3 work items are produced by the plan and
executed by nobody, while the FA2 tasks run correctly from the same plan -- which
confirms the plan, the scheduler plumbing and the ext wiring, and isolates the
remaining defect to **who executes task 3**.

Two index spaces have to be reconciled:

- `TwoStageHolisticPlan` distributes work over **clusters**:
  `num_clusters = num_sm * 2 / cluster_size`, i.e. 264 entries in `work_indptr`
  at `cluster_size = 1`, because the FA2 mainloop is planned at two CTAs per SM.
- FA3's tile scheduler indexes `work_indptr[blockIdx.x]` over the **grid**, which
  is 132 CTAs, and in asymmetric mode only CTAs `< sm_split` reach FA3 at all.

So task-3 items landing in clusters 8..263 are never pulled: the CTAs that could
run them either index a different slot or never enter FA3. Nothing is wrong with
the work items themselves.

Fixes, in the order worth trying:

1. **Make the index spaces identical for a 4-task plan**: force `cluster_size = 1`
   and cap `num_clusters` to `num_sm` so cluster index == `blockIdx.x`, then give
   task 3 the cluster range `[0, sm_fa3)` through the plan's existing per-task
   ranges (`FLASHINFER_SM_PARTITION`, `FLASHINFER_SM_SPLIT`). This keeps the FA2
   tasks on the remaining range and costs the FA2 side its 2-CTAs-per-SM
   planning, which is the geometry penalty section 12 already measures.
2. Let every CTA pull its own task-3 items, which is the symmetric structure and
   is blocked by the FA3 corruption of section 12.3.
3. Keep two CTAs per SM for FA2 and have FA3's scheduler index
   `work_indptr[2 * blockIdx.x]`-style, i.e. teach it the cluster mapping rather
   than changing the plan.

### 15.1 Aligning the index spaces did not change the outcome

With a 4-task plan now indexing one slot per CTA (`num_sm *= 1`) and the wgmma
task pinned to the low cluster range (`FLASHINFER_SM_PARTITION=1`,
`FLASHINFER_FA3_SM_SPLIT_CLUSTERS=8`, both sentinels verified on the pod), the
result is unchanged: 511 of 512 q rows unwritten, tile-16 row exact at 2.5e-3.

That is informative. FA3 writing *nothing* -- rather than writing to wrong
addresses -- means `get_initial_work` returned invalid on every CTA, i.e.
`work_indptr[blockIdx.x] == work_indptr[blockIdx.x + 1]` everywhere. Wrong
packed-row addressing would still have written garbage somewhere. So the
hypothesis to test next is upstream of the scheduler: **task 3 may be receiving
no work items at all.**

Concrete next check, and it is an inspection rather than another theory: dump the
plan. `tasks[3].work_indptr` over its cluster range, plus the per-task item
counts, printed from the ext or read back from the int workspace. Three things it
will distinguish:

1. the routing branch never ran -- `FLASHINFER_FA3_TASK` not seen by the plan
   call, so `NUM_TASKS` was 3 and `idx_qo_kv_len_vec[3]` does not exist (the
   60-entry vector says otherwise, but the ext and the plan are separate calls
   and only the plan's length was checked);
2. routing ran but admitted nothing -- `packed_qo_len >= 128` not satisfied
   because `gqa_group_size` inside the plan differs from the ext's view;
3. items exist but the cluster override put them outside `[0, 8)`, e.g. because
   `sm_partition` was false in that call so the override's guard
   (`sm_partition && task_work[3] > 0`) did not fire.

Case 3 is the most likely: the override is gated on `sm_partition`, which is read
from `FLASHINFER_SM_PARTITION` inside the plan, and the default is **false** --
so if that variable did not reach the plan process the ranges stayed
`[0, num_clusters)` for every task and the MinHeap spread task 3 across all 132
clusters, of which only 8 run FA3.

### 15.2 Work items are now consumed; the fault moves to Q addressing

With the q extent presented in packed units, the same configuration stops
silently skipping and starts faulting:

```
CUDA error: operation not supported on global/shared address space
```

That is a TMA access outside its descriptor's mapped region, which is what a
packed tile index does to a Q tensormap built over token-indexed rows: tile 15 of
a 424-token row addresses row 1920, far past the descriptor. So the sequence is
now fully explained end to end --

| stage | state |
|---|---|
| plan emits task-3 items | **correct**: 116 items, clusters [0, 7] |
| items reach the FA3 CTAs | **correct** |
| tiles pass the range check | **correct** after the packed-extent fix |
| Q/O addressing | **wrong**: packed rows addressed as tokens -> TMA fault |

The remaining work is the packed row -> (token, head) map, and the fault tells us
where it has to live. FA3 loads Q through TMA, and a tensormap is a dense
strided region, so a packed tile -- whose rows are `(t, g)` pairs with `t = p / G`
and `head = kv_head * G + p % G` -- cannot be expressed as one descriptor at
gqa 7. Q must come in through `cp.async` instead, which is precisely what
`FLASHINFER_SKIP_TMAQ` already does on this path (`sparse_mainloop.cuh` sets
`USE_TMA_LOAD_KV = false` for KV for the same reason: a page gather is not a
dense region).

So the concrete remaining change is: compile the wgmma role with the Q TMA load
replaced by a `cp.async` gather that maps packed row `p` to
`(token = qo_indptr + p / G, head = kv_head * G + p % G)` for both the load and
the epilogue's store. The epilogue's partial-write path for split-KV groups rides
on the same mapping, so the two land together.

### 15.3 The packed-Q gather, concretely

The Q load sits at `hopper/sparse_mainloop.cuh:275`, one TMA `copy` issued by
`warp_idx_in_warpgroup == 0` under `elect_one_sync`, with
`shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ)` for
completion. Immediately above it the KV loads already do the thing Q now needs:
`copy_if(gmem_tiled_copy_k, k_predicate_fn, tKgKiGroup, tKsKiGroup)` over a
gathered layout, committed with `pipeline_k.producer_commit(...,
cpasync_barrier_arrive)`.

So the change has a working template in the same function. What it must compute,
for packed row `p` inside a tile of `CTA_TILE_Q` packed rows belonging to kv head
`h` of a row whose q tokens begin at `qo_indptr`:

```
token = qo_indptr + (tile_base + p) / G        // G = gqa_group_size
head  = h * G + (tile_base + p) % G
offset = token * q_stride_n + head * q_stride_h
```

with `tile_base = q_tile_idx * CTA_TILE_Q` and the row predicated off when
`tile_base + p >= packed_qo_len`. Three consequences to carry:

1. **`barrier_Q` changes kind.** TMA completion is a transaction barrier; a
   `cp.async` gather completes through `cpasync_barrier_arrive`, as the KV path
   does. The consumer's `barrier_Q.wait` has to match, which is the same
   coupling the `SKIP_TMAQ` branch notes when it says skipping the wait is
   required or the kernel hangs.
2. **The epilogue needs the identical map** for its O store, and the split-KV
   partial write needs it too, so all three move together. Getting one wrong
   shows up as misplaced rows rather than a fault, which is harder to see than
   the fault we have now -- so the regression must check row *placement*, not
   just magnitudes.
3. **`TmaTransactionBytesQ` and the Q tensormap become dead** on this path for
   the wgmma role. They stay live for the ordinary FA3 prefill entry points,
   which keep their dense descriptor, so the change has to be scoped to the
   unified role rather than applied to the shared mainloop unconditionally.

This is the whole remaining gap between the unified queue running and being
measurable: everything upstream is verified (plan emits 116 items into the right
clusters, items reach the FA3 CTAs, tiles pass the range check), and the one
fault left is this addressing.

### 15.4 The packed Q layout is affine — no gather needed

Reading `get_local_tile_tensor` (`hopper/utils.cuh:65`) changes the shape of the
remaining work. It builds Q's tile tensor as `m_tensor(_, _, head_idx)`, tiled at
`offset = qo_indptr` over `seq_len` — i.e. one head's token-major slab.

For one kv head's group, a packed row `p = t * G + g` maps to
`token = qo_indptr + t`, `head = kv_head * G + g`. **Both are affine strides**, so
the packed view is one CuTe layout with a nested row mode, not a gather:

```
shape  ( (G, seq_len), D )
stride ( (q_stride_h, q_stride_n), _1 )
```

with the base pointer at `(qo_indptr, kv_head * G)`. Flattening the row mode gives
exactly `p = g + G * t` in CuTe's left-fastest order, which is the packed index the
plan's `q_start` already uses. `local_tile` by `CTA_TILE_Q` then works unchanged.

That matters for three reasons:

1. **No `cp.async` gather and no predicate machinery** — the earlier framing
   (mirror the KV page gather) was heavier than necessary. The KV gather needs
   indirection because page ids come from an array; the packed Q view does not,
   because the head is a stride, not a lookup.
2. **TMA is not fundamentally excluded either.** A 3-mode descriptor over
   `(G, seq_len, D)` is dense and strided, so the "a tensormap cannot express
   this" conclusion in 15.2/15.3 was wrong. Keeping TMA would preserve
   `barrier_Q`'s transaction semantics and avoid the consumer-wait coupling
   entirely, which is the cheaper route to try first.
3. The epilogue's O store takes the same layout with `o_stride_n` / `o_stride_h`,
   and the split-KV partial write indexes `partial_o` by packed row already, so
   the three stay consistent by construction rather than by matching three hand
   written maps.

Correction to 15.2 and 15.3: they framed this as needing `cp.async` because a
packed tile "cannot be expressed as one descriptor at gqa 7". That is false — the
confusion was with the KV *page* gather, which genuinely cannot, since its row
indices come from an index array rather than from a stride.

### 15.5 The packed view indexes correctly but cannot be tiled with `local_tile`

Two separate facts, and only the first is settled.

**Indexing works.** A host test over a 28-head, gqa-7, token-major Q layout walks
every `(t, g)` of the nested view — shape `((G, seq_len), D)`, stride
`((head_stride, token_stride), 1)` — and every packed row lands on
`(off + t) * H * D + (kv * G + g) * D`. So `p = g + G * t` addresses exactly the
right row, with no gather.

**Tiling it does not.** Wrapping that view in `local_tile(..., tile_shape,
make_coord(_, _0{}))` fails to compile:

```
cute/layout.hpp(1572): error: static assertion failed
  "logical_divide: Too many modes in tiler."
```

`local_tile` wants a tiler whose rank matches the tensor's, and mode 0 here is
itself rank 2, so dividing it by a scalar `CTA_TILE_Q` is not the operation it
expects. The underlying reason is real rather than cosmetic: at `G = 7` a
128-row packed tile **straddles heads unevenly** — it spans 18 tokens plus 2 rows,
so no uniform `(G, n)` sub-shape describes a tile. A flat row mode is not
available either, since `G * head_stride = 7D` while `token_stride = 28D`, so the
packed rows are genuinely non-contiguous.

Two ways forward, both small:

1. `zipped_divide` the nested view and index the resulting tile mode, which is
   what `local_tile` does internally but without requiring rank agreement.
2. Skip tiling entirely: the copy already loops over rows of the tile, so give it
   the verified nested view plus `tile_base = q_tile_idx * CTA_TILE_Q` and let each
   row compute `make_coord(make_coord((tile_base + i) % G, (tile_base + i) / G), _)`.
   Predicate off rows past `packed_qo_len`. This keeps the one verified mapping as
   the single source of truth, and the same expression serves the epilogue's store.

Option 2 is preferable: it needs no CuTe divide semantics, and the mapping it uses
is the one the host test already checks.

### 15.6 The regression batch has no split-KV groups, so it isolates the packed mapping

Read back from the plan: task 3 holds 116 work items and **0 of them carry a
group id** (`group_id >= 0` count is zero, distinct groups zero). The batch's KV
lengths (4001 / 87 / 14) sit under the chunking limit, so every wgmma work item is
write-through.

Two consequences worth keeping:

1. The unified regression now running tests the **packed-GQA mapping alone**. If
   it passes, the unified queue is functionally correct for non-split batches,
   and nothing about the epilogue's partial path is implicated either way.
2. The split-KV partial write and in-kernel reduce therefore need their own test
   with a long-KV batch -- `kv_len_limit` has to be exceeded for any group to
   exist. The microbenches already have the shapes: `row_cmp.py`'s
   `500 x 32768` row is 256 KV tiles and will split.

So the remaining device work is not on the critical path to a first correct
unified run, which reorders it behind validating what is already built.

## 16. The unified queue is numerically correct

```
sm_split=8: unwritten (NaN) rows = 0/512
row 0 (FA3    qo= 424 kv= 4001) rel err 0.00020  ok
row 1 (task0  qo=  87 kv=   87) rel err 0.00145  ok
row 2 (task2  qo=   1 kv=   14) rel err 0.00255  ok
PASS: every row correct at every split, nothing left unwritten
```

The wgmma role is driven entirely by task 3 of our own plan -- packed-GQA tiling,
KV chunking and the JIT-reduction group id -- with `PrefillSM90Plan` out of the
picture. Rows 0 and 1 both execute as packed task-3 tiles (one straddling a head
boundary), row 2 on the FA2 tile-16 task, and every row lands in the right place.

Reproduce (flashinfer `e9d3ef8` + the commits following it, sglang `24dceca2d`):

```
bash tests/sync_to_pod.sh          # all sentinels must be non-zero
FLASHINFER_SM_PARTITION=1 FLASHINFER_FA3_SM_SPLIT_CLUSTERS=8 \
UNIFIED=1 SPLITS=8 FA3_PLAN_CTAS=8 \
TORCH_EXTENSIONS_DIR=<fresh dir> python tests/test_fused_server_batch.py
```

`UNIFIED=1` sets `FLASHINFER_FA3_TASK=1`, plans one pass over the whole batch,
passes `pers_row_offset = 0`, and compiles with
`-DFLASHINFER_FUSED_UNIFIED_WGMMA=1 -DFLASHINFER_FUSED_PACKED_Q=7`.

What this does and does not establish:

- **Does**: the plan, the scheduler, the packed Q load, the packed O store and the
  ext wiring are correct together, on a real dumped serving batch, at the
  measured class boundary (rows with packed_qo 2968 and 609 on wgmma, 7 on FA2).
- **Does not**: say anything about speed. Both the load and the store are per-row
  loops through registers rather than TMA or `cp.async`, chosen so the mapping
  could be validated first. The async forms are the optimisation, and they do not
  change the mapping.
- **Does not**: exercise split-KV. This batch produces 0 groups (15.6), so the
  partial write and in-kernel reduce remain untested and need a long-KV batch.

Next, in order: serving-path wiring in sglang (one plan over all rows,
`FLASHINFER_FA3_TASK=1`, the two `-D`s in the ext build, `pers_row_offset = 0`),
then the A/B against the 24,892.7 tok/s baseline with no `sm_fa3` fraction in
play, then the async copies, then split-KV on a long-KV batch.

## 17. Unified queue on the serving path: correct, and 3.2x slower

| arm | tok/s | mean ITL | mean TTFT |
|---|---|---|---|
| baseline, fused off | 24,800.1 | 18.57 ms | 2,697 ms |
| unified queue (`SGLANG_FUSED_UNIFIED=1`) | **7,693.5 (-69%)** | 60.45 ms | 10,275 ms |

The fused share is 96.2%, so the path really is running, and the standalone
regression says it is numerically exact. The loss is the copy implementation, and
the size of it is arithmetic rather than mystery:

- The packed Q load is issued by **one warp** (`warp_idx_in_warpgroup == 0`),
  which must move a 128 x 128 bf16 tile = 32 KB. Vectorised at 16 B that is 2,048
  transactions across 32 threads, 64 serialised round trips per thread, against
  **one** TMA instruction before. The O store is similar, spread over
  `NUM_COPY_THREADS` but still a per-row scalar read of swizzled smem.
- There are 116 wgmma work items per layer in this batch, so that cost is paid
  116 times per layer, 28 layers per step.

So the measurement does not indict the unified queue; it indicts a deliberately
naive copy that was written to validate the mapping. The mapping is validated.

What must change, in order:

1. **Tiled `cp.async` for the packed Q load**, issued by the whole producer
   warpgroup rather than one warp, with the gather expressed through the same
   packed view. The KV path in the same function is the template:
   `copy_if(gmem_tiled_copy_k, predicate, src, dst)` committed with
   `cpasync_barrier_arrive`. This is where the 69% lives.
2. **Tiled store for O**, same reasoning, using `tiled_copy_O` with the packed
   view as destination instead of the per-row loop.
3. Re-measure. Only then is the unified queue's real cost known, and only then
   does the comparison against -11.5% (static split) and the +20% Amdahl ceiling
   mean anything.

Recorded as a negative result with its cause rather than as a verdict on the
architecture, because the two are separable here and the evidence separates them:
same kernel, same plan, exact numerics, and a copy that is two orders of magnitude
off the instruction count it replaced.

### 17.1 Why the tiled copy needs an indirection table (or a G-aligned tile)

The obvious fix for 17 is to replace the single-warp loop with the tiled
`cp.async` the KV path uses. That needs a `(CTA_TILE_Q, D)` source tensor to
partition across the producer threads — and the packed view cannot provide one at
`CTA_TILE_Q = 128` with `G = 7`:

- address of packed row `p` is `(p / G) * token_stride + (p % G) * head_stride`,
  which is affine in `(t, g)` but **not** in `p`, because of the wrap at `G`;
- a tile is a contiguous range of `p`, so over a tile the address is piecewise
  affine with a discontinuity every 7 rows — no single stride, which is the same
  reason `local_tile` refused it (15.5);
- it *would* be expressible as `((G, n), D)` if every tile were G-aligned, i.e.
  if the tile covered whole tokens.

So two real options, and they trade different things:

1. **Indirection table, the KV pattern.** Compute 128 row offsets per work item
   into smem (cheap: 32 threads, one divmod each), then do the tiled `cp.async`
   through that table exactly as the paged KV read does through `kv_indices`.
   Keeps `CTA_TILE_Q = 128` and the existing mma shape. Costs one small smem
   array and a divmod per row, amortised over 128 x 128 elements.
2. **Make the wgmma class's tile a multiple of G.** At gqa 7 that is 126 instead
   of 128, and then the packed view tiles directly as `((7, 18), D)` with no
   indirection at all. But `CTA_TILE_Q` is a kernel-traits constant bound to the
   mma tile shape, so this changes the wgmma geometry per GQA ratio — fine for a
   fixed deployment, awkward as a general kernel.

Option 1 is the one to build: it is local to the load, uses a pattern already
proven in the same function, and does not make the tile shape depend on the
model. Option 2 is worth remembering if a single deployment ever justifies
specialising.

Note this is the same obstacle in a third guise -- 15.5 met it as `local_tile`
refusing a scalar tiler, 15.2 as a TMA descriptor fault, and now as a tiled copy
needing a partitionable source. The underlying fact each time: at gqa 7 a 128-row
packed tile straddles heads unevenly.

### 17.2 cp.async alone recovered 9%; the rest was a self-inflicted pipeline drain

| unified arm | tok/s | mean ITL | TTFT |
|---|---|---|---|
| register round trips | 7,693.5 | 60.45 ms | 10,275 ms |
| cp.async + `cp_async_wait<0>` | 8,379.5 (+8.9%) | 55.42 ms | 9,400 ms |
| baseline | 24,885.3 | 18.51 ms | 2,681 ms |

Converting the load to `cp.async` moved it 9%, which is far too little, and the
residual does not fit the copy volume: 116 work items x 32 KB is 3.7 MB per
layer, under 2 us at HBM bandwidth, against a step that is **+37 ms**.

The cause is in the completion signal I used, not the copy. `cp_async_wait<0>`
drains **every** outstanding cp.async group in the thread, and this producer has
already committed the K and V stages through `pipeline_k`/`pipeline_v` with
`cpasync_barrier_arrive`. So each of the 116 work items per layer flushed the KV
prefetch pipeline and serialised the mainloop against it. The fix is to signal
through `barrier_Q` with `cpasync_barrier_arrive`, which is precisely what the KV
loads next to it already do.

Worth stating as a rule rather than an incident: in a warp-specialised mainloop
with a software pipeline, any blanket `cp_async_wait<N>` is a cross-pipeline
barrier. Completion must be expressed per consumer, through that consumer's
barrier.

## 18. Spec path: where the CPU-GPU syncs actually are

Audited rather than assumed, because the v1 and v2 workers differ sharply.

**`eagle_worker_v2.py` (the overlap path) has none.** No `.item()`, `.cpu()` or
`tolist()` in the file. `draft()` writes the tree mask and positions straight into
the verify CUDA-graph buffers with `build_tree_kernel_efficient`, returns
`seq_lens_sum=None, seq_lens_cpu=None`, and plans verify on a side stream the
compute stream waits on. Verify metadata is already device-resident there.

**`eagle_worker.py` (v1) had one on the hot path**, now fixed:

| site | what it was | status |
|---|---|---|
| `:454` draft step | `torch.sum(batch.seq_lens).item()` — a device sync **every draft step** | **fixed**: reads the `seq_lens_cpu` mirror |
| `:926-928` draft extend | already prefers `seq_lens_cpu`, falls back to a device sum | fine |
| `:643-647` verify | `retrive_next_token.cpu()` and friends — but only under `if batch.has_grammar` | not on the default path |
| `:310` | `global_need_forward[0].item()` after an all-reduce on the **CPU** group | not a device sync |
| `:812-813` | `.tolist()` on logprobs and verified ids, at output assembly | after the step, not in it |

So the structural claim behind a 4-mode schedule holds: with
`--speculative-eagle-topk 1` the draft is a chain, every row grows by exactly
`k+1`, and the verify metadata is length-deterministic — it needs nothing from
the draft's output except the token ids, which stay on device through the future
map. The remaining blocker to measuring it is not a sync but
`mix_with_running` crashing when the running batch is a spec batch.

## 19. Schedule quality on a 4-mode batch — the three properties, measured

`tasks/fused_specdec_workload/schedule_quality.py`, read back from the plan the
kernel runs. Batch: 1 chunked-prefill row (500 over a 2595 prefix), 8 verify rows
at k=3, 8 draft rows, 11 decode rows, gqa 7, 132 clusters.

**Near-zero tile waste** (occupied / allocated tile rows):

| tiling | all four modes | verify class alone |
|---|---|---|
| unpacked, as `PrefillSM90Plan` tiles it | **13.9%** | **6.2%** |
| packed, as the unified plan tiles it | **54.8%** | **43.8%** |

3.9x less waste overall and 7.1x on the verify class, which is the mode the
unpacked tiling hurts most: a 4-token verify row occupies 4 of 128 tile rows and
pays that 28 times, once per qo head, against 28 of 128 in a single packed item.

**Near-perfect load balance**, with a caveat that matters:

| task | items | live clusters | imbalance (max/mean) | idle |
|---|---|---|---|---|
| 0 (tile 128) | 64 | 64 | **1.000** | 68 |
| 2 (tile 16) | 76 | 76 | **1.000** | 56 |
| 3 (wgmma) | 112 | 8 | **1.000** | 124 |

Every live task is balanced exactly. But no single task fills the device -- 68,
56 and 124 clusters sit idle respectively -- which is the quantitative reason the
classes have to **co-reside** rather than each owning a slice, and why the
measured static split loses at every ratio (section 9). The union saturates; no
member does.

**Smaller sync surface**: unified pays 1 plan, 1 H2D and 1 launch per step. The
split design pays 2 plans and 2 H2D copies and must keep two schedules mutually
consistent; serial dispatch pays 2 of each plus a second launch, measured at -48%.
On the spec side, the draft step's device sync is now gone (section 18) and v2's
verify metadata is already device-resident, so a 4-mode step needs no host round
trip between draft and verify.

### 19.1 Next: the tree build as an optional epilogue task

The JIT-reduction framing makes this natural. A split-KV group's last arrival
already runs a *second kind of work* in the epilogue -- the reduction -- chosen per
work item rather than per kernel. `build_tree_kernel_efficient` is the same shape
of thing: it consumes the verify group's output and produces the next step's tree
mask and positions, it is data-parallel over rows, and today it is a separate
launch between draft and verify.

So represent it as an **optional epilogue stage**, keyed off the work item the way
`group_id >= 0` keys the reduction:

- the plan marks the verify class's groups with a flag meaning "after reducing
  this group, build its tree";
- the last arrival, already holding the reduced O and LSE in registers and smem,
  writes the mask and positions into the same CUDA-graph buffers
  `get_verify_buffers_to_fill_after_draft()` hands out today;
- nothing else changes: the reduction's completion ordering is the ordering the
  tree build needs, so no new barrier appears.

What this buys, in the paper's terms: the tree build stops being a kernel boundary
and becomes a task, so the 4-mode step is one launch with no host round trip -- the
property section 18's audit shows is otherwise already within reach but still
split across two kernels.
