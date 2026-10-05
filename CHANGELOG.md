# Changelog

## v0.7.6 — Two findings that invalidate measurements, not numbers (2026-10-05)

### The wgmma role's split-KV path is a latent wrong answer

Recorded as "unexercised" before, which was too kind. `scheduler.cuh`'s chunking
loop covers every task including `kFA3Task`, the unified scheduler carries
`partial_indptr`/`group_id` per work item, and `persistent_fa3_fused.cuh`
references neither -- nor anything else about partials -- so each chunk of a
split task-3 row writes its own slice straight to the final O rows and the
chunks overwrite each other. Fixture 2 is quiet about it only because its
FA3-class row is `kv=236` and never splits.

Corrected after measuring: I expected this to break the 8192 workload too, but
the unified arm ran all 200 prompts with the guard compiled in, so nothing
split there and the earlier unified numbers are slow rather than wrong. Whether
that is because no task-3 row exceeds its `kv_len_limit` or because the guard is
inert is still open -- the gate fixture decides it, and a guard that has never
fired proves nothing.

### The restored pod's serving tree was upstream sglang, not our fork

`/dev/shm/w/sgl_src` is a clone of upstream at `1818db4`, and the tree on
`PYTHONPATH` was built from it with four of our files copied on top. Against an
md5 manifest of our fork's 725 `.py` files: 4140 files on the pod, 3573 extra,
158 missing, 514 present in both with different contents -- `bench_serving.py`
among them. Fixtures are unaffected (they import `flashinfer` and `torch`, never
`sglang`), so fixture 2's pass and the `barrier_Q` diagnosis stand. Both forks
are private and this machine has no push credentials, so the repair is a chunked
webshell transfer of the real tree. Details, and the torch-ABI dependency chase
that preceded it, in `tasks/debug_kernel_server_perf_4_mode.md`.

## v0.7.5 — Q's drain stops draining K/V (2026-10-05)

### The packed-GQA Q load is now issued before the first K tile

The packed-GQA Q load on the sparse path finishes with a cp.async group drain
(`cp_async_fence` + `cp_async_wait<0>`) and then a single-thread arrive on
`barrier_Q`. That drain is what made the correct unified kernel ~3x slow
(8,379 against 24,885 tok/s), and the reason was mis-stated in the code: the
comment claimed narrowing the wait needed the KV pipeline's outstanding group
count.

There is no such count to learn. The KV copies signal completion through
`cpasync_barrier_arrive`, the mbarrier mechanism, and are never committed to a
cp.async group at all -- so `cp.async.commit_group` sweeps them, still
uncommitted, into the very group it closes for Q. Committed groups also retire
in order, so no `wait_group N` bound can retire Q's group while an older group
is outstanding. Both facts point the same way: the only way for Q's drain to
cover Q alone is for nothing older to be in flight when it runs.

So the Q load moves above the last-k-tile load, as a `load_q_tile` lambda
called from one of two sites (`FLASHINFER_FUSED_Q_LATE` restores the old
ordering for A/B). A per-lane `cpasync_barrier_arrive` on `barrier_Q` was the
other candidate and cannot work: that instruction tracks only the cp.async of
its own issuing thread, so all 32 lanes would have to arrive on a barrier whose
arrival count is 1 -- which is exactly the 1500 s hang already paid for once.

Compile-checked in three configurations (plain, packed-GQA, and `Q_LATE`).
Not yet measured on device: the serving A/B is what decides whether the ~3x
comes back, and it is reported in
`tasks/debug_kernel_server_perf_4_mode.md`.

### Fix: the lambda no longer captures structured bindings

Hoisting the block into a lambda made it capture `q_tile_idx`, `kv_head_idx`,
`qo_indptr`, `qo_len` and the Q tensors, all of which come from structured
bindings -- capturing those is C++20 and this builds as C++17 (nvcc only warns).
They are passed as init-captures over plain aliases instead, keeping the
original names so the body is unchanged.

### Provisioning: install sglang's dependency closure from the released package

Hand-listing dependencies cost several server-start cycles (pynvml, IPython,
`sgl_kernel`, one at a time), and `sgl_kernel` cannot come from a source tree.
`drivers/install_deps.sh` installs `sglang[srt]` for the closure and then
re-pins `torch==2.8.0+cu128`, which is mandatory rather than cosmetic: that
install drags torch forward, the fused extension needs 2.8.0, and a drifted
torch would quietly make measurements incomparable.

## v0.7.4 — One Scheduler for Both Roles (2026-10-05)

### Refactor: the wgmma role becomes a task in our own plan, not `PrefillSM90Plan`

The FA3 role was planned by flashinfer's `PrefillSM90Plan`, which measurement
showed cannot serve this kernel: it emits one work item per (q-tile x qo head x
row) carrying the **full** `kv_len`, with no partial-O/LSE buffers and no merge
map. Concretely, as the server planned it, the wgmma role had

- **no GQA packing** — `num_qo_tiles = ceil_div(qo_len, cta_tile_q)`
  (`scheduler.cuh:939`) plus upstream's own note that FA3 "do[es] not fuse query
  and group heads", against `qo_len * gqa_group_size` in the persistent plan
  (`scheduler.cuh:1199`);
- **no KV splitting** — one CTA walks a whole KV range.

At the measured serving shapes (median 11 decode rows totalling 47,935 KV, one
500-token prefill chunk over a 2,595-token prefix) that leaves the role almost no
parallel width, which is why fusing it changed a mixed step by -0.21% and
end-to-end throughput by -0.65%: parity was the *expected* result, not a tuning
failure.

`TwoStageHolisticPlan` is now generic in its task count. A 4-task plan adds the
wgmma class as task 3, which therefore inherits, with no new machinery:

- packed-GQA tiling (every class in this plan tiles packed rows),
- split-KV via the existing `kv_len_limit` chunk loop,
- a JIT-reduction group (`group_id`, `partial_indptr`, `merge_indptr`,
  `group_counter`) so partials are merged in-kernel by whichever work item
  arrives last, with no grid barrier and no second launch.

A 3-task `plan_info` selects exactly the previous behaviour, so `BatchAttention`
cannot silently drop task-3 work. Admission is `packed_qo_len >=
FLASHINFER_FA3_MIN_PACKED_QO` (default 128 = fills one packed tile), tunable for
the boundary sweep rather than asserted. Both instantiations compile for sm_90a.

### Measured: the symmetric flag is worth 1.59x on the serving path
At `sm_fa3=8`, asymmetric (8 CTAs own all FA3 work) vs symmetric: 6,279 vs
9,990 tok/s, mean ITL 73.05 vs 46.08 ms, p99 ITL 170.84 vs 63.44 ms, on identical
batch shapes (p10/p50 step time matched to 2%; only the tail moved). Previously
this was a kernel-level number (0.687x -> 1.003x of monolithic) only.

### Measurement hygiene: a stale diagnostic invalidated a day of numbers
A detached `compute-sanitizer synccheck` from the CLUSTER=2 investigation was
still resident after 2h54m at 100% CPU, holding 2,760 MiB. Every arm measured
alongside it ran 2.45x slow (fused 9,990 vs 24,431 tok/s on the same command) and
*stayed consistent with its own control*, so no A/B looked wrong. Killing it
restored yesterday's scale. The measurement driver now refuses to start while
`nvidia-smi --query-compute-apps` is non-empty, and synccheck is recorded as
non-viable on this kernel (3 hours without reaching a launch result).

### Per-step trace carries features and GPU time
`SGLANG_STEP_TRACE=1` now logs per-step row classes (prefill rows, chunk tokens,
prefix, decode rows, decode KV, total KV, max chunk) and the step's own GPU span
from CUDA events read a step later, so nothing synchronises on the measured path.
Cost: 0.02% (24,668 vs 24,431 tok/s traced/untraced), which corrects the earlier
claim that these runs were "depressed by per-step logging".

## v0.7.3 — Hopper 4-Mode Fused Kernel: Multicast Tested, Doc Split (2026-10-05)

### Docs: split by effort, with lineage
`tasks/debug_kernel_server_perf.md` had reached 3428 lines spanning two unrelated
efforts. Split:

| doc | covers |
|---|---|
| `debug_kernel_server_perf.md` (1783 lines, closed) | Parts 1-4: sync-free plan, measured mixed-kernel gate, workload stats, mixed-tile 4-mode *dispatch* |
| `debug_kernel_server_perf_4_mode.md` (new, active) | the fused FA3 + FA2 SM pre-assignment **kernel**: correctness, server integration, split/cluster-cap tuning, backfill, multicast |

Each carries a lineage header pointing at the other, and the old file ends with a
forward pointer. Corrections to earlier claims are kept in place in both rather
than edited away — several conclusions were reversed by later measurement and the
reversals are the useful part.

### Measured: multicast tested on hardware (flashinfer `4eb41c6`, `88fc805`)
Multicast had been written off as needing a `sparse_mainloop.cuh` gather→TMA
rewrite. That conflated multicast with *TMA* multicast: Hopper DSMEM gives the same
load-once-consume-in-N-CTAs property with no tensormap and no KV-pool change, via
`cg::cluster_group::map_shared_rank` (already used in-tree by
`trtllm/fused_moe/RoutingKernel.cuh`).

`tests/dsmem_kv_share.cu`, built to match the redundancy this kernel has — CTAs
paired on the same KV tile, since every q-tile of a prefill row re-gathers the same
KV:

| working set | redundant gather | DSMEM shared | speedup |
|---|---|---|---|
| 16 MB (**L2-resident, in-situ**) | 682.436 us | 514.583 us | **1.326x** |
| 256 MB (cold) | 1325.362 us | 835.079 us | 1.587x |

L2 halves the baseline but costs DSMEM only 1.587x → 1.326x, because a
`page_size=1` gather is bound by *transaction count*, not by which level answers
it. L2 shortens a repeat; DSMEM removes it.

### Added: real cluster launch, opt-in (flashinfer `610d8d7`, `986270d`)
`FLASHINFER_FUSED_CLUSTER=N` switches the fused dispatch to `cudaLaunchKernelExC`
with a real `clusterDim`, default 1 so the shipped path is byte-identical. It
refuses rather than reshapes when N does not divide the SM count, since rounding
the grid would break the CTA count FA3's contract checks.

Isolation matrix via `FLASHINFER_FUSED_SKIP_FA3=1`:

```
CLUSTER=1 SKIP_FA3=0   rc=0  PASS
CLUSTER=2 SKIP_FA3=0   rc=1  CUDA error: unspecified launch failure
CLUSTER=1 SKIP_FA3=1   rc=1  424/512 unwritten (expected), no CUDA error
CLUSTER=2 SKIP_FA3=1   rc=1  424/512 unwritten (expected), no CUDA error
```

**The FA2 side is cluster-safe; FA3 alone is cluster-hostile.** This overturns an
earlier conclusion in this work that the single-grid two-role design forecloses
per-role cluster features — the side that wants DSMEM is already cluster-ready, and
the remaining work is making FA3 tolerate a cluster it does not use.

### Resource correction
Two H20 trials had been allocated for the PD-colocation experiments before the
single-GPU kernel work finished, and both sat idle for the session:
`81d861acb7b7c964` (4 nodes x 8 = **32 GPUs**, over the 16-GPU cap) and
`a1edeb8114365af4` (8 x 2 = 16 GPUs, within cap). Both now `STOPPED`. The
multi-node work is gated behind the kernel work by the goal's own ordering, so
allocating ahead of it was wrong regardless of the cap.

### Still open
4-mode MTP **serving** schedule unreachable (`forward_batch_speculative_generation`
would need reordering against the scheduler), so every server number remains
spec-decoding-off. Multicast measured but not integrated. Best server
configuration is the confirmed +0.7-0.8% win from v0.7.2.

## v0.7.2 — The Fused Kernel Reaches the Serving Path: Three Bugs, and a Measured 8.1% Regression (2026-10-04)

### Fixed: the page-index array handed to the fused kernel was wrong
The previous session's planning optimisation reused `wrappers[0]._kv_indices` on
the assumption that the mixed wrapper, planned over the whole batch moments
earlier, already held exactly the page-id array the launch needs. It does not.

Found by dumping the launch arguments to disk immediately before the faulting
launch (`SGLANG_FUSED_DUMP=<path>`, written at the launch site so it survives the
abort) rather than by a third hypothesis. On the faulting step:

```
qo = [424, 87, 1]   prefix = [3577, 0, 13]   ->  kv = [4001, 87, 14]
kv_indices: 3846 entries, batch needs 4102
            256 entries outside the valid page range [0, 1816955]
            first invalid at index 3590, value 1085595437
q, o, kv_indices, fa3_sparse base pointers: all 16-byte aligned
```

So nothing was misaligned. The kernel was gathering garbage page ids — up to
1.1e9 against a 1.8e6 page pool — from an array both too short and partly
uninitialised, and faulted on the resulting address. The optimisation it replaced
was worth ~0.3% of throughput.

- `_fused_kv_indices` now builds the array vectorised on the GPU (one
  `repeat_interleave` for each output slot's row, one subtract for its offset,
  one gather from `req_to_token`), so correctness does not cost the planning time
  back, and no per-request Python loop returns.
- It is checked against the plan's own page total before launch. The absence of
  that check is why a wrong buffer reached the GPU as a fault rather than a
  declined plan.

### Fixed: the tile-128 task was never executed (flashinfer `b43e50b`)
The fused extension built persistent params for tasks 1 and 2 only, on the
comment "task 0 went to FA3". The persistent plan is built over the FA2
sub-batch *after* the FA3 rows are removed, so its task 0 is ordinary FA2 work:
`TwoStageHolisticPlan` sends every row with `packed_qo_len > 64` there, and folds
the whole tile-64 mid class in too whenever that class carries less than
`FLASHINFER_TRI_TILE_MIN_KV` total KV. Nothing executed it.

No harness could have caught it: they all use MTP decode rows for the FA2 side,
whose `packed_qo = 2 * gqa = 14` lands in task 2, so task 0 was always empty. In
the dumped batch row 1 (`packed_qo = 609`) landed in task 0 and was skipped.

Adds `R128 = PKT<128, 2>` threaded through `fused_drain_queue`,
`FusedFA3PersistentKernel` and `FusedFA3PersistentDispatched`. Shared memory is
unchanged — tile 128 needs 49152 bytes against tile 16's 69632, so `Runner3`
still sets the bound.

### Measured (H200, flashinfer `b43e50b`, sglang this commit)
`tests/test_fused_server_batch.py`, which replays the dumped batch shape verbatim
so one launch covers FA3, task 0 and task 2:

| kernel | unwritten rows | FA3 row | task-0 row | task-16 row |
|---|---|---|---|---|
| pre-fix | **87 / 512** at every split | 0.00029 | **nan** | 0.00293 |
| fixed | 0 / 512 | 0.00023 | 0.00196 | 0.00271 |

87 is exactly row 1's token count — the task-0 row. Relative error is against an
fp32 reference, at `sm_split` 48 / 32 / 16.

Command: `FI_ROOT=/dev/shm/w/fi python3 tests/test_fused_server_batch.py`

### Ruled out
`tests/align_fused.py` reproduces each server-side tensor provenance in the
standalone harness — `q` as a non-contiguous view of a fused QKV buffer, `o` as a
slice of a reused buffer, the index array built by concatenating `req_to_token`
rows. All pass, so provenance was never the cause, contrary to the hypothesis
recorded in the previous session.

### Fixed: the KV cache does not have a fixed rank (flashinfer `4d59914`)
The extension read `k_cache.stride(1)` as the token stride and `stride(2)` as the
head stride. The standalone harness slices a `[pages, 2, page_size, H, D]` buffer
and so passes **rank 4**; sglang's pool passes **rank 3** `[tokens, H, D]`. On the
dumped batch that gave `k_stride_h = 1`, so all four KV heads aliased one head's
memory and `get_gmem_layout` built a degenerate layout whose tensor map made
`cudaLaunchKernel` **itself** fail with "misaligned address". The failed launch
poisoned the CUDA context, so the next unrelated attention call failed too and
killed the server — which is what earlier runs mistook for a fault inside
flashinfer's own persistent kernel.

The page step survived both layouts by accident (`ratio * k_stride_n` is
2*512 for rank 4 and 4*128 for rank 3, both correct), which is why only the head
stride was wrong and the symptom was a launch rejection rather than wrong numbers.

`kv_strides()` now derives `(page, n, h)` from the rank and rejects any other
rank; `_fa3_sparse_indices` picks the matching vector-sparse ratio.

Ruled out by measurement, not argument: tensor provenance (`tests/align_fused.py`)
and the CUDA 12.8 vs 13.1 toolchain split (`tests/test_fused_server_batch.py`
passes under the server's own venv python).

### Measured: the fused kernel runs server-side, and costs 8.1%
First valid server-side A/B. `--random-input-len 8192 --random-output-len 192
--num-prompts 300 --request-rate 12 --max-concurrency 32`, qwen7b on H200,
CUDA 12.8 / torch 2.8.0, sglang this commit + flashinfer `4d59914`:

| metric | fused | base | delta |
|---|---|---|---|
| total token throughput (tok/s) | 23250.44 | **25291.44** | **-8.1%** |
| mean TTFT (ms) | 3443.97 | 3150.61 | +9.3% |
| mean ITL (ms) | 20.18 | 18.54 | +8.8% |
| p99 ITL (ms) | 31.87 | 31.62 | +0.8% |
| `mixed_fused_fa3` share | 29540/32400 (**91.2%**) | absent | — |

300/300 requests succeeded in both arms, 0 misaligned, 0 launch failures. The
direction agrees with the kernel-level result (0.410x the sequential co-launch,
0.709x at the best split on H200), so the remaining deficit is the structural one,
not wiring.

An earlier A/B at `--random-input-len 2048` showed a null result (18182 vs 18203
tok/s) and was **invalid**: at that length no row clears the FA3 gate's
`kv_tiles >= 32` (>= 4096 KV) requirement, so `mixed_fused_fa3` was absent from
both arms and they ran the same kernel. Every fused measurement must carry its
`mixed_fused_fa3` share or it is not evidence about this kernel.

### Optimized: sweeping the SM split recovers more than half the regression
`sm_fa3` was pinned at 48 on every serving run by `plan_dispatch_prefix`'s
`MAX_SM_FA3`, so the kernel's main knob had never been varied server-side.
`SGLANG_FUSED_SM_FA3` now pins it. Matched sweep, 200 prompts, input 8192, H200:

| `sm_fa3` | tok/s | vs base | mean ITL | p99 ITL |
|---|---|---|---|---|
| 16 | 19369.38 | -22.4% | 23.77 | 45.46 |
| 32 | 22034.74 | -11.7% | 20.93 | 35.94 |
| **64** | **24055.80** | **-3.6%** | 19.18 | 34.74 |
| 80 | 23800.46 | -4.6% | 19.38 | 35.72 |
| 96 | 22856.25 | -8.4% | 20.26 | 37.05 |
| 112 | 19457.78 | -22.0% | 23.95 | 44.89 |
| base (fused off) | 24946.41 | — | 18.47 | 32.85 |

200/200 requests in every arm, 0 faults, fused share ~90.6% throughout. The curve
is sharply peaked at 64, and the old ceiling of 48 sat below it: **-8.1% -> -3.6%**.
`MAX_SM_FA3` is now 64 so the default lands on the measured optimum.

**The analytic balance point is not usable for this knob.** Unclamped,
`plan_dispatch` returns 115-131 for real serving batch shapes — nearly the whole
device on FA3 — and the sweep measures 112 as one of the two *worst* settings.
The clamp, not the model, was choosing the value. That cost model was fit for the
overlap kernel and does not transfer to this kernel's makespan.

### Tested: cross-role backfill on the server
`FLASHINFER_FUSED_BACKFILL=1` at `sm_fa3=64`: 24064.65 vs 24055.80 tok/s,
**+0.04% — within noise**, against 1.2% (H200) / 1.9-6.9% (H20) at kernel level.
Consistent rather than contradictory: backfill reclaims the prefill tail of one
mixed step, and at 32 concurrent requests the server already has other work to
overlap it with.

Correctness with backfill on is now clean under the server's own toolchain on the
dumped server batch shape, at splits 64/48/32: 0/512 unwritten, rel err
0.00028/0.00218/0.00282 (vs 0.00032/0.00245/0.00313 with it off). Scope: that is
one shape, and it is not the one the earlier backfill fault was recorded on, so it
does not retire that fault. The default stays off, now for a measured reason.

### Multicast: the precise blocker
Not a flag. The fused kernel's FA3 side uses `hopper/sparse_mainloop.cuh`, where
`USE_TMA_LOAD_KV = false` (line 87) — KV is gathered with `cp.async`, so there is
no TMA descriptor to multicast from — and the Q descriptor is built with `_1{}`
and the comment "no mcast for Q". Enabling it needs `page_size >= CTA_TILE_KV`
(128) threaded through the fused launch (currently hardcoded to 1) **and** the
sparse mainloop's gather replaced by a TMA load. The second is a rewrite, not
configuration.

### Still open
Best fused configuration is -3.6% against the persistent kernel, so
`SGLANG_FUSED_FA3_SM_SPLIT` stays 0 by default. Closing the last 3.6% needs
warpgroup-reentrant FA2; the split is now tuned and backfill is measured at zero.

## v0.7.1 — Split-KV and GQA Packing for FA3; Gate on Tile-Grid Quantities (2026-09-30)

### Measured
- **FA3 split-KV: 1.75x over FA3 as-is and 1.38x over the persistent kernel**
  (507 -> 290 us, S=2, one 1024x32768 row at 8 q heads). Width, not KV traffic,
  was FA3's problem, and it is a property of flashinfer's sm90 planner having no
  KV split rather than of wgmma. Emulated via batch expansion + `merge_states`.
- **GQA packing: 5.0x on the verify class** (2541 -> 511 us, 58 rows @ kv 11000),
  moving it from 0.23x to 1.13x against the persistent kernel at packed_qo 56;
  still 0.78x at packed_qo 32. Nothing on large prefill (509.8 vs 507.2 us).
- **CUDA graphs: the fused path is strictly better.** At fixed tokens and batch
  size, one captured fused graph replays correctly across p/d ratios 1/32 -> 8/25,
  while FA3's `batch_size`/`qo_indptr` length change with the ratio and are baked
  into the captured launch. Sequential needs a capture per (n_prefill, n_decode)
  pair. This reverses the earlier parity claim.

### Rewritten: mixed_dispatch gate
- Routes on the two tile-grid quantities the measurements identify:
  `packed_qo_len` vs `MIN_PACKED_QO = 48` (tile occupancy) and `kv_tiles` vs
  `MIN_KV_TILES = 32`. Work is counted in (q tile x kv tile) pairs.
- The two kernels get **disjoint SM sets** at the makespan balance point; FA3's
  KV-split S is the smallest that covers the SMs it was given (returns S=2 for
  the (8,1,128) chunk, the measured optimum).
- A class folds into the persistent kernel unless one SM's worth of it is at
  least half the makespan the rest of the device needs anyway -- no separate SM
  set that cannot be kept busy.
- Latency 2.15 us on decode-only batches (short-circuits on `max(qo)`) and
  13.8-14.7 us when an FA3 row is present, once per forward.

### Corrections
- The gqa-7 verify figure was first reported as 2.72x; it packed one GQA group
  while the persistent kernel processed all 4, so FA3 did a quarter of the work.
  Matched-work value is 1.13x.

## v0.7.0 — 4-Mode Dispatch: Separate Kernels, Scheduler-Partitioned Grids (2026-09-30)

### New: per-tile-class kernel dispatch with a grid-partitioning scheduler
- `python/sglang/srt/layers/attention/mixed_dispatch.py` (new): picks per batch
  between `fused` (one persistent launch), `seq` (FA3 then persistent) and `conc`
  (both co-launched at a split grid), and returns the grid split in SMs.
- flashinfer (`tri_tile_`): `FLASHINFER_FA3_MAX_CTAS` caps `PrefillSM90Plan`'s CTA
  count and `FLASHINFER_PERSISTENT_MAX_CLUSTERS` caps the holistic planner's grid,
  each rebalancing its own work over what it owns. Only the grid shrinks, so graph
  capture still sees one shape per bucket — green contexts were rejected because
  captures would multiply by the number of groups.
- `plan_4mode_dispatch` in `flashinfer_backend.py` now delegates to the measured
  scheduler; the unsupported `DISPATCH_MIN_MONOLITHIC_US` guard and the fixed
  `DISPATCH_CLASS_KERNEL` table are removed (both are contradicted by measurement).

### Results (held out on captured sets; see tasks/debug_kernel_server_perf.md)
- vs sequential (vLLM-style): **1.015x** overall, **1.054x** (h200 TP8) and
  **1.147x** (gb200 TP16); parity on the small-model long-prefix set, where the
  scheduler correctly picks `seq` and becomes the baseline.
- vs a fixed fused megakernel: **1.383x**. vs dual-stream: **1.000x** (no gain).
- Graph compatibility is **not** better: only 27 of 41 shape buckets resolve to a
  single mode, so parity (41 graphs) costs most of the gain and the full gain needs
  55 captures.

### Key measurements
- FA3's wgmma mainloop is **1.86x** the persistent kernel's per-CTA compute rate,
  but it reloads KV once per q head and its width is `h_qo * ceil(qo_len/128)` —
  only 64 CTAs at the (8,1) TP8 shard. At 8 q heads per rank the persistent kernel
  therefore **beats FA3 on pure prefill** (401us vs 507us at 256 KV tiles).
- Co-launch does overlap (up to **1.65x** at a 64/68 split), but over-provisioning
  the cooperative kernel stalls it behind FA3 (2286us at 88/44 vs 1470us).
- Boundary fitted on 50 cells, **50/50** correctly classified; within 5% of the
  per-batch oracle on 26/28 held-out batches.

### Doc hygiene
- `tasks/archive_h100_era.md` (new): 581 lines of H100/Qwen3-8B results, closed
  investigations and v3-scheduler notes moved out of `debug_kernel_server_perf.md`,
  which is reorganised into four story parts (gate -> workload -> kernels ->
  dispatch) and shrinks 2,184 -> ~1,540 lines. The superseded 4-mode *estimate*
  section collapses from 125 to 27 lines.
- Corrected the FA2/FA3/persistent planner table: it had given FA3
  `cta_tile_q = 64` and a `total_kv_lens/num_clusters` KV split, citing lines that
  are actually inside `TwoStageHolisticPlan` (our planner). FA3 is
  `PrefillSM90Plan`: `cta_tile_q = 128`, **no KV split**, no group-head fusion.

### Fixes
- Routing tested KV length alone and sent 58 spec-verify rows to FA3, **14x slower**:
  FA3 does not fuse group heads, so the test must be on `qo_len`, not `packed_qo_len`.
- Split accounting mixed units — the persistent planner sets `num_sm *= 2`, so a
  "96/36 split" subscribed 114 of 132 SMs. This invalidated four earlier `conc` wins.
- Out-of-bounds write in the per-task SM apportionment's remainder redistribution
  (`task_hi` could exceed `num_clusters`).

## v0.6.0 — Real-Workload Capture, Hopper Scheduling Study, Gate v3 (2026-09-19 → 09-22)

### Workload capture (new)
- `SGLANG_LOG_MIXED_COMPOSITION=<path>` records every mixed batch's real composition
  from the gate: per-row decode KV, chunk tokens, the **true attended prefix** (the
  scheduler's `#cached-token` is 0 for continued chunks), and the gate's decision.
  Wired into `run_mixed_workloads.py`, plus four `loadsweep_{hi,med,lo,min}` scenarios.
- Captured 3,493 mixed batches across a high→low co-tenancy load sweep. The realized
  shape: chunk size barely moves with load (p50 395–446) because it is the leftover
  token budget, while decode rows go 46 → 2 and the gate's persistent share 77% → 23%.
  Prefixes are fresh 73–96% of the time.
- **The sweep grid was badly mis-sampled**: 91.4% of real chunks fall in 128–512 tokens,
  a bin the grid covers with one point, while five of its seven chunk values sit where
  7% of traffic lands.

### Corrections (2026-09-23)
- The captured ~400-token "chunks" are **not chunks**: the distribution is bimodal and the
  small mode is the chat tenant's whole prompt (uniform[256,512], prefilled in one shot
  since 512 ≪ chunked_prefill_size=8192). It is load-invariant because it is a scenario
  parameter, not a leftover token budget; EDF was off (`fcfs`). The gate-v3 gain is
  therefore established for this workload's prompt-length mix, not in general.
- **Scope limit**: a decode row's `packed_qo_len` is `1 × gqa_ratio`, and the dual-tile
  premise only exists at ratio ≤ 16. At ratio 64–128 (MLA: Kimi-K2, DeepSeek-V3/V4)
  decode rows already sit in the large tile class — no heterogeneity to exploit and no
  16→128 flip for a chunk to cause — and those models use separate MLA backends that
  never call `select_mixed_attention_kernel`. The win region is low-GQA-ratio models
  (Llama/Qwen dense, 7–16) on the standard flashinfer backend.

### Kernel replay on real compositions (`sweep_from_capture.py`)
- Time-weighted over the captured distribution: FA2 1.00×, **FA3 0.67×**, persistent
  **2.60×**, deployed v2 gate 1.90×. FA3 loses to FA2 at every decode-row count
  (0.28–0.84×) — its fixed `cta_tile_q = 64` pads a GQA-7 decode row ninefold, which
  costs more than its better planner saves. Porting to FA3 is not the Hopper win.
- Gate decline diagnosis over 3,520 batches: 55.1% accepted, **27.0% declined by the
  out-of-support guard**, 17.9% non-unanimous — the grid mismatch, costing 14.2% of
  total FA2 kernel time.

### Gate v3 (deployed)
- Rebuilt `mixed_kernel_gate_h800_gqa7.json` with 60 measured serving compositions
  folded into the grid (1,376 points). On **held-out** buckets: accept 63.1% → 98.1%,
  speedup vs FA2 **2.17× → 3.50×**, 99.6% of oracle. A 1.61× kernel-policy gain with
  no kernel change.
- Serving validation (single run, `loadsweep_med`): live gate acceptance 34.1% → 89.3%,
  100% retokenization, **node throughput +0.0%** — the kernel-policy gain does not
  translate into node throughput. Chat tail improves (mean ITL −10.3%, P99 −43.1%);
  codex P99 +7.2% / TTFT +8.3%, inside the documented ±6% saturation run-to-run band,
  so neither direction is established without a paired repeat.
- Doing this at k=5 would have cost the zero-regression property (worst-case LOO
  0.971×); **k=7 restores worst-case LOO to exactly 1.000×** with no loss of held-out
  gain. `build_mixed_kernel_gate_table.py` gains `--k` / `--label`.

### Hopper instruction-level study (`tasks/debug_kernel_server_perf/wgmma/`)
- Five microbenchmarks, marginal cycles, validated at 96% of H800 peak bf16:
  wgmma shape mixing does **not** overlap within a warpgroup (32.00 + 20.00 = 52.00,
  order irrelevant) **or across warpgroups** on one SM (52.10) — the tensor core is a
  serial per-SM resource. exp2 does **not** hide under an in-flight wgmma (+50.5 cyc/8
  ops), but **hides 73% of the MMA when placed on a different warpgroup** (FA3's
  ping-pong premise, quantified). Small tiles absorb 8 FFMA ops at **zero** cost where
  large tiles pay 3.5 — cost is ≈ `16 + N/4`, so `n16` runs at 38% of peak.
- Closes one β hypothesis: the non-overlappable 0.53 is not the MMA pipe.
- Two measurement traps documented: ptxas C7515 (non-wgmma writes to accumulators)
  silently serializes the pipeline, and any runtime branch in the issue loop inflates
  the baseline 3× — both produced false "no overlap" readings in earlier drafts.

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
- External-validity subsection for the gate sweeps (are isolated-kernel benchmarks
  representative of peak serving?): measured from the mixed-tenancy captures that exactly
  one kernel is resident 93.25% of serving wall (≥2 only 0.36%), so the bench's
  one-kernel-alone condition is the serving condition; serving kernel times (BP
  0.007–16.6 ms, persistent 0.055–2.26 ms) sit inside the swept envelope (BP 0.019–28.6 ms,
  PA 0.019–25.7 ms, decode KV 2K–6.3M); and BP's p90 kernel time is 0.267 ms in the gated
  arm vs 0.906 ms in DMC, i.e. the predicted routing is visible end to end. Documents the
  gaps too — contiguous `arange` KV pages vs a fragmented radix allocator, sustained-load
  clocks, flushed vs dirty L2, graph replay, shared GPUs.
- New WGMMA microbenchmark (`tasks/debug_kernel_server_perf/wgmma/`) answering whether
  different tile sizes overlap or get reordered: they do not. Marginal cost is exactly
  additive (m64n64k16 32.00 + m64n16k16 20.00 = 52.00 cycles/iter), issue order is
  irrelevant, and two warpgroups on one SM give 52.10 — the tensor-core pipe is a serial
  per-SM resource. Validated at 96% of H800 peak bf16 for the large tile. Closes one
  hypothesis for the β TODO (overlap cannot come from MMA-with-MMA, only MMA-with-memory)
  and quantifies a 2.5× per-unit-work penalty for small query tiles (cost ≈ 16 + N/4
  cycles, so n16 runs at 38% of peak).
- Mechanism arithmetic re-verified against flashinfer source, fixing one error: the
  persistent planner's split was documented as `kv_len_limit ~ total/264`, but 264 is
  fa2's `max_grid_size` (2 × 132 SMs) — the persistent side computes
  `f(total_kv_lens × num_kv_heads / num_clusters)` with `num_clusters = num_sm = 132`,
  i.e. ≈ total/33 here, halved again for the prefill tile class. The two BatchPrefill
  numbers turn out exact rather than approximate for this config: the split budget is
  exactly 66 units/KV head (`max_grid_size / num_kv_heads` = 264/4) and a 1,024-token
  chunk's q-tiles are exactly 56 of them (`ceil(1024 × 7 / 128)`), so splitting the eight
  decode rows two ways (+16) would exceed the budget and the search stops splitting
  entirely. Also notes that the tile decision keys on the batch *mean* only in the
  non-graph path (what the sweeps use) and on `max_qo_len` under CUDA graphs, and flags
  that the 0.366 ms decode-only datapoint is not reproducible from saved artifacts.
- **Corrected an overstatement**: "BatchPrefill's cost surface has cliffs no smooth model
  can track" was too strong. New `fit_learned_cost_model.py` fits MLP and GBDT to both
  kernels on the same 1,316 sweep compositions, features and model class (5-fold CV × 3
  seeds): an MLP reaches 9.9% MAPE on BatchPrefill, beating the 13.3% of the analytic
  parts-based fit, so part of that gap was the functional form. What survives is the
  relative claim — BP is 1.7–3× harder than the persistent kernel (GBDT 10.1% vs 3.3%,
  worst case 103% vs 63%) — plus two practical points: adding fa2's own tile-decision
  variables recovers only 10.1%→8.6%, consistent with the residual living in the
  split-budget search over the per-row KV vector; and fitting BP needs ~1,300 measured
  compositions per GPU versus 35 for the analytic law, reintroducing exactly the
  profiling campaign it removes. Docstring in `overlap_cost_model.py` updated to match.
- Recorded why the kernel layer exists when only `predict()` is consumed: it is the
  interior of the one model in use, not a second model, and it is what makes the model
  training-free (35 kernel measurements vs the replaced MLP's 25,444 profiled configs),
  what makes the fused-vs-split predictability contrast visible at all (the step-level
  `c0 + c1·tokens` term masks it), and what an overlap-aware chunk controller would need.
  That controller is now an Open TODO rather than only prose.
- Clarified that the analytic cost model is **kernel-level** (its per-layer attention law
  is fit on 35 kernel measurements: T_d 2.04% MAPE, T_p 3.47%, composition law 3.10%
  in-sample and newly 3.22%/9.7% leave-one-out, so the one-parameter law is not overfit);
  only the `c0 + c1·tokens` step wrapper is server-level, and those two constants are
  derived from hardware specs rather than fit, with a single serving anchor. Records the
  two scope limits: persistent kernel only (no BatchPrefill equivalent by design), and a
  fit grid that sits above the realized operating point of the SLO runs it drives.
- Batch-mixture representativeness measured from the gated runs' scheduler logs and
  recorded as a known weakness: the sweeps are a Cartesian grid, but realized mixed
  batches form a narrow correlated cluster per workload (win scenario: 386-token chunks
  p90 532 against 46 decode rows @189K KV; TraceLab: 8,182-token chunks against 10 rows
  @904K). Chunk size and decode count are anti-correlated in serving because the chunk is
  the leftover budget, so the product grid spends points on combinations the scheduler
  cannot emit, and samples the actual operating region with two points. Notes that
  leave-one-out 1.000 is therefore generalization within the grid, not to serving.
  New TODO: rebuild the table by importance-sampling logged compositions.
- Removed the "DMC baselines were silently persistent" post-mortem now that every result
  it invalidated is gone from the docs; its one live invariant (baseline runners must set
  `SGLANG_FLASHINFER_USE_PERSISTENT=0` — it defaults on) is promoted to Setup as a rule.
- Cost-surface and win-region sections unified into one: adds a sweep-provenance table
  (479 uniform + 477 fresh → 956 merged, + 360 blended = the deployed 1,316 points) with
  every file checked in, and shows where the headline multiple comes from — at a fixed
  fresh 2K chunk, BatchPrefill is flat (2.97 → 5.05 ms) while the decode pool grows 64×
  and the persistent kernel scales with the work (0.105 → 4.274 ms), so the speedup runs
  28.2× → 1.2×. Corrects "up to 20×" to the per-sweep maxima (21.0× uniform, 28.2×
  merged) and records the loss region from the same data (162/956 below 1.0×, worst
  0.57× at 1-4 tiny rows). New data files: `gate_sweep_blend_qwen25_h800.json` (was only
  in an ephemeral scratchpad, though it is 360 of the deployed table's points) and
  `probe_fresh_qwen25_h800.json`.
- Profiling post-mortem added to the overhead diagnosis, mined from the mixed-tenancy
  nsys captures. It leads with the conclusion — **our plan path does no D2H at all**
  (host tensors only; verified by `cudaDeviceSynchronize` appearing once in 279 s, so
  neither flashinfer's device-tensor branch nor the `seq_lens_cpu is None` fallback ever
  fires) — and presents the rest as the mechanism behind the old 56.5 ms, measured on
  the copies that remain elsewhere in the stack. Findings: `non_blocking=True` is not
  honored for D2H into pageable memory, and the
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
