# LPT Persistent Kernel + Mixed CG Performance

## Goal
Implement and optimize LPT (Longest Processing Time) dynamic scheduler for persistent attention kernel with mixed batch CUDA graphs in SGLang flashinfer backend.

**Reading map.** Part 1 is the sync-free plan and the measured gate. Part 2 is
what the captured workload actually contains, plus the GQA-ratio scope limit that
bounds everything. Part 3 compares the three Hopper kernels the dispatcher
chooses between. Part 4 is the current work: mixed-tile fusion and the measured
4-mode dispatch — its last section holds the live numbers. H100-era results are
in [`archive_h100_era.md`](archive_h100_era.md).

## Setup
- Model: `Qwen/Qwen3-8B`, H100/H800 80GB
- Server: `--enable-mixed-chunk --attention-backend flashinfer --mem-fraction-static 0.85 --enable-piecewise-cuda-graph`
- Env vars: `SGLANG_FLASHINFER_USE_PERSISTENT=1`,
  `SGLANG_EDF_PLAN_OVERHEAD_MS=auto` for the peak-load SLO-aware points
- Benchmark: `bench_serving --dataset-name random --request-rate 8 --num-prompt 400`
- Profile logs: `tasks/debug_kernel_server_perf/profile_logs/`
- gpu_keepalive MUST be stopped before benchmarks (corrupts results — cooperative launch is especially sensitive to SM contention)
- **Baseline runners MUST set `SGLANG_FLASHINFER_USE_PERSISTENT=0` explicitly.**
  It defaults to **1**, so a "BatchPrefill baseline" that omits it silently runs
  the persistent kernel for mixed batches and the comparison becomes
  persistent-vs-persistent. This contaminated every H800 DMC baseline between
  2025-11-14 and 2026-07-30; those results have been removed from this file and
  `realistic_workload_analysis.md` (git history retains them). The honest
  baseline is 6.1–10.7% *faster* than the contaminated one, so the bug
  understated both sides.

## Part 1 — Sync-free plan and the measured kernel gate

The original thesis: pass host tensors so `plan()` never syncs, then gate
between the persistent kernel and BatchPrefill per batch from a measured cost
surface rather than a heuristic. Aim was to remove the low-concurrency
regressions of the persistent overlap kernel (LPT planning + cooperative-launch
overhead) with zero regression elsewhere. sglang `9bbc82e28`+, flashinfer
`f4d2a80`; sweep scripts and data are inventoried in the cost-surface section,
and the gate table is built by `build_mixed_kernel_gate_table.py`.

### Overhead diagnosis (H800, Qwen2.5-7B heads 28/4)

- `BatchAttention.plan()` ran three D2H copies + `torch.cuda.synchronize()`
  per mixed step. With ~40 ms of queued GPU work the plan call stalls
  **56.5 ms**; this serialization (not cooperative launch) was the dominant
  serving overhead. Fix: host-built indptrs from `seq_lens_cpu` +
  double-buffered pinned staging fenced by CUDA events (the pinned-buffer
  reuse race is why earlier naive sync removals hung). Plan wall time with a
  busy GPU: **0.47 ms**, outputs bitwise identical
  (`tasks/debug_kernel_server_perf/test_syncfree_plan.py`).
- Persistent `run()` CPU dispatch is **cheaper** than BatchPrefill
  (16 µs vs 26 µs/call); cooperative launch is not a CPU-side cost.
  `persistent_attn_forward` now reuses out/lse buffers (was 2 allocs/layer).

**Post-mortem: why that D2H cost 56.5 ms, and where the pattern still lives.**
*Our plan path no longer does any D2H.* It passes host tensors only —
`plan_qo_indptr`/`plan_kv_indptr`/`plan_kv_lens` are built by NumPy cumsum
from `seq_lens_cpu`/`extend_prefix_lens_cpu` — so flashinfer's device-tensor
branch never runs. Verified in the mixed-tenancy captures: that branch ends in
`torch.cuda.synchronize()`, and `cudaDeviceSynchronize` appears **once** in
279 s (startup), so neither it nor the `seq_lens_cpu is None` fallback ever
fired. What follows is the mechanism behind the old 56.5 ms, measured on the
copies that remain *elsewhere* in the stack (2026-09-14, sglang `368f5e6fe`,
flashinfer `c6d2a00`; full-run nsys captures of the `agent_plus_shortchat`
scenario, 279 s span, one CUDA-issuing thread — 0.8–1.2 GB SQLite exports kept
as local artifacts, not checked in; reproduce by joining
`CUPTI_ACTIVITY_KIND_MEMCPY` to `CUPTI_ACTIVITY_KIND_RUNTIME` on
`correlationId` and grouping by `copyKind`/`dstKind`).

- **`non_blocking=True` is a no-op for a D2H copy, and the cost is a pipeline
  drain, not a copy.** `.to(torch.device("cpu"), non_blocking=True)` allocates
  a *pageable* destination; `cudaMemcpyAsync` D2H into pageable memory is
  synchronous with respect to the host, so the call returns only after
  everything already queued on the stream has run. Pinned destinations in the
  same runs take the async fast path:

  | run | D2H dest | n | avg bytes | avg host-blocked |
  |-----|----------|--:|----------:|-----------------:|
  | LPT | pageable | 46,064 | 65 | 3,607 µs (max 564 ms) |
  | LPT | pinned | 23,331 | 107 | **6.9 µs** |
  | DMC | pageable | 36,175 | 102 | 4,910 µs |
  | DMC | pinned | 16,909 | 146 | **7.2 µs** |

- **All of the cost is the first copy in a group.** These pageable copies
  arrive back-to-back (92% in pairs), and position, not size, decides:

  | run | pairs | 1st copy mean / median | 2nd copy mean | total 1st vs 2nd |
  |-----|------:|-----------------------:|--------------:|-----------------:|
  | LPT | 20,443 | 8,104 / 5,060 µs | **14.4 µs** | 165.7 s vs 0.3 s (562×) |
  | DMC | 14,334 | 12,361 / 5,054 µs | **15.5 µs** | 177.2 s vs 0.2 s (795×) |

  Same direction, same pageable destination, same size class, 560–800× apart:
  the first call absorbs the queue, the rest are free. Applied to the old plan
  path, essentially all of the 56.5 ms was the **first** `.to("cpu")`; copies
  2–3 and the following `torch.cuda.synchronize()` were ~15 µs each. (Measured
  on other call sites in the same server, not on plan() itself — the mechanism
  is CUDA's, not the call site's.)

- **A blocking readback does not necessarily starve the GPU.** Steady-state
  20 s window, union of all kernel intervals across 12 streams:

  | run | kernels | GPU busy | host blocked in pageable D2H | GPU idle *while* host blocked |
  |-----|--------:|---------:|-----------------------------:|------------------------------:|
  | LPT | 244,225 | **93.6%** | 74.7% of wall | 0.64% of wall (10% of idle) |
  | DMC | 230,612 | **94.0%** | 74.6% of wall | 0.64% of wall (11% of idle) |

  The host thread genuinely stops for three quarters of wall time, yet the GPU
  stays ~94% busy: already-queued run-ahead covers the wait. So such a copy
  hurts only when it removes run-ahead the GPU needs — which is exactly the
  old plan() case (it blocked *before* issuing the step's own work, for data
  that was available host-side all along), and is not the case for the
  scheduler's end-of-step token readback (it waits for a result that genuinely
  is not ready). This is also why removing the plan sync was worth more at low
  concurrency than at saturation.

- **Block duration measures queue depth, not overhead** — so it must not be
  read as an arm-vs-arm cost comparison. Medians are identical (5,060 vs
  5,054 µs) and blocked wall fraction is identical (74.7% vs 74.6%); LPT's
  lower mean is a thinner tail spread over 43% more readbacks (more completed
  steps), i.e. the signature of the faster server, not of lower overhead.

- **Where the residual copies are.** Sized `bs+1` then `bs` int32 — the
  signature of flashinfer's stock `plan()` (`decode.py:941-942`
  `indptr.to("cpu")`/`last_page_len.to("cpu")`, and `prefill.py:1711-1719`),
  which sglang still triggers by passing device tensors to `begin_forward()`
  without `max_token_per_sequence`/`max_sequence_kv`. So the BatchPrefill and
  decode paths still pay per step what the persistent path no longer does;
  exact attribution needs a capture with `--cudabacktrace`. See Open TODOs.

### The cost surface: where the persistent kernel wins, and why

Serving-matched composition (the non-causal cached-prefix part of a mixed
batch), `bench_gpu_time` medians, BatchPrefill fa2 vs both persistent
schedules. Three sweeps, all checked in, which together are the deployed
1,316-point gate table:

| sweep | n | decode rows | decode KV/row | **chunk tokens** | **chunk prefix** |
|-------|--:|-------------|---------------|------------------|------------------|
| uniform | 479 | 1–128 (all equal) | 2K–98K | 128, 512, 1K, 2K, 4K, 7K, 8K | 512, 2K, 4K, 8K, 16K, 32K, 64K, 80K |
| fresh-prefix extension | +477 | 1–128 (all equal) | 2K–98K | 128, 512, 2K, 4K, 8K | **0, 128, 256** (+32K/64K controls) |
| blended pools | 360 | 2–88 (**mixed lengths**) | 2K–64K within one batch | 512, 2K | 0, 4K |

| sweep | script | data file |
|-------|--------|-----------|
| uniform | `gate_sweep3.py` | `gate_sweep3_qwen25_h800.json` (gqa4: `gate_sweep3_qwen3_h800.json`) |
| fresh-prefix extension | `sweep_fresh.py`, `probe_fresh_qwen25_h800.json` | merged with the above into `gate_sweep_merged_qwen25_h800.json` (956) |
| blended pools | `sweep_blend.py` | `gate_sweep_blend_qwen25_h800.json` |

The fresh-prefix extension is the one that mattered: the uniform sweep never
sampled a chunk prefix below 512, so the entire fresh-arrival region — where
the kernel is worth 8–28× — was invisible to it. The blended sweep is the only
one whose decode rows have *different* lengths within a batch, which is what a
real co-tenancy pool looks like and what the KV-concentration feature keys on.

Peak speedup is **21.0×** in the uniform sweep and **28.2×** once fresh
prefixes are sampled.

**Do these sweeps represent peak server load?** They are isolated-kernel
benchmarks (`bench_gpu_time`: warmup + L2 flush, one kernel at a time on an
idle GPU), so this was checked rather than assumed (2026-09-18, against the
mixed-tenancy captures):

- **Kernel isolation is the serving condition, not an artifact.** In a
  steady-state 20 s window, exactly one kernel is resident for **93.25%** of
  wall; two overlap for 0.36% and three or more for 0.00%. sglang issues
  essentially everything on one stream, so an attention kernel gets the whole
  GPU in serving just as it does in the bench. The sweep is not measuring an
  unrealistically quiet device.
- **Serving sits inside the swept envelope.** Sweep covers total decode KV
  2K–6.3M tokens with BP 0.019–28.6 ms and persistent 0.019–25.7 ms; measured
  in serving, BP kernels run 0.007–16.6 ms and persistent 0.055–2.26 ms
  (p50 0.209 ms). No serving composition lands outside the sampled range.
- **The predicted routing shows up in serving.** BatchPrefill's p90 kernel
  time is **0.267 ms in the gated arm vs 0.906 ms in the DMC arm** — same
  kernel, same workload, differing only in that the gate moved the
  compositions the sweep flagged as pathological onto the persistent kernel.

What the sweeps do *not* reproduce, in rough order of concern:

- **Contiguous KV pages.** `Bench` uses `kv_indices = torch.arange(...)`,
  while a long-lived server's radix allocator hands out fragmented pages. Both
  kernels gather through that array, but they gather differently (persistent
  splits one row's KV across CTAs; BP streams long runs from few CTAs), so
  fragmentation need not penalise them equally. Untested.
- **Sustained-load clocks.** The bench is short bursts on a cool GPU; serving
  holds ~94% occupancy for minutes. No clock telemetry was captured.
- **L2 residency.** `bench_gpu_time` flushes L2; in serving attention inherits
  a dirty L2 from the preceding GEMM. Immaterial for long rows (working set
  >> 50 MB) but potentially material for the small-batch compositions where
  the measured 0.57–0.70× losses live.
- **Eager launch vs CUDA-graph replay** (kernel time is the same; launch path
  is not), and a **shared** GPU (MPS or a co-tenant process) would change
  cooperative-launch timing specifically — both sweeps and serving runs assume
  a dedicated device.

The load-bearing defence is not that the microbenchmark is realistic, but that
the table is only ever used as a *decision* rule whose decisions were then
validated end to end: parity controls, paired repeats, and the mixed-tenancy
A/Bs below are what would expose a bench-to-serving mismatch.

**The batch mixtures, though, are a grid — not the realized distribution.**
This is the weaker part of the methodology. The sweeps enumerate a Cartesian
product of (decode rows × per-row KV × chunk tokens × prefix); a real scheduler
emits a narrow, correlated cluster that differs per workload. Parsed from the
`Prefill batch` log lines of the gated runs (sampled at the decode log
interval, so tens of steps per run, not all):

| scenario | chunk tokens p50 (p90) | decode rows p50 | total decode KV p50 | gate → persistent |
|----------|-----------------------:|----------------:|--------------------:|------------------:|
| agent+shortchat (the win) | **386** (532) | 46 | 189K | 1,046/1,901 = **55%** |
| shortchat_only (control) | 382 (496) | 25 | 11K | 6/2,566 = 0.2% |
| multiturn_slo_clean | 2,151 (8,173) | 5 | 87K | — |
| TraceLab g16 (parity) | 8,182 (8,191) | 10 | 904K | 18/500 = 3.6% |

Three mismatches follow:

- **Density.** The chunk grid is {128, 512, 1K, 2K, 4K, 7K, 8K}, but the win
  scenario operates at 380–530 tokens — between the two lowest grid points —
  while most sweep points sit at 2K–8K, where that scenario essentially never
  runs. Realized decode rows (46) likewise fall between grid points 32 and 64.
  k-NN interpolates, but both its accuracy and the out-of-support radius are
  set by local sample density, which the grid, not the workload, determines.
- **Correlation.** In serving the chunk is the token budget *left over* after
  decodes are scheduled, so chunk size and decode count are strongly
  anti-correlated (46 decodes → 386-token chunks; 10 decodes → 8.2K chunks).
  The Cartesian sweep treats them as independent and spends points on
  combinations the scheduler cannot produce.
- **Heterogeneity.** Only the 360-point blended sweep has rows of differing
  lengths inside one batch, and it covers just chunk ∈ {512, 2K} ×
  prefix ∈ {0, 4K}. Every real decode pool is heterogeneous.

So the leave-one-out worst-case of 1.000 is generalization *within the grid*,
not to the serving distribution — the serving-level parity controls are what
cover the latter, and they are the reason the zero-regression claim is
defensible despite this. Measurement caveat: `#cached-token` is the radix hit
at admission, so it is a valid prefix proxy only for a request's first chunk;
for continued chunks the attended prefix is the already-processed portion and
is not in the log line, which is why no prefix column appears above.

**Mechanism (corrected 2026-08-06; arithmetic re-verified against flashinfer
source 2026-09-19).** BatchPrefill's split-KV works on its own — a decode-only
batch of 8 rows @64K runs in 0.366 ms through it, matching the persistent
kernel. The failure is a planner coupling that appears only in mixed batches,
and both halves of it are exact, not approximate, for this configuration
(H800: `num_sm`=132, `num_kv_heads`=4, `gqa_group_size`=7):

- **(a) One chunk flips the batch-wide CTA tile.** `scheduler.cuh:549-550`
  computes `avg_packed_qo_len = sum_packed_qo_len / batch_size` and calls
  `FA2DetermineCtaTileQ`, which returns 128 above 64, 64 above 16, else 16
  (`utils.cuh:303`). Eight decode rows contribute 7 each; a 1,024-token chunk
  contributes 7,168, so the mean is 802 and the tile is 128 — for *every* row,
  including the decode rows that wanted 16. (This is the non-graph path, which
  is what the sweeps and the mixed BatchPrefill path use. With
  `enable_cuda_graph` the planner instead keys on `max_qo_len`
  (`scheduler.cuh:534-537`), which a chunk raises just as surely.)
- **(b) The same chunk exhausts the split budget.**
  `max_batch_size_if_split = max_grid_size / num_kv_heads`, where
  `max_grid_size = num_blocks_per_sm * num_sm = 2 × 132 = 264`
  (`scheduler.cuh:716-718`) — so the budget is exactly **66 work units per KV
  head**. `PrefillBinarySearchKVChunkSize` (`scheduler.cuh:101-130`) searches
  for the smallest `kv_chunk_size` with
  `Σ ceil(packed_qo_len_i / cta_tile_q) × ceil(kv_len_i / chunk) ≤ 66`. The
  chunk's own q-tiles alone are `ceil(7168 / 128)` = exactly **56** of those
  66. Splitting the eight decode rows even two ways would cost 16 more
  (56 + 16 = 72 > 66), so the search terminates at `kv_chunk_size = max_kv_len`
  and the decode rows are not split at all.

Eight rows × 4 KV heads then stream 32 MB each (64K tokens × 128 dim × 2 bytes
× K and V) from 32 CTAs while ~100 of the 132 SMs idle: BP plateaus near
3.1 ms for any chunk ≥1,024 tokens, while the chunk's own attention work is
negligible.

The persistent planner has no such coupling. It sizes its KV split from the
*total* work rather than from a fixed unit budget:
`kv_len_limit = f(ceil_div(total_kv_lens × num_kv_heads, num_clusters))` with
`cluster_size = 1`, `num_clusters = num_sm = 132` (`scheduler.cuh:1148-1149`,
`1195`) — i.e. work per SM, ≈ `total_kv_lens / 33` for this head config, and
halved again for the prefill tile class (`1196-1199`). Every row is then
partitioned into (tile, head, kv-chunk) units regardless of batch
heterogeneity, so it stays bandwidth-bound (~0.38 ms).

*Caveat:* the 0.366 ms decode-only figure comes from an ad-hoc probe during the
2026-08-06 investigation; every checked-in sweep point has a chunk, so that one
number is not reproducible from saved artifacts.

**This is the whole win region in one table.** Chunk held fixed at **2,048
fresh tokens (prefix 0)**, decode rows @96K each, sweeping the pool over 64×
(`gate_sweep_merged_qwen25_h800.json`):

| decode rows @96K | total decode KV | BatchPrefill | persistent | speedup |
|-----------------:|----------------:|-------------:|-----------:|--------:|
| 1 | 98K | 2.970 ms | 0.105 ms | **28.2×** |
| 2 | 197K | 2.979 ms | 0.169 ms | 17.7× |
| 4 | 393K | 2.983 ms | 0.298 ms | 10.0× |
| 8 | 786K | 4.655 ms | 0.574 ms | 8.1× |
| 16 | 1,573K | 4.709 ms | 1.093 ms | 4.3× |
| 32 | 3,146K | 4.725 ms | 2.131 ms | 2.2× |
| 64 | 6,291K | 5.049 ms | 4.274 ms | 1.2× |

BatchPrefill is **flat** (2.97 → 5.05 ms) while the work grows 64×; the
persistent kernel is **proportional** (0.105 → 4.274 ms), i.e. bandwidth-bound
as it should be. So the headline multiple is not a separate "decode-dominant"
phenomenon — it is just this ratio read at its left edge, where BP's plateau
is divided by the smallest real workload. It follows that the multiple is
largest with *few long* rows, not many: concentration, not aggregate KV, is
the predictor, which is why the gate carries a KV-concentration feature.

**Crossover by chunk prefix** (decode pools ≥ 250K total KV tokens):

| chunk prefix | median speedup | fraction >1.5× |
|-------------:|---------------:|---------------:|
| 0 – 8K | 1.27 – 1.72× | 46 – 57% |
| 16K | 1.17× | 36% |
| ≥ 32K | 1.02 – 1.07× | 0 – 11% |

So the rule is: **persistent wins when prefill chunks attend a prefix shorter
than roughly 16K against a substantial decode pool.** Realistic anchors, from
the fresh and blended data: 8 decodes @80K plus a fresh 2,048-token chunk is
**7.92×** (`probe_fresh_qwen25_h800.json`), and blended pools of 40–56 decodes
where only 4–8 are long (@64K, rest @2K) with a fresh 512–2,048-token chunk
give **6.90–7.50×** (`gate_sweep_blend_qwen25_h800.json`, whose maximum is
15.95× at 2 long rows @64K). Only 4–8 concurrent long-context decodes are
needed; short-KV decodes flooding the batch barely dilute the gain.

**Where persistent loses**, from the same merged table: 162/956 compositions
below 1.0× and 80 below 0.95×, all of them tiny batches where fixed overhead
dominates — worst cases are 1–4 decode rows @2–8K KV with a 512-token chunk
(0.57–0.70×, e.g. 0.027 ms vs 0.048 ms). Distribution over the merged sweep:
p10 = 0.97×, median = 1.24×, p90 = 4.88×, with 345/956 at ≥1.5×. These are the
compositions the gate must decline, and the reason its label margin is 1.10
with an out-of-support guard rather than a threshold rule.

**Work ordering.** The old flip heuristic (`total_kv < bs*32768`) is wrong on
this composition: mean 6.1% loss, worst 1.52×, 228/479 cases >5% off.
Non-flipped is better on 336 of the 368 cases where the schedule matters ≥5%,
so the gate learns the flip from the same neighbors.

**What this explains.** Pure TraceLab replay is the *worst* case for the
kernel: every chunk is a session continuation attending a 32–100K prefix —
the ≥32K row above, i.e. parity — and causal replay rarely has a prefill and
decodes pending at once. That is the whole reason the early serving results
were null, and why the win had to be looked for in composition rather than in
the kernel. Favourable and common in production: fresh arrivals into a busy
decode pool (chat + coding agent on one node), reasoning/thinking traffic
where decode KV grows from generated tokens while prompts stay short, and
cache-miss RAG/document traffic whose early chunks still have small prefixes.

How much of this is temporal prefill/decode overlap versus heterogeneity-
robust scheduling is separated in **Overlap attribution** below: overlap is
1.2–1.26× in the balanced band, scheduling supplies the rest. Part of the
scheduling gap is an fa2 planner limitation upstream could shrink (lift the
split budget, per-row tiles), which would narrow the decode-row penalty but
not supply dual tiles or overlap; the gate re-measures if the baseline
improves.

### Overlap attribution (2026-08-06)

`bench_overlap_attribution.py` isolates temporal prefill/decode overlap from
planner effects by comparing, with the same persistent kernel and planner,
the mixed launch against the two parts run separately:
`ovl_eff = (T_decode + T_prefill) / T_mixed`.

- Overlap contributes up to **1.26x**, only where the chunk's compute time
  balances the decode pool's memory time (T_p/T_d in roughly [0.5, 3]):
  2-4K short-prefix chunks against 16-32 long-context decodes. It decays to
  ~1.0 on both flanks (chunk-dominated or decode-dominated batches).
- The remainder of the mixed-batch win over BatchPrefill (up to ~2.4x at the
  same points, 8x at decode-dominated ones) comes from heterogeneity-robust
  scheduling, which upstream fa2 planner fixes could partially reclaim.
- Framing: the fused persistent formulation is what makes kernel-level
  overlap possible (dual tile classes co-resident in one launch; BP has one
  tile class per batch and cannot overlap by construction); the planner
  robustness is its prerequisite. An overlap-aware chunk-size controller
  (choose chunk so T_p ~ T_d against the live decode pool) would hold serving
  inside the 1.25x band instead of sampling it, and is the natural next step.

### The cost model is kernel-level; only its wrapper is server-level

Worth stating explicitly, because the model's *use* (EDF chunk control) is
server-level while its *fit* is not. `overlap_cost_model.py` is two layers:

| layer | form | constants | fit from | error |
|-------|------|-----------|----------|-------|
| decode attention | `T_d = a_d·decode_kv + c_d` | 2 | kernel times (`overlap_attribution.json`, n=35) | 2.04% MAPE, 4.8% max |
| chunk attention | `T_p = a_p·(n·prefix + n²/2) + c_p` | 2 | same, n=30 with a chunk | 3.47% MAPE, 7.0% max |
| **mixed composition** | `T_attn = max(T_p,T_d) + β·min` | 1 (β=0.5285) | same | **3.10% MAPE / 9.1% max in-sample; 3.22% / 9.7% leave-one-out** |
| step wrapper | `T_step = c0 + c1·tokens + L·T_attn` | 2 | **not fit** — derived from hardware (15.2 GB weights ÷ 3.35 TB/s + 0.6 ms; 2·params ÷ 600 TFLOP/s) | one serving anchor: 15.1 ms predicted vs 15.75 ms measured mean ITL |

So the kernel layer is measured, fit, and cross-validated (LOO barely moves the
error, so the one-parameter law is not overfit); the server layer is an
analytic roofline with a single sanity check.

**Only `predict()` is consumed** — `attn_layer_ms()` has exactly one caller
(line 114 of the same file), and the EDF controller asks only for whole-step
ms. So there is one model in use, not two; the kernel law is its interior.
That interior is doing three jobs:

1. **It is why the model needs no training.** The MLP it replaced had to learn
   `T_step(n_p, r_p, r_d, decode_bs)` end to end from **25,444 profiled
   configs** per model/GPU. The analytic model needs **35 kernel
   measurements** because the composition rule is known rather than learned —
   the structure does the work the data used to. Re-fitting for a new GPU is
   an afternoon, not a profiling campaign.
2. **It is where the paper's predictability claim is even visible.** "The
   split-kernel baseline has no closed form" is a *kernel-level* statement
   (fitted β=2.19; 13.3%/43% for the best parts-based fit). At step level the
   `c0 + c1·tokens` term is large enough to mask the attention pathology, so
   measuring only whole steps would hide exactly the effect being claimed.
3. **It is what an overlap-aware chunk controller would need.** Choosing a
   chunk so `T_p ≈ T_d` (holding serving inside the 1.2–1.26× overlap band
   instead of sampling it) requires the two terms *separately*, which the
   step total does not expose. Not implemented — see Open TODOs.

**Is BatchPrefill just hard for *that form*? (learned-model check, 2026-09-19.)**
Fitting `max + β·min` is a strong assumption, so the claim was re-tested with
flexible learners on the **same 1,316 sweep compositions, same 6 gate features,
same model class per row**, 5-fold CV × 3 seeds, log-target
(`fit_learned_cost_model.py`):

| features | model | target | MAPE | p95 | max |
|----------|-------|--------|-----:|----:|----:|
| composition (6) | MLP (128,64,32) | BatchPrefill | 9.9% | 27.2% | 85.3% |
| composition (6) | MLP (128,64,32) | persistent | **5.7%** | 15.4% | 40.7% |
| composition (6) | GBDT (400) | BatchPrefill | 10.1% | 30.5% | 102.7% |
| composition (6) | GBDT (400) | persistent | **3.3%** | 11.0% | 62.8% |
| + planner vars (10) | GBDT (400) | BatchPrefill | 8.6% | 25.7% | 101.1% |
| + planner vars (10) | GBDT (400) | persistent | **3.5%** | 11.2% | 60.5% |

Three conclusions, one of which corrects the earlier wording:

- **"Unmodelable" was too strong.** An MLP reaches 9.9% MAPE on BatchPrefill,
  better than the 13.3% of the best parts-based affine fit, so part of that
  13.3% was the functional form, not the kernel. The defensible claim is the
  *relative* one: on identical data, features and model class, BatchPrefill is
  **1.7–3× harder to predict** than the persistent kernel, with a far worse
  tail (85–103% max error vs 41–63%).
- **The planner variables recover only part of it.** Adding the batch-average
  packed-qo length and the CTA tile it implies — the exact quantity
  `FA2DetermineCtaTileQ` branches on — moves GBDT from 10.1% to 8.6% and
  leaves persistent unchanged. So the tile flip is a real but partial
  explanation; the residual is consistent with `PrefillBinarySearchKVChunkSize`
  depending on the full per-row KV vector, which no batch aggregate can carry.
  Replicating the planner, not fitting a smoother function, is what would close
  it — i.e. a simulator of one kernel version, not a portable model.
- **Fitting BP well would cost exactly what the analytic model saves.** These
  numbers need 1,316 measured compositions per GPU/model; the analytic law
  needs 35. And at 8.6–10% MAPE with 25–30% p95, a controller binary-searching
  chunk size against a 100 ms ITL budget would systematically over- or
  undershoot, where 3.3%/11% is workable.

Two scope limits:

- **It models the persistent kernel only.** There is deliberately no
  BatchPrefill equivalent — its mixed cost is super-additive in its parts
  (fitted β=2.19; best parts-based affine fit 13.3% MAPE / 43% max), which is
  why kernel *selection* needs a measured table rather than comparing two
  formulas.
- **Its fit grid is small and sits above the realized operating point**:
  decode rows {8, 12, 16, 32}, per-row KV {32K, 40K, 64K}, chunk {2K, 4K, 8K},
  prefix {0, 8K, 16K, 32K}. The SLO runs it actually drove operated at chunk
  p50 2,151 (inside) but decode rows p50 5 and total decode KV 87K (both
  *below* the grid) — extrapolation of a linear roofline below its fit range,
  which is the benign direction but is extrapolation. Same grid-vs-realized
  gap as the gate table above.

### Measured k-NN gate

Deployed version (v2, 2026-08-05, built by `build_mixed_kernel_gate_table.py`):
1,316 measured compositions, features `[log1p(decode_kv),
log1p(prefill_work=Σe·c), log1p(prefill_tokens), log1p(decode_bs),
3·D/(D+P+1), log1p(kv_concentration=Σkv²/Σkv)]`, k=5, persistent only on
unanimous neighbors labeled `bp/pa_best ≥ 1.10`, flip by neighbor majority,
plus an out-of-support guard (query farther than `max_nn_dist_sq` from every
table point → BatchPrefill). Leave-one-out: worst-case regression **1.000**,
win capture 81%, mean speedup 2.71× on accepted batches. The original v1
(479 points, 5 features, k=3, label ≥1.05, LOO worst 1.028) is what the
2026-07-30 validation screens below ran; v2 adds the fresh-prefix and
blended-pool sweeps and is what every mixed-tenancy run uses. Deployed as
`python/sglang/srt/layers/attention/mixed_kernel_gate_h800_gqa7.json`
(+gqa4 for Qwen3-8B) behind `SGLANG_MIXED_ATTENTION_KERNEL=auto` with
forced `persistent`/`batch_prefill` overrides. Decision latency on the
current 1,316-point/k=5 table (re-measured 2026-08-13): ~39 µs for the core
k-NN, 52–69 µs for the full entry point including feature computation from
per-request CPU lists (batch 16→256); once per mixed step, CPU-side, ~12x
cheaper than the 0.47 ms sync-free plan it precedes.

### Validation screens (TraceLab decode-heavy, N=100, 2026-07-30)

sglang `313b97611`, flashinfer `c6d2a00`. Command:
`TRACELAB_MIN_OUTPUT_LEN=512 TRACELAB_NUM_GROUPS=<g> NUM_PROMPTS=100
LPT_CHUNKED_PREFILL_SIZE=6656 METHODS_SPEC="DMC_CG LPT"
bash tasks/realistic_workload_analysis/run_tracelab_claude.sh`, results in
`tasks/realistic_workload_analysis/h800_tracelab_decodeheavy512_gated_g{10,16}_n100/`.
These screens ran the v1 gate (479 points/k=3); their zero-regression
conclusion was re-confirmed under the v2 gate by the mixed-tenancy controls.
DMC is a true BatchPrefill baseline here (explicit
`SGLANG_FLASHINFER_USE_PERSISTENT=0`, see Setup).

The first g10 LPT run exposed a mid-replay JIT compile of the second
flipped-schedule module variant (18.3 s global stall freezing all nine active
sessions; both variants' cached_ops rebuilt at 19:16/19:19 in the server
process). Fixed by prewarming both variants at backend init
(flashinfer `c6d2a00` + sglang `313b97611`); the polluted run is archived in
the session scratchpad and was re-run.

| Screen | Method (cap) | Total tok/s | Mean/P99 TTFT | Mean ITL | pooled P99 ITL | reqmax P99 |
|--------|--------------|------------:|--------------:|---------:|---------------:|-----------:|
| g16 (conc 10.9) | true DMC (8192) | 7,882 | 31.0 / 90.2 s | 42.88 ms | 775.97 ms | 1,367 ms |
| g16 (conc 11.0) | gated LPT (6656) | **7,943 (+0.77%)** | 31.8 / 89.8 s | **42.28 ms** | **662.92 ms (−14.6%)** | **1,205 ms** |
| g10 (conc 3.2) | true DMC (8192) | 6,382 | 3.27 / 40.2 s | 17.52 ms | 31.36 ms | 1,075 ms |
| g10 (conc 3.2) | gated LPT (6656) | 6,375 (−0.11%) | 3.29 / 41.1 s | 17.57 ms | 35.01 ms (+11.6%) | **886 ms (−18%)** |
| g10 (conc 3.2) | gated LPT (8192) | **6,386 (+0.07%)** | 3.37 / **40.0 s** | **17.48 ms** | **29.41 ms (−6.2%)** | 1,072 ms |

At g16 the gate chose BatchPrefill for 482/500 mixed batches — the 6,656-token
chunks against ~80K prefixes sit in BatchPrefill's win region, so the pooled
P99 gain comes from the cap, the corrected schedule choices, and the 18
persistent picks. Both methods retokenize 100% and share one 32-s KV-pressure
outlier on a single request.

The g10 pooled-P99 gap at cap 6656 is a **cap artifact, not a kernel
artifact**: with the gate routing nearly all mixed batches to BatchPrefill,
the only remaining config difference was the LPT-only 6,656-token cap (more,
shorter interference windows). Matching the baseline cap (8192) turns the old
"+14.5% pooled P99 ITL regression at 10 sessions" into **−6.2%** with parity
on throughput/mean ITL/P99 TTFT/request-max. Cap 6656 remains the
tail-optimized point (−18% request-max at g10, −14.6% pooled P99 at g16).

At the default cap 8192, g16 gated LPT is parity on throughput/pooled P99
(−0.43%/−1.33%) with request-max P99 **−23%** (1,053 vs 1,367 ms) and no
32-s outlier; the −14.6% pooled-P99 point requires cap 6656.

### WL1 screens, Qwen3-8B / gqa4 table (N=100, `run_gain_search.sh`, 2026-07-30)

`MODEL=/opt/tiger/models/Qwen3-8B CASES_SPEC="4096:1024:8 4096:1024:2"
NUM_PROMPTS=100 METHODS_SPEC="DMC_CG LPT"`, results in
`tasks/debug_kernel_server_perf/wl1_gated_check_n100/`. Both methods
retokenize 100%.

| Rate | LPT vs true DMC |
|-----:|-----------------|
| 8 | tput +0.16%, mean TTFT −2.6%, mean ITL −0.5%, pooled P99 ITL +0.5% |
| 2 | tput −0.01%, mean TTFT −1.4%, mean ITL −0.1%, pooled P99 ITL −2.6% |

Zero regression on the Qwen3-8B head config at both loads.

### Paired repeat runs (default cap 8192, `*_cap8192_rep2_n100/`)

| Screen | Run | LPT vs true DMC |
|--------|-----|-----------------|
| g16 | 1 | tput −0.43%, mean TTFT +6.8%, pooled P99 ITL −1.3%, reqmax P99 −23% |
| g16 | 2 | tput **+2.10%**, mean TTFT **−6.0%**, pooled P99 ITL **−4.1%**, reqmax parity |
| g10 | 1 | tput +0.07%, mean TTFT +3.2%, pooled P99 ITL −6.2%, reqmax parity |
| g10 | 2 | tput −0.03%, mean TTFT −0.6%, pooled P99 ITL −0.7%, reqmax parity |

Mean TTFT at saturation swings ±6% between identical-config runs, so the
run-1 TTFT deltas were noise. Averaged, the default configuration is parity
or better on every axis at both loads.

### Conclusion

The low-concurrency regression of the persistent overlap path is eliminated
and the high-concurrency gains are retained against a strictly stronger
(honest BatchPrefill) baseline:

- **Default config** (gate auto, cap 8192): zero regression at 10 sessions
  (all axes within noise, pooled P99 ITL −0.7–6.2% better), and at 16
  sessions +0.8% mean throughput across paired runs with −1.3–4.1% pooled
  P99 ITL and up to −23% request-max P99.
- **Tail-optimized** (cap 6656): −14.6% pooled P99 ITL and +0.77% throughput
  at 16 sessions; at 10 sessions request-max P99 −18% while pooled P99 pays
  +11.6% (a cap tradeoff, not a kernel one).
- Kernel level, the gate preserves the persistent kernel's wins where they
  exist — up to 21.0× in the uniform sweep, 28.2× once fresh prefixes are
  sampled, on batches with few long decode rows and a short-prefix chunk —
  while capping any composition's regression (v1 gate: 2.8% leave-one-out;
  deployed v2: 1.000, i.e. none). WL1 on Qwen3-8B is parity at both rates
  with 100% retokenization everywhere.

Mechanically, the three fixes were: (1) sync-free double-buffered planning
(56.5 ms pipeline stall → 0.47 ms), (2) the measured per-batch kernel +
schedule gate replacing always-persistent and the wrong flip heuristic, and
(3) startup prewarm of both persistent module variants (a mid-replay JIT
build froze all sessions for 18 s). Every number below is against an honest
BatchPrefill baseline (see Setup).

**Phase 2 (2026-08-04..06), the current headline results:**

- **Mixed tenancy is the win scenario** (`run_mixed_workloads.py
  agent_plus_shortchat`, honest baseline, identical tokens, 100% retok):
  coding tenant **+36.5% throughput** (188→138 s wall), chat tenant
  **−42% mean ITL / −23% P99 / −37% TTFT**; `shortchat_only` control at
  parity with the gate declining persistent on 2,560/2,566 mixed batches.
  Details in `tasks/realistic_workload_analysis.md`.
- **The win region is lawful**: per-request KV concentration × chunk prefix
  (< ~16K) decide it; the gate table is 1,316 points with a KV-concentration
  feature and an out-of-support guard (LOO worst regression 1.000).
- **Overlap attribution**: temporal PD overlap contributes 1.2–1.26× in the
  balanced band T_p ≈ T_d; the rest of the mixed-batch win is
  heterogeneity-robust scheduling (BP's planner collapses under one chunk:
  tile flip + split-budget exhaustion, 3.1 ms plateau).
- **Analytic cost model** (`SGLANG_EDF_LATENCY_MODEL=analytic`):
  `T_attn = max(T_p,T_d) + 0.53·min` at 3.1% MAPE/9% max, 7 physical
  constants, no training; BP's best parts-based fit is 13.3%/43% with
  fitted β=2.19 (super-additive). Predictable cost is what SLO scheduling
  consumes; the fused kernel is what makes cost predictable.

## Part 2 — What the real workload actually looks like

Captured compositions, and the scope limit that bounds every result above and
below.

### Captured mixed-batch workload (2026-09-22)

The gate table and every kernel sweep so far were built on a Cartesian grid.
`SGLANG_LOG_MIXED_COMPOSITION=<path>` now records the **real** composition of
every mixed batch from `select_mixed_attention_kernel` — per-row decode KV, the
chunk's token count, its *true attended prefix* (not `#cached-token`, which is
0 for continued chunks), and the gate's decision — so sweeps can be resampled
from the realized distribution. Captured across a co-tenancy load sweep
(`run_mixed_workloads.py --scenario loadsweep_{hi,med,lo,min}`, agent + chat
at 16/8/4/1.5 chat req/s), analyzed by `analyze_captured_workload.py`, replayed
by `sweep_from_capture.py`.

| load point | mixed batches | decode rows p10/50/90 | chunk tokens p10/50/90 | attended prefix p50 | decode KV p50 | gate→persistent |
|------------|--------------:|----------------------:|-----------------------:|--------------------:|--------------:|----------------:|
| hi (chat 16/s, 12 sessions) | 1,792 | 33 / **46** / 47 | 288 / **411** / 501 | 2 | 145K | **77%** |
| med (8/s, 8 sessions) | 1,088 | 2 / **7** / 47 | 280 / **395** / 496 | 2 | 45K | 34% |
| lo (4/s, 6 sessions) | 512 | 1 / **4** / 46 | 283 / **398** / 502 | 1 | 2K | 31% |
| min (1.5/s, 4 sessions) | 128 | 1 / **2** / 12 | 296 / **446** / 8186 | 1 | 1K | 23% |

**Correction (2026-09-23): those ~400-token "chunks" are not chunks.** The
distribution is bimodal — 4,113 samples at p10/p50/p90 = 278/**382**/484 and
304 at p50 8,145. The small mode is the *chat tenant's entire prompt*
(`--random-input 512 --random-range-ratio 0.5` → uniform[256,512], whose
p10/p50/p90 is 280/385/489), prefilled in one shot because 512 ≪
`chunked_prefill_size=8192`. The large mode is the codex tenant genuinely
chunked at the 8,192 cap. So chunk size is load-invariant because **it is a
scenario parameter I chose**, not because it is a leftover token budget — the
earlier explanation here was wrong, and it was not EDF either (the capture runs
used `schedule_policy='fcfs'`, `enable_priority_scheduling=False`). What
genuinely moves with load is the decode population (46 → 2 rows) and with it
the gate's persistent share (77% → 23%).

This limits how far the capture generalizes: the chunk axis of the captured
distribution is set by the chat prompt-length mix, so the gate-v3 gain is
established *for this workload* and would need re-capturing under a different
prompt-length profile.
Prefixes are essentially always fresh (73–96% below 512 tokens) — this is the
co-tenancy shape, not session replay.

The mismatch with the sweep grid is stark once chunk mass is binned against
the grid's own sample points:

| chunk bin | grid sample points in bin | share of real chunks |
|-----------|--------------------------:|---------------------:|
| 0 – 128 | 0 | 1.3% |
| **128 – 512** | **1** (just 128) | **91.4%** |
| 512 – 2,048 | 2 | 0.8% |
| 2,048 – 8,192 | 3 | 6.5% |
| ≥ 8,192 | 1 | 0.0% |

**Nine out of ten real chunks fall in a range the grid samples with a single
point**, while five of the grid's seven chunk values sit where 7% of traffic
lands. (An earlier "mean distance to nearest grid point" statistic read 1.5,
but it is dominated by the 1.3% tail of 2–16-token chunks and overstates the
problem; the binned mass above is the honest version.)

**On the low attended-prefix numbers above — this is *not* a cache-miss
workload (corrected 2026-09-28).** The same runs show a **79.7% radix cache hit
rate** in token terms (36,363 new vs 142,723 cached), with peak KV-pool
occupancy only 0.61, so nothing is being evicted. The per-row prefix
distribution is *bimodal*: the coding tenant's session continuations hit cache
heavily but are a small fraction of prefill **rows** (only 3 of 43 sampled
mixed batches carry `cached-token > 512`, yet those carry 3.8M of the cached
tokens), while the chat tenant's fresh random prompts have no reusable prefix
**by construction** and dominate the row count. So "prefix p50 = 2" is a true
statement about rows and a misleading one about caching. The prefix-reuse
capture (`h800_capture_prefixreuse`, multi-turn on 8K shared prefixes) is the
counterpart with prefix p50 8,592 and 56% of rows ≥ 8K.

The shape of real mixed batches is not what the grid samples: at high load a
mixed batch is **~46 decode rows against a ~400-token, zero-prefix chunk**,
where the grid's chunk axis starts at 128 and jumps to 512, and its decode-row
axis jumps 32 → 64. The mean relative distance from a realized chunk to the
nearest grid point is >1, i.e. the typical real batch is *farther from the
nearest sampled point than that point's own magnitude* — the gate is
interpolating, and its out-of-support guard is what keeps that safe.

### Scope limit: all of this needs a GQA ratio ≤ 16 (2026-09-23)

A decode row's `packed_qo_len` is `1 × gqa_ratio`, and two thresholds decide
whether any of the tile heterogeneity this work exploits exists at all:
the persistent kernel routes to its small tile only when `packed_qo_len <= 16`
(`CTA_TILE_Q_SIZES = {128, 16}`), and FA2 picks tile 16 for a decode-only batch
only when the batch-average is ≤ 16.

| model | qo/kv | ratio | decode packed_qo | FA2 tile | persistent tile |
|-------|------:|------:|-----------------:|---------:|----------------:|
| Qwen2.5-7B (ours) | 28/4 | 7 | 7 | 16 | **small (16)** |
| Llama-3.3-70B | 64/8 | 8 | 8 | 16 | small (16) |
| Qwen3-235B-A22B | 64/4 | 16 | 16 | 16 | small (16) — at the boundary |
| **Qwen3.8-Flash-Next** | **24/2** | **12** | 12 | 16 | **small (16)** |
| Kimi-K2 (MLA) | 64/1 | 64 | 64 | 64 | **large (128)** |
| DeepSeek-V3/V4 (MLA) | 128/1 | 128 | 128 | 128 | **large (128)** |

**At ratio > 16 the entire premise disappears.** Decode rows already live in the
large tile class, so there is no heterogeneity for the dual-tile kernel to
exploit — and, symmetrically, no 16→128 flip for one prefill chunk to cause,
which is the FA2 pathology the gate was built around. MLA models are also
routed to different backends entirely (`flashinfer_mla_backend.py`,
`flashmla_backend.py`, `trtllm_mla_backend.py`, `cutlass_mla_backend.py`);
`select_mixed_attention_kernel` exists only in `flashinfer_backend.py` and is
never reached from them.

**Qwen3.8-Flash-Next (24/2, ratio 12) is the notable modern exception** — it
keeps a small GQA ratio, so the dual-tile premise survives. But three details
of its config blunt the win anyway: `head_dim = 256` makes FA2's 128 tile
unreachable (`FA2DetermineCtaTileQ` requires `head_dim < 256`), so the flip
there is 16→64 rather than 16→128; only **12 of 48 layers are full attention**
(3 linear : 1 full), so attention is a quarter of the step it is for a dense
model; and a DSA-style sparse indexer with `indexer_budget = 2048` caps how
much KV any row attends, which flattens exactly the KV-concentration
heterogeneity the gate keys on.

So the win region is **low-GQA-ratio models on the standard flashinfer
backend**. That covers the Llama/Qwen-dense family (ratios 7–16), and excludes
the MLA frontier models where most current serving attention is going. The
`m64n16k16`-vs-`m64n64k16` instruction study is scoped the same way: with no
small tiles, its small-tile slack finding does not apply.

## Part 3 — Hopper kernels: FA2, FA3 and the persistent kernel

Planner structure and instruction-level behaviour of the three kernels the
dispatcher has to choose between.

### Hopper instruction-level scheduling study (2026-09-19..22)

What scheduling headroom exists for mixing the two tile classes on Hopper?
Five microbenchmarks in `tasks/debug_kernel_server_perf/wgmma/`, all on H800
sm_90a, all reporting *marginal* cycles (a 128→256 iteration delta, so fixed
overhead cancels), min of 25–30 runs.

| # | question | result |
|---|----------|--------|
| 1 | two wgmma **shapes**, one warpgroup, interleaved vs grouped | **no overlap**: 32.00 (n64) + 20.00 (n16) = 52.00 either way |
| 2 | two wgmma shapes, **two warpgroups**, same SM | **no overlap**: 52.10 |
| 3 | **exp2** co-issued under an in-flight wgmma, same warpgroup | **not hidden**: +50.5 cyc for 8 ops (large), +56.8 (small) |
| 4 | **FFMA** co-issued under an in-flight wgmma, same warpgroup | **small tiles hide it free**: n16 20.00→20.00 for 8 ops; n64 pays +3.5 |
| 5 | wgmma on wg0 ‖ **exp2 on wg1** (different pipes) | **overlaps**: ~23.5 of the 32 MMA cycles hide, 73% |

Read together these say where the headroom is and is not:

- **The tensor core is one serial per-SM resource.** Neither mixing wgmma
  shapes nor splitting them across warpgroups buys anything (#1, #2) — costs
  add to the cycle. Issue order is irrelevant (interleaved vs grouped differ
  by a constant 16 cycles, 0.00% marginal). Validated against hardware peak:
  `m64n64k16` = 131,072 FLOP → 30.7 cycles theoretical vs 32.0 measured,
  **96% of peak**.
- **Small tiles carry free issue slack, large ones nearly none (#4).** Cost is
  ≈ `16 + N/4` cycles, so `n16` runs at 38% of peak (1.25 cyc per unit N vs
  0.50 for `n64`) — but that fixed overhead is *usable*: a small tile absorbs
  8 FFMA ops at **zero** added cost where a large tile pays 3.5. This is the
  "squeeze other ops into the small-tile shadow" lever, and it is real but
  bounded (~8 cycles of CUDA-core work per small wgmma).
- **Softmax cannot hide inside a warpgroup, only beside one (#3 vs #5).**
  exp2 under an in-flight wgmma costs nearly its full price, but exp2 on a
  *different warpgroup* hides ~73% of the MMA. That asymmetry is exactly why
  FA3 warp-specializes, and it is the one lever with double-digit headroom
  here. The saving is ~23.5 cycles per MMA regardless of softmax size, i.e.
  it is the MMA hiding under the SFU work, so the win is capped by how much
  softmax the composition actually has.

Consequence for β (the 0.53 non-overlappable fraction): **it is not the MMA
pipe**, and no reordering of tile classes will recover it. Prefill/decode
overlap must come from MMA-vs-memory and MMA-vs-SFU concurrency, which means
the remaining budget is in the reduction pass, queue-switch granularity, L2
interference, or in keeping *both* pipes fed — the last of which is what the
FA3 direction below targets.

*Method caveats:* descriptors address valid aligned smem but the data is
arbitrary (wgmma cost depends on shape and dependencies, not values).
Accumulators stay in registers and each shape's first issue uses `scale-d=0`;
without that ptxas emits C7515 and serializes the pipeline. Any runtime branch
inside the issue loop blocks unrolling and inflates the baseline 3× (32 → 105
cycles), so all variants are compile-time template parameters. Both traps
silently produced "no overlap" answers in earlier drafts of these benchmarks.

### Mixing wgmma with mma.sync: no overlap, and worse than additive (2026-09-26)

A decode/draft row has `packed_qo_len = gqa_ratio` (4–16), but wgmma's M is
fixed at 64, so such a row wastes 4–16× of the M dimension. `mma.sync`
(`m16n8k16`) has M=16 and would waste far less — so: can small rows run on
mma.sync *underneath* the large rows' wgmma? Measured with the same
methodology as the other probes here (`wgmma_mma_sync.cu`, marginal cycles from
a 128→256 iteration delta, one warpgroup, min of 25):

| mma.sync per iter | mma alone | wgmma alone | sum | **interleaved** | excess |
|------------------:|----------:|------------:|----:|----------------:|-------:|
| 2 | 48.00 | 32.00 | 80.00 | **100.62** | **+20.6** |
| 4 | 96.00 | 32.00 | 128.00 | **158.75** | **+30.8** |
| 8 | 192.00 | 32.00 | 224.00 | **255.00** | **+31.0** |
| 16 | 384.00 | 32.00 | 416.00 | **446.75** | **+30.8** |

Not merely additive — **anti-additive**: interleaving the two instruction
classes costs a roughly constant **~24–31 cycles extra** on top of the sum,
consistent with a pipeline drain or mode switch between the async
warpgroup-wide path and the synchronous warp-level one. There is no overlap to
harvest.

The idea fails on throughput grounds too, independently of the interference.
`mma.sync m16n8k16` costs a flat 24 cycles per instruction; with 4 warps that
is 683 FLOP/cycle, **16% of H800 peak**, against wgmma's 4,096 FLOP/cycle
(**96% of peak**) — a 6× deficit. Accounting for M-padding, useful throughput
for a row of `packed_qo = p` is `4096·p/64` for wgmma versus `683·p/16` for
mma.sync, i.e. **wgmma wins by a constant 1.5× at every row size**: the 4×
padding advantage never covers the 6× throughput deficit. So small rows are
better served by a *padded wgmma* than by mma.sync, which is presumably why
the persistent kernel's small tile class is 16 rows of wgmma rather than a
different instruction.

Consequence for the fused 3-mode kernel: the third (verify) tile class should
be another wgmma tile, not an mma.sync path.

### Ours vs FA3: the complete picture (2026-09-22..27)

All FA2 / FA3 / persistent comparisons live here; the serving-level verdict is
in `realistic_workload_analysis.md`, which points back to this section.

**Planner structure.** Reading the three schedulers side by side
(`flashinfer/include/flashinfer/attention/scheduler.cuh`) reframes the gate:

| | q-tile selection | KV-split sizing | load balancing | group heads |
|---|---|---|---|---|
| **FA2** (`PrefillPlan :696`) | batch-wide adaptive 16/64/128 via `FA2DetermineCtaTileQ` — **one chunk flips it for every row** | fixed budget `max_grid_size/num_kv_heads` = 66 units — **one chunk eats 56** | none (static split) | fused (packed) |
| **FA3** (`PrefillSM90Plan :869`) | **fixed `cta_tile_q = 128`** (192 iff `head_dim_vo == 64`), no flip | **none — no KV splitting at all** | MinHeap over `num_sm` CTAs | **not fused**, one CTA per qo head |
| **persistent** (`TwoStageHolisticPlan :1111`) | **`{128, 64, 16}`**, per row by `packed_qo_len` | `f(total_kv×heads / num_clusters)` | MinHeap per task queue | fused (packed) |

**Corrected 2026-09-30.** An earlier version of this table gave FA3
`cta_tile_q = 64` and a `f(total_kv_lens / num_clusters)` KV split, citing
`:1468` and `:1507`. Both of those lines are inside `TwoStageHolisticPlan` —
*our* planner — so the row described the persistent kernel's traits and
attributed them to FA3. FA3's actual planner is `PrefillSM90Plan`, and its two
properties that matter are in the table above: a 128-wide q tile and **no KV
splitting**.

Consequences, all confirmed in Part 4's boundary section:

* FA3 has fixed both FA2 pathologies the gate was built around (no batch-wide
  tile flip, MinHeap balancing).
* Its tile waste on short rows is **worse** than previously stated and for a
  different reason: because it does not fuse group heads, a GQA-7 decode row
  occupies `1/128` of a tile, not `7/64`, and a 4-token verify row `4/128`
  regardless of the GQA ratio. That is why routing verify rows to FA3 measured
  14× slower.
* Our prefill tile is **equal** to FA3's (both 128), not larger as previously
  claimed.
* Having no KV split caps FA3's parallel width at `h_qo * ceil(qo_len/128)`,
  which is the single reason it loses at low `h_qo` — see Part 4.

**Who wins depends on the regime, not the kernel.** Measured on isolated
compositions (`fa3_prefill_probe.py`, H800, gqa 7, µs):

| composition | FA2 | FA3 | persistent | FA3/FA2 | pers/FA2 |
|---|---:|---:|---:|---:|---:|
| chunk 2,048 @ prefix 0 | 19.4 | 49.9 | 20.8 | 0.39× | 0.93× |
| chunk 2,048 @ prefix 8,192 | 778.0 | **503.9** | 865.7 | **1.54×** | 0.90× |
| chunk 8,192 @ prefix 32,768 | 10,597 | **7,547** | 10,706 | **1.40×** | 0.99× |
| 2 decode @8K + chunk 4,096 | 274.9 | 178.3 | **53.5** | 1.54× | **5.13×** |
| 8 decode @8K + chunk 4,096 | 406.7 | 309.5 | **89.2** | 1.31× | **4.56×** |
| 32 decode @8K + chunk 4,096 | 421.5 | 1,060 | **219.0** | 0.40× | **1.92×** |

**FA3 is 1.35–1.54× faster than FA2 on real prefill work** (long attended
prefix), where the persistent kernel is only 0.89–1.00× of FA2. With a prefix
of 0 there is no attention work to speed up and only fixed cost remains, which
is why FA3 looks bad there. The crossover is **~4 decode rows**.

**Time-weighted over real serving compositions.** Two captures, replayed
through all three kernels (`sweep_from_capture.py`):

| workload | FA2 | FA3 | persistent | gate | oracle best-of-3 |
|---|---:|---:|---:|---:|---:|
| co-tenancy, fresh chat prompts (2,845 batches) | 1.00× | 0.67× | **2.60×** | 1.90× | 2.60× |
| multi-turn w/ 8K shared prefixes (365 batches) | 1.00× | 0.68× | **1.34×** | 1.18× | **1.44×** |

**The persistent kernel wins both, but the margin halves** (2.60× → 1.34×) once
the workload contains real prefill work. Decode-row split on the prefix-reuse
workload:

| split | weight | FA3/FA2 | pers/FA2 | pers/FA3 |
|---|---:|---:|---:|---:|
| 1–3 decode rows | 74 | **1.39×** | 0.98× | **0.70×** |
| 4–15 rows | 127 | 1.30× | **1.48×** | 1.14× |
| 16–47 rows | 164 | 0.55× | **1.34×** | 2.43× |

So **FA3 wins outright on prefill-dominant batches** (1.43× over ours at 1–3
decode rows) and loses as the decode population grows. Its aggregate 0.68× is
not a verdict on FA3; it reflects where these workloads spend their time. The
**oracle best-of-3 is 1.44× vs 1.34% for always-persistent**, so ~7% is left on
the table by never considering FA3 — motivating multi-kernel dispatch.

**Levers tried on FA3, none landed.** (1) Overlapping differently-shaped wgmma
is ruled out at the instruction level (see the scheduling study above: costs are
additive within a warpgroup *and* across warpgroups). (2) Squeezing work into
small-tile slack is real but needs a small tile class; FA3's `cta_tile_q` is
fixed in its scheduler and `DISPATCH_CTA_TILE_Q` appears only in the FA2 and
POD paths, never the Hopper template — so it is a mainloop rewrite, not a
scheduler tweak. (3) FA3's own batch-average decision (`cluster_size = 2` when
`avg_packed_qo_len > 64`) does **not** reproduce FA2's flip: sweeping chunk size
across that exact boundary at 46 rows @3K KV, FA3 is flat at ~565 µs while
**FA2 steps +29% (124 → 160 µs)** and the persistent kernel is flat to within
1%. `fixed_split_size`, the one exposed FA3 knob, is inert (537 µs unset,
564–568 µs at every setting). FA3's deficit on decode-heavy batches is inside
its mainloop, unreachable from the scheduler.

## Part 4 — Mixed-tile fusion and 4-mode dispatch

Speculative decoding puts four differently shaped work classes in one batch.
This part covers the fused mixed-tile kernel, the third q-tile class, and the
final measured answer on separate kernels with scheduler-partitioned grids.

### Fusing spec-dec (draft + verify + prefill) into one mixed-tile kernel (2026-09-25)

Question: for GQA models, can a megakernel that fuses speculative draft,
verify, and prefill give PD-colocation the functionality of PD-disaggregation
with spec-dec on the decode nodes — and does mixed-tile buy anything there?

**Prior art.** SparseSpec (Zhao, Tang, Zhu, Ye et al., *Accelerating Large-Scale
Reasoning Model Inference with Sparse Self-Speculative Decoding*, MLSys 2026)
already does two thirds of this and validates the premise. It is
self-speculative (one model drafts sparsely via PillarAttn and verifies
densely), and its two relevant contributions are exactly the shape of ours:

- a **unified scheduler** that evenly mixes draft and verify requests in every
  batch by greedy bin-packing into `k` buckets, rather than running `k` draft
  steps then one verify step. Sequential costs
  `k·T_GEMM(B) + T_GEMM((k+1)B)`; uniform costs `(k+1)·T_GEMM(((2k+1)/(k+1))B)`
  — "up to 2× longer" for the naive order. Worth **1.23×** in their ablation.
- a **fused attention kernel in persistent-kernel style** which "on-chip
  dispatches verification and draft attention into their best template instead
  of launching two kernels". Their motivation is our tile argument measured on
  FlashInfer: a kernel tuned for verify/full attention hits ~85% of bandwidth
  but **<50%** on draft/sparse, while a sparse-tuned kernel gets >80% but drops
  to **50%** on full. The fused kernel is **1.3× over two sequential launches
  and 1.8× over naive batching**.

Evaluated on Qwen3-1.7B/8B/14B, DeepSeek-R1-Distill-Llama3-8B, QwQ-32B (all
GQA, H100, TP1–4), k=8, sparsity s=0.05, up to 2.13× end-to-end.

So mixed-tile benefit in a fused draft+verify kernel is real and published. The
open part is adding **prefill** as a third class, which is what PD colocation
needs and what SparseSpec does not do.

**Spec-dec adds a genuinely third tile class.** A row's `packed_qo_len` is
`qo_len × gqa_ratio`; a verify row carries `k+1` candidate tokens, so it lands
*between* decode and prefill:

| work type | qo_len | packed (r=7) | FA2 tile | our kernel `{128,16}` |
|-----------|-------:|-------------:|---------:|----------------------:|
| decode / draft step | 1 | 7 | 16 | 16 |
| **verify (k=3)** | 4 | 28 | **64** | **128 ← rounded up** |
| **verify (k=7)** | 8 | 56 | **64** | **128 ← rounded up** |
| prefill chunk 400 | 400 | 2800 | 128 | 128 |

The verify class is distinct exactly when `16 < (k+1)·r ≤ 64`:

| model (ratio) | k=1 | k=2 | k=3 | k=4 | k=5 | k=7 | k=8 | k=11 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B (4) | draft | draft | draft | **64** | **64** | **64** | **64** | **64** |
| Qwen2.5-7B (7) | draft | **64** | **64** | **64** | **64** | **64** | **64** | prefill |
| Llama-3.3-70B (8) | draft | **64** | **64** | **64** | **64** | **64** | prefill | prefill |
| Qwen3-235B (16) | **64** | **64** | **64** | prefill | prefill | prefill | prefill | prefill |

Typical depths (k=3–8) land in the band for ratios 5–16. **Our dual-tile
`{128, 16}` is therefore insufficient for the fused kernel** — it must round
verify rows up to the 128 tile, e.g. 4.6× padding for a `packed=28` verify row
against FA2's 64. A third class is the concrete kernel change.

**The fused batch is a *better* fit for the persistent kernel than the
prefill/decode mix we have been studying**, because SparseSpec's own schedule
builds in KV-length heterogeneity on top of tile heterogeneity: draft rows read
`s·M` (5%) of the context while verify rows read all of `M`. Instantiating
their parameters (B=128, k=8, s=0.05):

| context M | draft rows | verify rows | total decode KV | KV-concentration | row-length spread |
|----------:|-----------:|------------:|----------------:|-----------------:|------------------:|
| 8K | 114 @409 | 14 @8,192 | 161K | 5.9K | **20×** |
| 32K | 114 @1,638 | 14 @32,768 | 645K | 23.8K | **20×** |

Few long rows among many short ones is precisely the composition that starves
BatchPrefill of query tiles and that our KV-concentration gate feature keys on
— here it arises by construction rather than by luck of co-tenancy.

**Measured: the missing third tile class costs 13–27%, and it is exactly what
makes fusion worth doing.** `specdec_tile_probe.py`, H800, 32 rows @8K KV,
BatchPrefill (which picks tile 64) vs our persistent kernel (which must round
to 128). The penalty appears *only* inside the predicted band and vanishes on
both sides of it:

| | k+1=1 | k+1=2 | k+1=4 | k+1=8 | k+1=16 |
|---|---|---|---|---|---|
| **Qwen3-8B** (r=4) | 1.03× | 1.03× | 1.03× | **1.23×** | **1.27×** |
| **Qwen2.5-7B** (r=7) | 1.02× | 1.02× | **1.15×** | **1.13×** | 1.00× |
| **Qwen3-235B** (r=16) | 1.01× | **1.16×** | **1.14×** | 1.04× | 0.96× |

(persistent / BatchPrefill; >1 means our kernel is slower). Bold cells are
exactly where `16 < (k+1)·r ≤ 64` predicts a distinct tile-64 class, and the
band shifts left as `r` grows, as derived. Outside it the two kernels agree to
within 1–4%.

**Fusion itself pays only for a heterogeneity-robust kernel** (same probe,
32 rows @8K, 24 draft + 8 verify(k=3) + one 400-token chunk):

| | 3 separate launches | fused | fused / separate |
|---|---:|---:|---:|
| BatchPrefill | 235.1 µs | 400.2 µs | **0.59× (fusing hurts 1.7×)** |
| persistent | 264.9 µs | 230.8 µs | **1.15× (fusing helps)** |

Fusing a chunk into the batch flips FA2's batch-wide tile and wrecks it, which
is the same pathology as the prefill/decode case — so a fused megakernel is
only viable on a per-row-tile design. But the net today is
`best fused 230.8 µs vs best separate 235.1 µs = 1.02×`: **the ~1.15× fusion
gain is almost exactly cancelled by the ~1.13–1.15× verify-tile penalty.**
Adding the third class is what converts this from break-even into a real win,
and it is a bounded change — a third entry in `CTA_TILE_Q_SIZES` plus a third
work queue, not a new mainloop.

**Blocker, confirmed in this tree.** The fused batch cannot be expressed today:
`ForwardMode` makes `MIXED`, `TARGET_VERIFY` and `DRAFT_EXTEND` disjoint enum
values, each taking its own kernel launch, and **every speculative algorithm
force-disables mixed chunking** — `server_args.py:1505` for
EAGLE/EAGLE3/STANDALONE and `:1578` for NGRAM. The running server says so
itself: *"The overlap scheduler and mixed chunked prefill are disabled because
of using ngram speculative decoding."* So enabling spec-dec today costs you PD
colocation outright, which is exactly why the standard deployment is
PD-disaggregation with spec-dec confined to decode nodes. Fusing the three
modes is what would give colocation parity.

A capture attempt confirms it empirically: running Qwen2.5-7B with
`--speculative-algorithm NGRAM --speculative-num-steps 3
--speculative-num-draft-tokens 4` under a 300-request chat load logged **192
EXTEND batches and zero MIXED batches** — prefill ran as standalone single-row
launches throughout, never sharing a batch with decode/verify work. (Verify and
draft steps execute through the CUDA-graph replay path, which bypasses
`init_forward_metadata` and so the composition hook — the same graph-replay
blind spot noted for decode above; capturing those needs the hook extended to
`init_forward_metadata_replay_cuda_graph`.)

**Proposed next step**, in dependency order: (1) add a third `CTA_TILE_Q_SIZES`
entry (64) and a third work queue to the persistent planner — the measurements
above bound the payoff at 13–27% on verify rows; (2) allow `MIXED` to coexist
with `TARGET_VERIFY`/`DRAFT_EXTEND` rather than being disjoint forward modes;
(3) lift the `enable_mixed_chunk = False` guard for speculative algorithms once
(1) and (2) hold. Only then does PD colocation reach functional parity with
PD-disaggregation-plus-spec-dec.

### 4-mode dispatch: the early estimate, and why it was replaced (2026-09-28)

Superseded — kept only because it explains the shape of the design that followed.

The first pass at sizing the 4-mode idea summed **separately timed** class
launches with **oracle** kernel choice per class over 40 batches from
`sets/qwen_small_h800.jsonl` (`measure_4mode_dispatch.py`, then
`measure_grouping.py`). It suggested 1.22-1.29x over a monolithic persistent
launch, and it was never a dispatcher: it bundled four launches and four
`plan()` calls into each row, chose kernels with hindsight, and modelled a
serialized split rather than any real execution. It also produced an alleged
~53 µs fixed launch overhead that no measurement supported; that figure and the
`DISPATCH_SPLIT_FLOOR_US` constant built on it were removed.

Two of its conclusions survived into the implementation and one was wrong:

* survived — the verify class wants a tile of 64 (its `packed_qo_len` is
  `(k+1) * gqa_ratio`), which motivated the third q-tile class below;
* survived — the draft class is not worth a separate low-SM launch (measured
  separately, next section but one);
* **wrong** — that the large-prefill class should go to FA3 unconditionally.
  That holds only when FA3 can fill the grid; see the measured boundary in
  "4-mode dispatch: separate kernels per tile class" below.

Everything quantitative about this design is in that later section, which times
real launches end to end.

### Implemented: a third q-tile class in the persistent kernel (2026-09-29)

`CTA_TILE_Q_SIZES` goes from `{128, 16}` to **`{128, 64, 16}`**, giving
speculative verify rows (`16 < packed_qo ≤ 64`) their own tile instead of
rounding them up to 128. flashinfer branch `tri_tile_`, commits `e5dbc8e`+.

**What changed.** `NUM_TASKS 2→3` and three-way routing (`scheduler.cuh`);
`HolisticPlanInfo<3>` and `params[3]` (`csrc/batch_attention.cu`, already a
loop); a third `KTraits`/runner and a `PersistentKernelTemplate3`
(`persistent.cuh`, `persistent_template.cuh`) where the two CTAs per SM walk
the queues `128→64→16` and `16→64→128` respectively — the existing
`cta_count % 2` forward/reverse pattern, kept binary to match 2-CTA occupancy;
a third tile in the jinja instantiation. The int workspace grows 8→24 MB for
the extra task arrays. **Split-KV and the LPT heap are untouched** and
validated below.

**Correctness** vs BatchPrefill, both schedules, max rel err ≤ 6.3e-3 (bf16
accumulation noise):

| composition | tiles exercised | rel err |
|---|---|---|
| 2 rows @64K (forces split-KV) | [16] | 5.0e-3 |
| 3 rows @64K + verify + chunk | [16, 64, 128] | 5.9e-3 |
| LPT skew, KV 512..64K over 12 rows | [16] | 6.1e-3 |
| LPT skew + verify(k=7) + 2 chunks | [16, 64, 128] | 6.1e-3 |
| captured shape: 21 draft + 7 verify + chunk | [16, 64, 128] | 6.3e-3 |

**Register pressure is the binding constraint, not scheduling.** Both builds
sit at the 255-register ceiling; the third mainloop body pushes state to local
memory:

| build | registers | stack | spill st | spill ld |
|---|---|---|---|---|
| 2-class `{128,16}` | 255 | 16 B | 16 B | 16 B |
| 3-class, `NUM_MMA_KV_2=4` | 255 | 32 B | 32 B | **72 B** |
| 3-class, **`NUM_MMA_KV_2=2`** | 255 | 16 B | 16 B | **28 B** |

Because this is a compile-time cost it is paid even when the middle queue is
empty — measured: forcing every verify row back to the 128 tile
(`FLASHINFER_TRI_TILE_MIN_KV=1e8`) was still slower than the 2-class build. So
routing-level gating cannot recover it; shrinking the middle tile's KV pipeline
(`NUM_MMA_KV_2` 4→2) can, and does.

**Result: it depends on how much work the verify class carries.**

| workload | 2-class | 3-class (tuned) | Δ |
|----------|--------:|----------------:|---:|
| Qwen2.5-7B, H800, gqa 7 (verify KV ~450/row, <10K/batch) | 40,077 µs | 40,194 µs | **−0.3%** |
| Qwen3-235B shape, gqa 16 (verify KV ~11K/row, ~600K/batch) | 198,767 µs | **193,750 µs** | **+2.6%** |

vs BatchPrefill the same runs give 1.302× → 1.298× (small) and 1.096× →
**1.129×** (large). Untuned (`NUM_MMA_KV_2=4`) the small model was −2.1%, so
the tuning is what buys parity there.

The crossover matches the earlier per-batch correlation: the third tile costs
~5% where the verify class holds <10K KV and breaks even above ~100K. The
large-model mixture sits far above it because gqa 16 with k=1 puts 58 verify
rows of ~11K KV in every batch. **So the third tile should ship enabled for
high-GQA/large-context deployments and disabled for small-model low-verify
ones** — `FLASHINFER_TRI_TILE_MIN_KV` (default 100000) makes that a per-batch
decision, though note it only reroutes work and cannot avoid the register cost.

**Measurement hazard worth recording:** flashinfer's JIT caches by *config
name* (dtypes, head dims, flags), not source content, so switching the
flashinfer branch does **not** invalidate a built module. An earlier 2-vs-3
comparison here was invalid for that reason — and a `git checkout` that aborted
on uncommitted changes left the wrong branch in place. Every number above was
taken after `rm -rf ~/.cache/flashinfer/90a/{cached_ops,generated}/*head_dim_qk_128*`
and confirming the loaded build via `len(plan_info)` (28 = 2 tasks, 38 = 3).

**Acceptance-rate modelling** (`tasks/fused_specdec_workload/acceptance.py`):
accepted length follows a truncated geometric law, `P(L=j)=(1-a)a^j` for `j<k`
and `a^k` for `j=k`, giving mean accepted `a(1-a^k)/(1-a)`. Profiles span
EAGLE-style heads (a=0.55–0.80, k=4 → 2.1–3.4 tokens/step) and deeper sparse
self-speculation (a=0.72, k=8). **The mixture is insensitive to `a`**: under the
SparseSpec unified schedule verify rows are `1/(k+1)` of the running set
whatever the acceptance rate, which sets only how fast contexts grow. So the
workload sets stand across acceptance scenarios, and `k` — not `a` — is the
parameter that moves the tile mixture.

### Spec-draft: fuse into tile 16, do not give it a low-SM dispatch (2026-09-29)

The open design question was whether the smallest class (draft rows: `qo_len=1`,
so `packed_qo` = the GQA ratio) deserves a dedicated low-SM kernel or should
just share the tile-16 queue with plain decode rows. Measured both premises
(`draft_dispatch_probe.py`, H800, gqa 7, draft rows @41K KV):

**(1) Draft work saturates HBM from ~8 rows — it is not a low-occupancy class.**

| draft rows | µs | GB moved | TB/s | % of 3.35 TB/s peak | µs/row |
|-----------:|---:|---------:|-----:|--------------------:|-------:|
| 1 | 56.1 | 0.08 | 1.50 | 45% | 56.1 |
| 8 | 242.5 | 0.67 | 2.77 | **83%** | 30.3 |
| 16 | 469.0 | 1.34 | 2.86 | 86% | 29.3 |
| 64 | 1,789 | 5.37 | 3.00 | 90% | 28.0 |
| 182 | 5,012 | 15.28 | 3.05 | **91%** | 27.5 |

Cost per row flattens at ~27.5–29 µs from 8 rows on, at 85–91% of peak
bandwidth. Realistic draft populations are 21 rows (small model) to 182
(GB200 mixture) — all far above saturation. **A low-SM partition would throttle
a class that already uses the whole machine**, cutting its bandwidth roughly in
proportion to the SMs withheld. The "smallest tile" is smallest in *q*, not in
work: each row still streams its entire KV.

**(2) Fusing into the tile-16 queue already beats running them apart.**

| composition | draft alone | decode alone | fused | fused/max | fused/sum |
|---|---:|---:|---:|---:|---:|
| 21 draft + 24 decode @41K | 621.7 | 786.6 | 1,354.9 | 1.72× | **0.96×** |
| 48 draft + 48 decode @11K | 1,421.8 | 397.2 | 1,694.7 | 1.19× | **0.93×** |
| 96 draft + 96 decode @11K | 2,756.6 | 757.9 | 3,375.7 | 1.22× | **0.96×** |

Fused is 4–7% *cheaper* than the sum of separate launches, so one queue already
wins. That `fused ≈ sum` rather than `≈ max` is not a scheduling failure: both
populations are bandwidth-bound, so they contend for HBM and there is no
concurrency to recover — splitting them across SM partitions would divide the
same bottleneck, not parallelise it.

**Decision: keep spec-draft fused into the tile-16 class.** A separate low-SM
dispatch is worse on both counts — it throttles a bandwidth-saturating class and
adds a launch to a fusion that is already 4–7% ahead. This also means the 4-way
*tile* design collapses to **three tiles** (`{128, 64, 16}`, as implemented):
draft and mixed-decode share tile 16 and differ only in scheduling class, not in
tile shape.

### 4-mode dispatch: separate kernels per tile class, grid-partitioned by a scheduler (2026-09-30)

Commits: sglang `ed8383a3c`+ (this section), flashinfer `tri_tile_` `645b77e`+.
Scripts, all under `tasks/fused_specdec_workload/`:

```bash
# calibrate the two mainloops' scaling curves
python calibrate_split.py sets/qwen_small_h800_largeprefill.jsonl 6
# does co-launch overlap at all, given the cooperative launch?
python overlap_probe.py
# draw the boundary (25 cells per head config)
python boundary_sweep.py --heads 28,4,128 --dec-kv 8000
python boundary_sweep.py --heads 8,1,128  --dec-kv 11000
# check the scheduler picks the measured winner, cell by cell
python validate_scheduler.py
# held-out: real captured batches, all strategies + oracle
python bench_4mode.py --set sets/qwen_large_h200_tp8.jsonl   --heads 8,1,128 --n 14 --oracle
python bench_4mode.py --set sets/qwen_large_gb200_tp16.jsonl --heads 4,1,128 --n 14 --oracle
python bench_4mode.py --set sets/qwen_small_h800_largeprefill.jsonl --heads 28,4,128 --n 14 --oracle
python graph_census.py        # modes and graph buckets over all 2,148 batches
```

#### What was built

Separate kernels per tile class, with the grid partitioned by the scheduler
rather than by green contexts. Two planner caps were added so a split is a
plan-time integer:

* `FLASHINFER_FA3_MAX_CTAS` in `PrefillSM90Plan` — the MinHeap balances the same
  work over fewer CTAs.
* `FLASHINFER_PERSISTENT_MAX_CLUSTERS` in the holistic planner — `kv_len_limit`
  is sized to the clusters actually owned.

**Corrected:** for `seq` this is fine — both kernels keep their full, fixed
grids (FA3 `{num_sm}`, persistent `num_clusters`), and the MinHeap merely assigns
different work to the same CTAs, so the launch config does not depend on the
prefill/decode ratio. For `conc` it is *not* fine: capping a grid per batch makes
the grid itself vary, and a different grid is a different launch config, so a
capture would be needed per split ratio — the same objection that ruled out green
contexts, just smaller. Quantising the split to a few ratios, or moving the split
into a task-partitioned cooperative launch with a fixed grid (see "Split-KV for
FA3" below), are the two ways out. `conc` is chosen on 48/2,148 batches and on
0/28 of the held-out batches, so nothing measured here depends on it.

Routing is per row, not per class, and the test is on **`qo_len`, not
`packed_qo_len`**: `PrefillSM90Plan` uses `cta_tile_q = 128`, does no KV
splitting, and does not fuse group heads ("the group_size in cost_function is
always 1"). A 4-token verify row therefore occupies 4/128 of an FA3 tile
whatever the GQA ratio. An earlier version tested KV length alone, sent 58
verify rows to FA3, and **lost 14x** — that error is what made the first
measurement of this design look hopeless.

Three execution modes, chosen per batch:

| mode | what runs | wins when |
|------|-----------|-----------|
| `fused` | one persistent launch over all classes (tiles 128/64/16, split-KV + LPT) | FA3 cannot fill the grid, or the decode population dominates |
| `seq` | FA3 on the eligible rows, then persistent on the rest, one stream | long prefix with a small decode population |
| `conc` | both co-launched at a scheduler-chosen split grid | very long prefix **and** a balanced decode population |

#### The two mainloops, measured

Per-CTA compute rate, fitted on compute-bound cells:

| mainloop | MAC/ns/CTA | relative |
|----------|-----------:|---------:|
| FA3 (wgmma, `cta_tile_q` 128) | 982 | 1.86x |
| persistent (FA2 lineage) | 595 | 1.00 |

So wgmma is worth 1.86x per CTA — the premise behind separate kernels holds.

**Corrected 2026-09-30: the reason FA3 loses at low `h_qo` is width alone, not
KV traffic.** An earlier version of this section blamed per-q-head KV reload.
That is measurably wrong. Running one `(1024, 32768)` row at matched FLOPs:

| config | KV footprint | FA3 |
|--------|-------------:|----:|
| (8,1,128), gqa 8 — 8 qo heads share 1 kv head | 8.0 MB | 271.2 us |
| (8,8,128), gqa 1 — every qo head its own kv head | 64.0 MB | 272.0 us |

**1.00x.** An 8x change in KV footprint costs nothing, so FA3 is compute-bound
here and the per-head reload is free (the shared KV fits L2). FA3's kernel does
carry the group size — `cutlass::FastDivmod(num_qo_heads / num_kv_heads)` in
`prefill_sm90.cuh` — it maps a qo head to its kv head; it just does not pack
several q heads into one tile, which costs tile occupancy, not bandwidth.

What actually hurts is that its work units are only (qo_head x qo_tile) with
**no KV splitting anywhere in the sm90 path**, so its parallel width is
`h_qo * ceil(qo_len/128)`. That is 224 at (28,4) heads — device-filling — but
only **64 at (8,1)**, the realistic TP8 shard of Qwen3-235B. The model predicts
the measured plateau exactly (FA3 flat at 273us from 64 CTAs up) and reproduces
its staircase: 32 and 48 CTAs both measure ~502us because both need
`ceil(64/c) = 2` rounds. A smooth `W/(c*R)` cannot express that, so the
scheduler simulates the planner's own MinHeap over its own
`cost_function = 2*qo_len + kv_len`.

Consequence: **at 8 q heads per rank the persistent kernel beats FA3 on pure
prefill** — 401us vs 507us at 256 KV tiles with no decode rows at all. "Large
tile" is not sufficient reason to leave the fused kernel; the question is
whether FA3 can use the device for it.

But that is a property of **flashinfer's sm90 planner, not of wgmma**, and it is
fixable. FA3 plateaus at 64 CTAs; if split-KV let it use all 132, the same work
would land at roughly `271 * 64/132 = 131us` against the persistent kernel's
215us — about **1.6x the other way**. That would move the boundary substantially
and is the highest-value next step, not a footnote. See "Split-KV for FA3"
below.

#### The boundary

50 cells, `qo_len` 1024, sweeping KV tiles attended x co-resident decode rows:

| head config | region | winner | margin |
|-------------|--------|--------|-------:|
| (28,4,128) | any tiles, no decode rows | `seq` (pure FA3) | up to 1.62x |
| (28,4,128) | rho < 1.5 (prefill dominates) | `seq` | up to 1.43x |
| (28,4,128) | rho >= 1.5 (decode dominates) | `fused` | up to 1.46x |
| (8,1,128) | everything except the two longest-prefix cells | `fused` | up to 1.61x |
| (8,1,128) | >= 256 KV tiles, rho >= 0.4 | `conc` | 1.18x |

`rho = t_pers(decode side) / t_fa3(prefill side)` at the full grid. Two
thresholds plus the width test classify **50/50 cells** correctly
(`validate_scheduler.py`).

Co-launch does overlap — `overlap_probe.py` measures
`(t_fa3_alone + t_pers_alone) / t_colaunch` peaking at **1.65x** at a 64/68
split — so the cooperative launch is not a barrier. But it must not be given a
large grid share: the persistent kernel needs all its blocks co-resident, so
over-provisioning makes the whole grid wait for FA3's CTAs to drain (2286us at
88/44 SMs vs 1470us when the persistent side asked for a third of that). An
earlier version of this sweep reported `conc` winning four cells it does not
win; the cause was a **unit error** — the persistent planner sets `num_sm *= 2`
for `head_dim < 256`, so its natural grid is 2 clusters per SM while FA3 runs 1
CTA per SM, and a "96/36 split" was really subscribing 114 of 132 SMs. The
scheduler now counts in SMs and converts on the way out.

#### Held out on the captured sets

14 batches per set, every launch and the fork/join events inside the timed
region. Both `seq` and `conc` are charged the gather of q and scatter of o for
the FA3 subset (measured 13.5-24.3us, 2.4-2.6% of split-mode cost); `fused`
needs no permutation, and the sequential baseline pays it on every batch.

| set | fused | seq (vLLM-style) | dual-stream | **ours** | oracle |
|-----|------:|-----------------:|------------:|---------:|-------:|
| qwen_large_h200_tp8 (8 q heads) | 1.03x | 1.00x | 1.01x | **1.054x** | 1.07x |
| qwen_large_gb200_tp16 (4 q heads) | 1.07x | 1.00x | 1.00x | **1.147x** | 1.13x |
| qwen_small_h800_largeprefill (28 q heads) | 0.693x | 1.00x | 1.017x | **1.000x** | 1.00x |
| **total** | **0.733x** | 1.00x | 1.015x | **1.015x** | — |

Every column is normalised to sequential, so read the fused column as "a fixed
fused kernel is 0.733x of sequential overall". Ours against a fixed fused kernel
is the reciprocal ratio of those two columns: **1.015 / 0.733 = 1.383x**. (An
earlier draft printed 0.76x here from a stale intermediate; the totals above are
recomputed from the saved per-batch JSON.)

Modes chosen: 4/7 fused on h200, 6/7 fused on gb200, 14/14 seq on the
long-prefix set. Within 5% of the per-batch oracle on 26/28 batches.

#### Against each baseline, plainly

1. **vs sequential (vLLM-style split), on realistically captured workload:**
   faster — 1.015x overall, 1.054x and 1.147x on the two large-model sets,
   parity on the small-model long-prefix set (where the scheduler correctly
   picks `seq`, i.e. it *becomes* the baseline). It wins on 23/28 batches. The
   gain is not from a better kernel; it is from not using one fixed strategy
   when the best one flips with `h_qo` and with the prefill/decode balance.
2. **vs dual-stream:** 1.000x — parity overall, 1.05x and 1.15x on the
   large-model sets, 0.98x on the long-prefix set. No advantage overall.
3. **vs a fixed fused kernel (the previous design):** 1.383x — the ratio of the
   `ours` and `fused` totals in the table above (1.015 / 0.733). The fused kernel
   alone is 0.69x of sequential on long-prefix batches, which is what sent this
   work back to separate kernels in the first place.
4. **Graph compatibility — the fused path is strictly better, measured.**
   Corrected: an earlier version of this section claimed parity, reasoning that a
   sequential pair captures in one graph. It does not, and the experiment settles
   it. At **fixed** total tokens (1,056) and batch size (33 rows), varying the
   prefill/decode ratio 1/32 → 8/25:

   * **fused** — one captured graph replays correctly at every ratio (checksums
     4.199e4 → 4.004e4, all correct). The ratio is invisible to the launch
     config: the persistent kernel's grid is `num_clusters` regardless of
     composition, and the whole p/d split lives in the plan buffers.
   * **seq** — FA3's `batch_size` goes 1 → 8 and its `qo_indptr` length 2 → 9
     with the ratio. Both are baked into the captured launch's params, so one
     graph cannot cover several ratios: sequential needs a capture per
     (n_prefill, n_decode) pair, which is the product blow-up predicted at the
     outset.
   * **conc** — capping a grid per batch makes the grid itself vary, so it needs
     a capture per split ratio. That is the objection that ruled out green
     contexts, in smaller form, and it is the remaining argument for folding FA3
     into the persistent cooperative launch as a fourth task: a task-partitioned
     launch keeps the grid fixed and moves the split into the plan buffers.

   So the ranking is fused (one graph per (tokens, bs) bucket) > sequential
   (prefill × decode buckets) > conc with per-batch grids. A zero-work FA3 launch
   is not a way to give `seq` the fused path's property: it costs 44 µs and fails
   TMA descriptor initialisation.

#### Honest summary

The design is right where it is right for a measurable reason, and it is not a
general win. Separate kernels with scheduler-chosen grids buy 1.015x over
sequential overall and up to 1.147x on the large-model shards, they beat a fixed
fused megakernel by 1.383x, they tie dual-stream, and they are *worse* on graph
count unless the gain is largely given back. The real content of the result is
the boundary itself: FA3's 1.86x wgmma advantage per CTA is cancelled by
per-head KV reload and a parallel width of `h_qo * ceil(qo_len/128)`, so on the
TP8/TP16 shards that actually serve Qwen3-235B it loses even on pure prefill.

### H200: JIT reduction and SM pre-assignment on an MTP schedule (2026-10-03)

Hardware: 1x NVIDIA H200 143GB, 132 SMs, CUDA 13.1, torch 2.9.1 (merlin trial
`vibecuda`, run `0f3a358f186d5fa4`). flashinfer `tri_tile_` + JIT reduction
(`e702da10`). Driven over merlin-cli's webshell exec channel; ssh is unavailable
on the i18n control plane.

```bash
# workload: MTP (k=1, a=0.85) over real bench_serving traces
python simulate_fused_batches.py traces/*.json --k 1 --draft-tokens 2 --alpha 0.85 \
    --max-running 64 --chunk-budget 8192 --out sets/mtp_h200.jsonl --max-steps 3000
# kernel A/B, one build, env-selected
FLASHINFER_SM_PARTITION=$P FLASHINFER_JIT_REDUCTION=$J python3 bench_jit_reduction.py --heads 28,4,128
```

**MTP collapses the tile taxonomy.** MTP ships one draft module, so a verify row
is `qo_len = k+1 = 2` and `packed_qo_len = 2*gqa_ratio`. Over 3,000 simulated
steps (76.6k verify+draft rows, 652 prefill rows):

| config | tile mix | tile-64 class |
|---|---|---|
| gqa 7 (Qwen2.5-7B TP1) | tile16 **99.2%**, tile128 0.8% | **unused** |
| gqa 8 (Qwen3-235B TP8) | tile16 99.2%, tile128 0.8% | unused |
| gqa 128 (MLA) | tile128 100% | unused |

The middle tile class added for EAGLE k=4-7 is dead under MTP on GQA models:
verify and draft both land in tile 16. The "4 modes" are 2 tile shapes. Context
is long (kv p50 34,452, 23 rows/step), so the tile-16 side is overwhelmingly
split-KV -- which is where reduction cost lives. Prefill is 44.4% of total work
but concentrated: **81.3% of steps contain no prefill at all**, 7% are
prefill-dominated.

**Correctness.** JIT vs barrier at a fixed seed: LSE bit-identical, output norms
15.488681793 vs 15.488539696 (9.2e-6 relative, far below one bf16 ulp of
7.8e-3) -- pure reduction-order rounding.

**Measured, 2x2 (us, lower is better):**

| bs/ctx/chunk | part0 jit1 | part0 jit0 | part1 jit1 | JIT gain | partition gain |
|---|---:|---:|---:|---:|---:|
| 32/8192/0 | 259.8 | 261.9 | 259.9 | 1.008x | 1.000x |
| 32/32768/0 | 997.6 | 999.7 | 999.1 | 1.002x | 0.998x |
| 64/8192/0 | 502.0 | 503.7 | 502.8 | 1.003x | 0.998x |
| 64/32768/0 | 1940.5 | 1942.8 | 1939.0 | 1.001x | 1.001x |
| 128/16384/0 | 1945.1 | 1945.2 | 1944.1 | 1.000x | 1.000x |
| 32/32768/1024 | 2328.9 | 2396.7 | 2957.3 | **1.029x** | **0.788x** |
| 64/32768/1024 | 3147.2 | 3157.8 | 4475.1 | 1.003x | **0.703x** |
| 128/16384/1024 | 2699.9 | 2824.3 | 3803.9 | **1.046x** | **0.710x** |
| **total** | **13821.0** | 14032.1 | 16881.2 | 1.015x | 0.819x |

**Reproduced on a second H200 pod** (the first trial was killed mid-session;
`3c57f94de4568319` replaced it, environment rebuilt in ~10 min once the webshell
transfer moved to 1 MB chunks -- 3 min for a 6.6 MB bundle vs 31 min at 8 KB):
part0/jit1 13877.4, part0/jit0 14122.1, part1/jit1 16899.0, part1/jit0 16868.5 us.
JIT gain 1.018x, partition 0.82x -- both within 0.7% of the first pod.

Two results, each only partly as expected:

1. **JIT reduction pays only on mixed batches** -- up to 4.6% when a prefill
   chunk is co-resident with decode rows, and exactly 1.000x when one tile class
   owns the batch. That is the predicted shape (a barrier only costs something
   when the work either side of it is heterogeneous) but the magnitude is small,
   because unlike FlashMLA/NSA (combine = 37%/51% of runtime in a *separate
   kernel*) our merge already ran inside the same launch. What JIT removes is
   the barrier wait, not a kernel launch or an HBM round trip.
2. **SM pre-assignment as currently implemented LOSES 21-30% on exactly the
   batches it targets**, and is a no-op (1.000x) everywhere else. Proportional
   work-share partitioning starves the tile-16 task: `kv_len_limit` is sized to
   the clusters a task owns, so giving the rare tile-128 row its own SMs forces
   the decode side into fewer, larger KV chunks and costs it parallelism. The
   split has to be a measured makespan balance applied conditionally, not a
   static proportional share applied always -- which is what the sglang-side
   gate (`mixed_dispatch.plan_dispatch`) decides, and on this MTP workload it
   picks the fused path for the 81.3% of steps that contain no prefill.

#### Swept: when does SM pre-assignment actually pay? (2026-10-03, H200)

`FLASHINFER_SM_SPLIT="a,b,c"` forces per-task cluster counts and
`FLASHINFER_SM_TASK_RATE="r0,r1,r2"` divides each task's work by its mainloop's
rate before shares are taken (flashinfer `c51aef742`). MTP shapes throughout
(prefill chunk = tile 128; verify `qo=2` and draft `qo=1` both = tile 16).

**Case 128 rows @ ctx 16384 + 1024-token chunk** -- partitioning wins here:

| split (prefill / · / decode) | us |
|---|---:|
| part-OFF (shared LPT heap over all SMs) | 2518.2 |
| 32,1,99 | 5292.8 |
| 48,1,83 | 3824.0 |
| 64,1,67 | 3062.2 |
| 72,1,59 | 2812.8 |
| **76,1,55** | **2323.2** |
| 80,1,51 | 2369.2 |
| 84,1,47 | 2331.0 |
| 92,1,39 | 2437.5 |
| auto, proportional-by-MAC | 3832.5 |
| auto, rate-weighted r=2.0 | **2306.1** |

The optimum is a broad plateau at 76-84 clusters (~1.08x over the shared heap),
and the MAC-proportional share picks ~48 -- **0.66x of the optimum**. Dividing
the small-tile task's work by r=2.0 before taking shares lands on the plateau
automatically, so the share formula, not the idea, was the bug.

**But one constant does not generalize**, and worse, one shape has no winning
split at all:

| case | part-OFF | rate r=2.0 | best forced split |
|---|---:|---:|---|
| 128 @16384 + 1024 | 2514.6 | **2446.1** (1.03x) | 76,1,55 |
| 64 @32768 + 1024 | 3043.5 | **2978.4** (1.02x) | - |
| 32 @32768 + 1024 | **2302.7** | 2937.2 (0.78x) | 112,1,19 -> 2950.0, still 0.78x |
| 128 @16384, no chunk | 1944.0 | 1941.6 | n/a (one live task) |
| 64 @32768, no chunk | 1942.3 | 1941.7 | n/a |

On `32 @32768 + 1024` the full sweep (16/32/48/64/80/96/112) never reaches the
shared-heap time and is still improving at 112 -- i.e. it converges toward
giving prefill the whole device and running the rest after it, which is just
sequential execution. Net over the three mixed shapes the rate-weighted
partition is **0.94x**: still losing.

**Mechanism.** The shared LPT heap is a strong baseline because it *backfills*:
small tile-16 work items drop into SMs that would otherwise idle during the
prefill tail. A static partition forbids that backfill. So pre-assignment pays
only when both classes are large enough to keep their own SM set busy for the
whole step -- true at 128 decode rows, false at 32 -- and a static share, even
rate-weighted, is the wrong control. The split has to be chosen per batch from
predicted makespans, with a fallback to the shared heap when the predicted gain
is small, which is exactly what `mixed_dispatch.plan_dispatch` decides
server-side. Under MTP that fallback is the common case: 81.3% of steps carry no
prefill at all.

#### The fused SM-pre-assigned co-launch, measured on the MTP schedule (2026-10-03, H200)

`bench_fused_dispatch.py` runs four strategies over the captured MTP steps that
are actually mixed (a large tile co-resident with small tiles -- 198 of 3,000
steps, 6.6%; most steps are pure-decode or pure-prefill):

| strategy | total us (20 mixed steps) | vs mono |
|---|---:|---:|
| mono -- one persistent launch over every row (today) | 94,195.9 | 1.000x |
| **seq -- FA3 on the large tiles, then persistent on the rest** | **66,073.3** | **1.426x** |
| fused -- the two co-launched, grids pre-assigned by the scheduler | 72,489.9 | 1.299x |

**Routing by tile class is worth 1.43x.** That is the real win and it vindicates
the premise the whole design started from: the large-tile class belongs on FA3's
wgmma mainloop, not on the FA2-lineage tile-128 path.

**But the SM pre-assignment loses to plain sequential by ~9%**, consistently,
on every mixed step. The reason is structural and specific to MTP: a mixed step
carries ONE prefill chunk against 1-5 small rows (verify `qo=2` + draft `qo=1`),
so the scheduler hands FA3 124 of 132 SMs and the persistent side 8. There is
almost nothing to overlap, and capping both grids only slows each side down --
sequential lets each kernel use all 132 SMs in turn. Co-launching pays only when
both classes can keep their own SM set busy for a whole step, which under MTP
they cannot.

Step 21 is the counter-example worth keeping: `mono` 312.4 us beats `seq` 673.3
and `fused` 554.9, because its "large" row is small enough that FA3 is pure
overhead. The routing gate must therefore test absolute work, not just tile
occupancy -- which is what `mixed_dispatch.plan_dispatch`'s "fold the class in
unless one SM's worth of it can stay busy" rule does.

So the deliverable from this line of work is the **dispatch**, not the fusion:
route large tiles to FA3 and run the two kernels back to back, gated per batch.
The SM-partitioned co-launch, the three-tile fused kernel, and the static
rate-weighted share are all measured and all lose to it on this workload.

FA3 split-KV was also measured here and did not help (1.267x vs 1.289x for the
plain co-launch): once the scheduler has already given FA3 124 SMs, width is not
the constraint and the merge is pure overhead. It was worth 1.75x standalone
only because unsplit FA3 was width-starved at 64 of 132 CTAs.

#### Server-side: the stage boundary dominates, not the kernel (2026-10-03, H200)

Kernel-level numbers do not carry to a server step unchanged. Speculative
decoding runs several stages per step -- draft, verify, and a prefill chunk when
one is admitted -- and each stage pays its own `plan()`, host->device metadata,
launch and sync. A 4-mode fused schedule (SparseSpec-style: verify of batch N-1
and draft of batch N in the same batch) collapses those into one.

Measured over 10 steps of the realistic MTP schedule, timing each stage the way a
server does it (plan + launch + sync, CPU time included):

| | us/step | of which plan() |
|---|---:|---:|
| staged -- draft, verify, prefill as separate stages | 19,388 | 7,597 (39%) |
| unified -- one plan, one launch over every row | 9,794 | 3,313 (34%) |
| | **1.980x** | |

Excluding plan entirely it is still **1.82x** (11,791 -> 6,480 us), because a
stage launched on its own under-fills the device: the draft stage is a handful of
one-token rows and the verify stage not much more. Batching them is what fills
the SMs, which is the argument SparseSpec's unified scheduler makes and the
reason the 4-mode schedule exists at all.

So the value decomposes as: **stage collapse ~1.98x** (available to any unified
schedule, fused kernel or not) **times the kernel's 1.148x** over sequential
within a step. The SM pre-assignment is the smaller term, and it only pays once
the stages are already collapsed -- which is the right order to build them in.

Caveat worth stating: this is the non-CUDA-graph path. A server that captures
decode steps in graphs amortises much of the `plan()` cost, so the 39% shrinks.
What does not shrink is the launch/sync boundary between stages and the
under-filled device, i.e. the 1.82x GPU-side component -- and a graph cannot span
a CPU-side metadata update between two stages, which is precisely what collapsing
them removes.

#### End-to-end sglang server verification (2026-10-04, H200)

Real server run, not a proxy: sglang 0.5.4.post2 (this branch) + torch 2.8.0 +
the flashinfer fork carrying the fused kernel, Qwen2.5-1.5B-Instruct,
`--attention-backend flashinfer`, `SGLANG_FLASHINFER_USE_PERSISTENT=1`,
`--chunked-prefill-size 512 --max-running-requests 64 --disable-cuda-graph
--disable-radix-cache`, `bench_serving --dataset-name random --random-input-len
2048 --random-output-len 128 --num-prompts 150 --request-rate 10`.

The composition logger recorded **1,344-1,408 mixed batches per run**, so the
batch shape the kernel targets does occur on the real server path.

| | JIT reduction ON | OFF |
|---|---:|---:|
| Output token throughput (tok/s) | 833.2 / 834.6 / 835.9 | 835.6 |
| Request throughput (req/s) | 8.67 | 8.69 |
| Median TTFT (ms) | 88.1 | 84.1 |
| Median ITL (ms) | 11.5 / 11.0 / 11.2 | 10.9 |

**Just-in-time reduction has no measurable end-to-end effect on this workload.**
Three ON runs span 833.2-835.9 tok/s (0.3%), and the OFF run at 835.6 sits inside
that band -- a 0.13% difference against a 0.3% noise floor. That is an honest
negative result and it is consistent with what the kernel work showed once the
counter-reset bug was fixed: the reduction phase is a small part of attention,
attention is a fraction of a 1.5B model's step, and this run has no speculative
decoding (Qwen2.5-1.5B has no MTP head), so there is no verify class and the
split-KV population the reduction serves is small.

The value of JIT reduction is therefore **not** a standalone speedup. It is what
makes the FA3 fusion possible at all: FA3 is warp-specialised, a CTA holding a
TMA producer can never join `grid.sync()`, and the persistent kernel used to end
in exactly that. Removing the barrier is a precondition, and it should be
reported as one rather than as a win.

What this run does NOT yet cover: the fused FA3+FA2 kernel is not wired into the
server's attention backend, so these numbers exercise the persistent kernel and
the JIT reduction, not the fused dispatch. The kernel-level 1.148x over
sequential and the 1.98x stage collapse are still measured off-server.

## Open TODOs

- [ ] **Remove the baseline's own plan() drain** (makes DMC strictly
  stronger *and* speeds the gate's BatchPrefill picks): sglang calls
  `wrapper_paged.begin_forward(qo_indptr, kv_indptr, ...)` with device
  tensors and no `max_token_per_sequence`/`max_sequence_kv`, so
  `prefill.py:1711-1719` does pageable `.to("cpu")` per mixed step. All the
  inputs exist host-side already (`seq_lens_cpu`, `extend_prefix_lens_cpu`).
  Scale of the opportunity, from the same captures: the persistent kernel ran
  on only 29,484/28 ≈ 1,053 steps of the LPT run (the gate declines most
  compositions), while BatchPrefill kernels account for ~544k launches — so
  ~95% of steps still go through the stock plan, and the run shows 22,141
  blocking D2H groups. The sync-free plan fix covers only the persistent
  slice; this TODO covers the rest.
- [ ] **The whole `fast_decode_plan` path is dead code** (corrected
  2026-09-14; supersedes the 2026-04-12 "it is shadowed" note). It is not
  shadowed — it does not exist: `flashinfer.fast_decode_plan` is absent from
  the pinned build, so `getattr(...)` at `flashinfer_backend.py:57` yields
  `None`, the `partial` binding at :928 never runs, and
  `wrapper_uses_fast_decode_plan` (:1581) is always False. Consequence per
  decode step: sglang allocates `global_override_indptr_cpu`, host-cumsums
  `seq_lens_cpu` into it (:1573-1577), never reads it (its only consumer is
  the dead branch at :1604), resets it to None (:1630) — and then stock
  `BatchDecodeWithPagedKVCacheWrapper.plan()` does the pageable D2H
  (`decode.py:941-942`) anyway. Note the D2H runs on graph-replayed decodes
  too, since replay still calls `indices_updater_decode.update()`. Fix by
  passing host `indptr`/`last_page_len`, as the BatchAttention path already
  does; the 2026-04-12 class-alias analysis (`begin_forward = plan` at
  `decode.py:1092`, `prefill.py:1915/2773`) stays true and would matter again
  if the symbol returned.
- [ ] **Re-test the persistent kernel for pure-decode batches.** It was used
  for decode (and decode CUDA graphs) until `b91b5afaf` (2026-03-25), removed
  because `BatchAttention.plan()` ran per decode step with no fast path. That
  reason is now obsolete — the sync-free host plan is 0.47 ms with a busy GPU.
  The commit's supporting number ("B200 WL1 18,847 tok/s vs ~13K before fix on
  H100") is **cross-machine and not a valid A/B**, and no sweep has ever
  compared persistent against `BatchDecode` (gate_sweep3 and sweep_fresh both
  use BatchPrefill as the baseline), so the removal currently rests on
  mechanism alone. Needs: kernel-level persistent-vs-BatchDecode sweep over
  decode-only compositions, then a matched serving A/B.
- [ ] **Confirm the copy-vs-sync split on plan() itself** (needs a GPU; none
  visible in the 2026-09-14 session). The positional result above already
  measures the mechanism (1st pageable D2H in a group: 5-8 ms median; 2nd:
  14 µs), so only the on-plan() reproduction is outstanding: time the three
  `.to("cpu", non_blocking=True)` lines with and without the following
  `torch.cuda.synchronize()` under ~40 ms of queued work.
- [x] **Rebuild the gate table from realized serving compositions.** Done
  2026-09-22 for gqa7: capture via `SGLANG_LOG_MIXED_COMPOSITION`, replay via
  `sweep_from_capture.py`, rebuild at k=7. Held-out 2.17× → 3.50× vs FA2 with
  worst-case LOO still 1.000. **Still open for gqa4** (Qwen3-8B), which needs
  its own capture run.
- [ ] **Overlap-aware chunk sizing (the one use that needs the kernel layer
  directly).** The EDF controller currently sizes chunks against a whole-step
  latency budget, so it *samples* the overlap band rather than aiming at it.
  `attn_layer_ms` already computes `T_p` and `T_d` separately: pick the chunk
  where `T_p ≈ T_d` against the live decode pool (subject to the SLO budget)
  and serving would sit inside the measured 1.2–1.26× band instead of
  drifting through it. This is the only consumer that would call the kernel
  layer on its own rather than through `predict()`.
- [ ] **Reduce β (non-overlappable fraction, 0.53)**: NCU a balanced
  composition to attribute the unrealized half of the overlap (reduction
  pass, queue-switch granularity, or L2 interference); each point of β is
  direct overlap efficiency. **One hypothesis is now closed: it is not the
  MMA pipe** — see "WGMMA tile-size mixing" below.
- [ ] **Gate table: sample long-prefix chunks over blended pools** (the one
  remaining unswept region) and refresh gqa4 with the blended sweep; also
  root-cause the storm-phase retraction cascade archived in
  `h800_mw_agent_plus_shortchat/run1_regressed/`.
- [ ] **Remaining mixed-workload suite scenarios**: `rag_plus_agent`,
  `long_cot`, `agent_plus_chat` (`run_mixed_workloads.py`) to complete the
  workload matrix.
- [ ] **GPU-side LPT plan kernel** [DEPRIORITIZED 2026-08-06]: the sync-free
  host plan removed this from the critical path (plan is hidden by
  overlap-scheduler run-ahead); revisit only if nsys shows plan CPU cost
  bounding step rate.
- [x] **EDF end-to-end A/B with the analytic model**: done 2026-08-11
  (`multiturn_slo_clean`, both tenants rate-limited, node throughput matched
  to four digits across arms): kernel alone codex pooled P99 ITL −28.3%;
  adding the analytic-model controller pins turns pooled P99 at 100.24 ms
  against the 100 ms budget and cuts request-max P99 −60% on both tenants.
  See `tasks/realistic_workload_analysis.md`.
- [x] Realistic workload analysis — see `tasks/realistic_workload_analysis.md`

---

## Historical results (H100 era) — archived

Moved to [`archive_h100_era.md`](archive_h100_era.md) on 2026-09-30:
H100/Qwen3-8B kernel and server results, the closed investigations, and the
v3 scheduler / split-KV / mixed-CG implementation notes. They predate the
gate stack and the H800 phase.

## GPU Keepalive
```bash
bash ../VibeCUDA/scripts/gpu_keepalive.sh start   # hold GPU during setup
bash ../VibeCUDA/scripts/gpu_keepalive.sh stop    # STOP before benchmarks!
bash ../VibeCUDA/scripts/gpu_keepalive.sh start   # restart after benchmarks
```
**IMPORTANT:** gpu_keepalive corrupts benchmark results. Cooperative launch (persistent kernel) is especially sensitive to SM contention.

## Profiling Configurations (legacy)
| Config | nsys trace | sqlite |
|--------|------------|--------|
| Default | `default_bt.nsys-rep` | `default_kernel.sqlite` |
| Persistent | `persistent_bt.nsys-rep` | `persistent_kernel.sqlite` |
| POD | `pod_bt.nsys-rep` | `pod_kernel.sqlite` |

## H200 server-side kernel A/B (2026-10-04)

Hardware: 1x NVIDIA H200 143GB, 132 SMs, sm_90a (merlin trial `vibecuda`, job
`e59d6e2974d3ed90`, queue `h100-sxm-80gb.vmss-eastus2-vlmh200.agent.merge`).
sglang `c0dfe0419` (branch tree on PYTHONPATH), flashinfer `tri_tile_` @
`94909aa` in-tree. Qwen2.5-7B-Instruct, TP1, 28 qo / 4 kv heads.

```bash
# server
SGLANG_FLASHINFER_USE_PERSISTENT=1 SGLANG_MIXED_ATTENTION_KERNEL=$K \
python -m sglang.launch_server --model-path qwen7b --port 31000 \
  --mem-fraction-static 0.80 --attention-backend flashinfer \
  --enable-mixed-chunk --chunked-prefill-size 512 \
  --max-running-requests 64 --disable-radix-cache --log-level info
# load
python -m sglang.bench_serving --backend sglang --port 31000 \
  --dataset-name random-ids --random-input-len 4096 --random-output-len 512 \
  --num-prompts 300 --request-rate inf --max-concurrency 64
```

`K` in {`persistent`, `batch_prefill`}. Both arms logged an **identical forward
composition** -- 3016 mixed, 969 `decode_cuda_graph`, 15 `pure_prefill` -- so
only the mixed-batch attention kernel differs.

| | persistent overlap | BatchPrefill | delta |
|---|---|---|---|
| Total token throughput | **23,594 tok/s** | 22,576 | **+4.5%** |
| Output token throughput | 2,660 | 2,545 | +4.5% |
| Request throughput | 10.10 req/s | 9.67 | +4.4% |
| Mean TTFT | 1,327 ms | 1,373 | -3.4% |
| P99 TTFT | 4,846 ms | 5,017 | -3.4% |
| Mean ITL | 17.63 ms | 18.54 | -4.9% |
| P99 ITL | **21.20 ms** | 22.84 | **-7.2%** |

### Three method errors worth not repeating

Forcing `SGLANG_MIXED_ATTENTION_KERNEL` is not a convenience, it is what makes
this an A/B at all. Three earlier attempts produced arms that agreed to 0.1% on
both throughput and P99 TTFT, which is not a null result about the kernels:

1. **Unsaturated operating point.** 8 req/s at concurrency 32 leaves the GPU idle
   most of the wall clock, so no kernel difference can surface. `--request-rate
   inf` with concurrency 64 was needed before anything was being measured.
2. **`--log-level warning` discarded the diagnostic.** `_count_forward` logs the
   path mix every 2000 forwards at INFO, which is exactly the instrument that
   catches error 3; running at warning threw it away.
3. **The gate routed both arms to the same kernel.** With
   `SGLANG_FLASHINFER_USE_PERSISTENT=1` and the gate active, the mix was
   `mixed_batch_prefill=3811 (63.5%)` against `mixed_persistent=83 (1.4%)` --
   both arms were running BatchPrefill. `select_mixed_attention_kernel`'s
   out-of-support guard defaults there, and a synthetic `random-ids` workload is
   outside the fitted 1,316-point region. Moving to the KV-concentration regime
   (32K x 8) only raised it to 1.85% of forwards, so 32K is still short of the
   64K the gate wants.

Each was catchable up front with one question: is the machine busy, is the code
path executing, am I varying what I think I am varying. Always read the forward
mix before believing a server A/B.

### Scope

This verifies the **persistent overlap kernel**, which is what the analytic cost
model and the PD-colocation story in `tasks/pd_colocation_vs_disagg.md` rest on.
It does **not** verify the fused FA3+FA2 SM-pre-assignment kernel: `fused_run`
has no call site in `flashinfer_backend.py`, so that kernel never executes in
the server. Wiring it needs the four constraints the standalone harness
established -- FA3's plan capped to `sm_split`, the FA2 side's KV indices
rebased past the FA3 rows' pages, FA3 given vector-sparse offsets rather than
page ids, and FA3 confined to a contiguous batch prefix.

### Fused FA3+FA2 on the serving path (2026-10-04)

Same H200 trial. sglang `859ff75b8`+ (fused call site), flashinfer `tri_tile_`
in-tree. Qwen2.5-7B TP1, 150 prompts, 4096/512, `--request-rate inf`,
concurrency 64, `SGLANG_MIXED_ATTENTION_KERNEL=persistent`,
`SGLANG_FORWARD_MIX_EVERY=200`.

```bash
/dev/shm/w/abf2.sh {1,0}   # SGLANG_FUSED_FA3_SM_SPLIT
/dev/shm/w/runb.sh {fusedB,ctrl}
```

| | fused path ON | control (OFF) | delta |
|---|---|---|---|
| Total token throughput | 20,260 tok/s | **22,203** | **-8.7%** |
| Output token throughput | 2,306 | 2,527 | -8.7% |
| Mean TTFT | 2,066 ms | 1,896 | +9.0% |
| P99 TTFT | 5,426 ms | 5,028 | +7.9% |
| Mean ITL | 17.30 ms | 15.52 | +11.5% |
| P99 ITL | 25.67 ms | **21.55** | **+19.1%** |

Forward mix proves what served each arm, and the arms are matched:
`mixed_persistent=796` and `pure_prefill=8` in both, with
`mixed_fused_fa3=39 (3.2%)` present only in the fused arm.

**The integration works and the kernel executes**: the extension builds through
the server's JIT, planning succeeds, and there were **0 launch failures** across
the run -- so every batch the contract accepted was served by the fused kernel
without falling back.

**This workload cannot isolate the fused kernel.** It claimed only 3.2% of
forwards, because at `--chunked-prefill-size 512` a prefill row needs >=4096
accumulated KV to clear `kv_tiles >= 32`, so a request's early chunks never
qualify. 66.3% of forwards went to the persistent kernel in both arms.

**The -8.7% is mostly planning cost, not kernel time.** Decomposing: the
end-to-end time ratio is 1.096, and if only 3.2% of forwards changed cost then
those forwards were **3.9x** more expensive. That is far worse than the
kernel-level deficit (1.345x behind a sequential co-launch), which points at
`_plan_fused_split`: it runs two extra full plans **every step** -- an FA3 plan
and a persistent plan, each with its own ~10 MB H2D copy -- while the kernel it
is planning for serves 3.2% of steps. Planning per step to use a kernel on 3% of
steps is the wrong shape; the plans should be built only when the contract has
already accepted the batch, and ideally cached across steps whose layout is
unchanged.

A prediction made before the run, recorded for calibration: 15-25% end-to-end
loss. Actual -8.7%, so the prediction was too pessimistic -- but for the wrong
reason. It assumed the fused kernel would serve all mixed forwards; it served
3.2%. Had it served all of them the loss would have been much larger than 25%.

To isolate the kernel server-side one would need prefill chunks clearing the gate
from the first chunk: a large `--chunked-prefill-size` over long prefixes. That
is the shape `tests/bench_mtp_fused.py` already drives directly and more
cleanly, which is why the kernel-level numbers remain the stronger evidence.
