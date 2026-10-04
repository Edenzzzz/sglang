# Hopper 4-mode fused attention: FA3 + FA2 SM pre-assignment

**Lineage.** Split out of `debug_kernel_server_perf.md` on 2026-10-05, which had
reached 3428 lines spanning two unrelated efforts. That document keeps the earlier
work -- the sync-free plan, the measured mixed-kernel gate, the workload
characterisation and the mixed-tile 4-mode *dispatch*. This one continues from it
and covers only the **fused FA3 + FA2 SM pre-assignment kernel**: one launch, two
roles split by `blockIdx.x < sm_split`, just-in-time reduction in place of a grid
sync.

| doc | covers | status |
|---|---|---|
| `debug_kernel_server_perf.md` | Parts 1-4: sync-free plan, kernel gate, workload stats, mixed-tile dispatch | complete, archived |
| **`debug_kernel_server_perf_4_mode.md`** (this) | fused FA3+FA2 kernel: correctness, server integration, split/cluster tuning, backfill, multicast | active |
| `pd_colocation_vs_disagg.md` | global PD colocation vs disaggregation, the four routing/caching/EDF/network points | written, experiments pending |
| `server_perf_model.md` | MLP latency prediction for prefill-decode colocation | prior work |
| `realistic_workload_analysis.md` | dataset length stats, arrival rates, per-turn growth | prior work |

Read the sections below in order; each records what was measured, and corrections
to earlier claims are kept in place rather than edited away, because several
conclusions here were reversed by later measurement and the reversals are the
useful part.

---
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

#### The -8.7% is not planning overhead (correction)

| | tok/s | vs control |
|---|---|---|
| control, fused path off | 22,203 | - |
| fused, original wiring | 20,260 | -8.7% |
| fused, planning deferred + indices reused + plan cached | 20,314 | **-8.5%** |

The planning optimization recovered 0.3%, i.e. nothing. **The decomposition
above it was wrong**: inferring "3.9x per affected forward, therefore planning"
from an end-to-end ratio and a 3.2% share was arithmetic applied to an
unverified assumption, namely that the only thing enabling the path changes is
the cost of the forwards it serves. Removing the planning cost falsified it.

Remaining candidates, in order of suspicion:

1. **768 MB of extra workspace** (256 MB FA3 + 512 MB persistent) allocated
   lazily on first fused plan, *after* the KV pool was sized at startup from
   `--mem-fraction-static`. It therefore competes with activations rather than
   coming out of a budget.
2. **`FLASHINFER_FA3_MAX_CTAS` leaks.** It is set in `os.environ` and never
   cleared, so after the first fused plan it caps CTAs for *every* later
   flashinfer fa3 plan in the process, not only the fused one.
3. The kernel itself, on the 39 forwards it serves.

**The decisive test, not yet run**: enable the path but force the planner to
return None. The workspaces still allocate and the env var is still set, but the
kernel never runs. If throughput stays near 20.3k the cost is environmental
(1 or 2) and the kernel is exonerated; if it returns to 22.2k the cost is the
kernel (3). That single run separates all three and should precede any further
tuning of this path.

#### Retraction: the fused kernel never ran on the serving path

Every "fused server-side" figure above (-8.7%, and the 20,314 / 20,967 / 20,880
tok/s follow-ups) was **the persistent kernel plus integration overhead**. The
fused kernel did not execute once.

The certification was broken, not just the result. `_count_forward` fired at
*plan* time, one step upstream of the launch, so 39 plans that each failed while
building the plan dict were reported as `mixed_fused_fa3=39` and read as 39
successful launches. Warning deduplication (one log per distinct message) hid the
39 failures behind a single line. Moving the counter to the launch site exposed
the truth: **declines 0, launch failures 39, `mixed_fused_fa3` absent.**

Root cause of the failures: `get_kv_buffer` returns a `(k, v)` tuple for this
pool layout rather than a fused `[N,2,...]` tensor. The same wrong assumption
appeared twice -- in `_fa3_sparse_indices` and again in the launch path -- and
fixing only the first left the second raising `tuple indices must be integers or
slices` on every launch.

With both fixed, the first genuine launch produced:

```
CUDA error: misaligned address
Fatal Python error: Aborted
```

and killed the server. A misaligned-address fault corrupts the CUDA context, so
no Python-level guard can contain it; the path must stay disabled
(`SGLANG_FUSED_FA3_SM_SPLIT` defaults to 0) until the alignment is found.

Leading hypothesis, untested: the rebased KV index pointer. `pers.plan` is given
`kv_indices[n_pages:]`, and slicing an int32 tensor at an arbitrary `n_pages`
puts the base at `base + 4*n_pages` bytes, which is only 16-byte aligned when
`n_pages % 4 == 0`. The standalone harness never caught this because its
`kvF = 8704` happens to be a multiple of 4. The launch separately passes the full
array plus `pers_kv_page_offset`, so the C++ side offsets again -- the two paths
need reconciling, and the plan should take the full array with an offset rather
than a slice.

What stands, because none of it depends on the server: the kernel-level H200
numbers, correctness (72/72 rows over 6 matrices against fp32), the backfill A/B,
the persistent-overlap server A/B (counted at the right site), and the ablation
showing the kernel contributes -0.05% server-side while the -8.5% was leaked
`FLASHINFER_FA3_MAX_CTAS` and 768 MB of late-allocated workspace.

#### Status: fused kernel launches in the server and faults in-kernel

After fixing the `(k, v)` tuple assumption in both the sparse-index helper and
the launch path, and removing the misaligned index slice (the persistent side's
`kv_indptr` is now global and the index array is passed unsliced, so
`pers_kv_page_offset` is 0):

```
plan declines: 0    python-level launch failures: 0    misaligned/abort: 1
```

So the plan succeeds, the launch is attempted with no Python error, and the
kernel faults **inside** with `CUDA error: misaligned address`, aborting the
server before the first forward-mix log. That is the first genuine launch of this
kernel in the server, and it does not survive it.

The slice was a real hazard -- measured, `int32[n]` is 16-byte aligned only for
`n % 4 == 0`, and the standalone harness's page count of 8704 is divisible by 4
so it never hit it -- but fixing it did not fix the fault, so it was not the
cause. What distinguishes the two environments is now the open question:

- the standalone harness builds `q`, `o` and the page array as fresh contiguous
  tensors; in the server `q` is a view of the model's fused QKV output and `o` is
  a slice of a reused buffer, so neither base pointer is guaranteed to share the
  harness's alignment
- the harness plans both sides over one synthetic batch, while the server's
  persistent plan is built from a global `kv_indptr` over the real paged pool

Next step is compute-sanitizer against the server process to get the faulting
instruction and operand, rather than more hypotheses -- this bug has already
survived two plausible ones.

`SGLANG_FUSED_FA3_SM_SPLIT` stays 0 by default. A misaligned-address fault
corrupts the CUDA context, so this path cannot be guarded at the Python level and
must not be enabled until the fault is understood.

#### Root cause: the page-index array, found by dumping the launch arguments

Rather than a third hypothesis, the launch arguments were dumped to disk
immediately before the faulting launch (the dump survives the abort) and read
back. Instrumentation added at the launch site in
`python/sglang/srt/layers/attention/flashinfer_backend.py`, enabled with
`SGLANG_FUSED_DUMP=<path>`.

The faulting step:

```
qo      = [424, 87, 1]        prefix = [3577, 0, 13]   ->  kv = [4001, 87, 14]
gqa     = 7                   packed_qo = [2968, 609, 7]
n_rows  = 1 (FA3)             sm_fa3 = 48
q, o, kv_indices, fa3_sparse base pointers: all 16-byte aligned
kv_indices: 3846 entries, but the batch needs 4102
            256 of those entries outside the valid page range [0, 1816955]
            first invalid at index 3590, value 1085595437
```

So no pointer was misaligned at all. The kernel was gathering **garbage page
ids** — values up to 1.1e9 against a 1.8e6 page pool — from an array that was
both too short and partly uninitialised, and faulted on the resulting address.

The cause was mine, and it was the planning optimisation from the previous
session: `_plan_fused_split_inner` reused `wrappers[0]._kv_indices` on the
stated assumption that "the mixed wrapper was just planned over the whole batch,
so it already holds exactly this array". It does not. That assumption replaced a
correct per-request build to save planning time, and the saving was ~0.3% of
throughput.

Two separate reasons this was not caught earlier, both worth recording:

- the tensor-provenance hypothesis was tested and is **false**. A new test
  (`tests/align_fused.py`) reproduces each server-side provenance in the
  standalone harness — `q` as a non-contiguous view of a fused QKV buffer, `o`
  as a slice of a reused buffer, the index array built by concatenating
  `req_to_token` rows — and all of them pass. Provenance was never the issue.
- nothing checked the array's length against the plan's own page total, so a
  wrong buffer reached the GPU as a fault instead of a declined plan.

Fix: build the array, but vectorised on the GPU (one `repeat_interleave` for the
row of each output slot, one subtract for its offset, one gather from
`req_to_token`) so correctness does not cost the planning time back, plus a hard
length check that declines the plan loudly.

#### Second bug, found while reading the dump: the tile-128 task was never run

The persistent plan is a `HolisticPlanInfo<3>` over three q-tile classes
(128 / 64 / 16). `TwoStageHolisticPlan` (`scheduler.cuh:1161`) sends every row
with `packed_qo_len > 64` to task 0, and also folds the whole tile-64 mid class
into task 0 whenever that class carries less than `FLASHINFER_TRI_TILE_MIN_KV`
(default 100000) total KV.

`tests/fused_fa3_persistent_ext.cu` built params for tasks 1 and 2 only, with
the comment "task 0 went to FA3". That is wrong: the persistent plan is built
over the FA2 sub-batch *after* the FA3 rows have been removed, so its task 0 is
ordinary FA2 work and nothing else executed it.

In the dumped batch row 1 (`qo=87`, `packed_qo=609`) lands in task 0 and was
silently skipped. Every previous harness used MTP decode rows for the FA2 side,
whose `packed_qo = 2*gqa = 14` lands in task 2, so task 0 was always empty and
the omission could not show.

Fix: a third runner `R128 = PKT<128, 2>` threaded through `fused_drain_queue`,
`FusedFA3PersistentKernel` and `FusedFA3PersistentDispatched`. Shared memory is
unaffected — tile 128 needs 49152 bytes of `SharedStorage` against tile 16's
69632, so `Runner3` still sets the bound.

Regression test: `tests/test_fused_server_batch.py` replays the dumped batch
shape exactly, so it covers FA3, task 0 and task 2 in one launch and asserts no
output row is left unwritten.

#### The fault is at launch, not in the kernel -- and the two environments use different toolchains

Re-running the server under `CUDA_LAUNCH_BLOCKING=1` moved the error from the
next sync point to the call that caused it:

```
CUDA Error: misaligned address  persistent_fa3_fused.cuh: line 420
  at function cudaLaunchKernel(kernel, grid_dims, block_dims, args, smem_size, stream)
fused FA3+FA2 launch failed, using persistent: fused kernel failed: misaligned address
...
flashinfer_backend.py line 1339, in persistent_attn_forward
    o, lse = batch_attn_wrapper.run(...)
RuntimeError: Failed to run persistent paged attention, error: misaligned address
```

So it is **`cudaLaunchKernel` itself** that rejects the launch -- the kernel never
starts -- and the Python-level guard then falls back to `batch_attn_wrapper.run`,
which fails too because a failed launch has already poisoned the CUDA context.
That second failure is what killed the server and what made earlier runs look
like a fault in flashinfer's ordinary persistent kernel. It is a consequence, not
the cause.

A launch-time misaligned address is the failure mode this header already
documents: FA3's mainloop params carry TMA descriptors that must sit 64-byte
aligned in the kernel parameter buffer.

Matched controls, concurrent traffic, `CUDA_LAUNCH_BLOCKING=1`, 16 requests of
~4000 tokens each at `chunked-prefill-size 512`:

| arm | requests | misaligned | server at end |
|---|---|---|---|
| `SGLANG_FUSED_FA3_SM_SPLIT=0` | 16/16 ok | 0 | alive |
| `SGLANG_FUSED_FA3_SM_SPLIT=1` | 0/16 | yes, at `cudaLaunchKernel` | dead |

Two earlier attempts at this control were invalid and are recorded so the mistake
is not repeated: the first sent its requests **serially**, so almost no mixed
batches formed and the fused path was never exercised; the second was not a
control at all, because `serve.sh` contained a hard `export
SGLANG_FUSED_FA3_SM_SPLIT=1` that overrode the per-run value. `serve.sh` now uses
`${SGLANG_FUSED_FA3_SM_SPLIT:-1}` so the flag is actually togglable.

**The toolchain difference is real but is not the cause.** There are two torch
extension build directories on the pod:

```
~/.cache/torch_extensions/py312_cu131/fused_fa3   <- system python3, torch 2.9.1, CUDA 13.1
~/.cache/torch_extensions/py312_cu128/fused_fa3   <- /dev/shm/w/venv, torch 2.8.0, CUDA 12.8
```

`serve.sh` activates the venv, so the server compiles and runs under CUDA 12.8
while the kernel tests had all been run under 13.1. That gap was worth closing
and is closed: `tests/test_fused_server_batch.py` run with
`/dev/shm/w/venv/bin/python` **passes** at splits 48/32/16, 0 unwritten rows,
every row within 0.0035 of the fp32 reference. So the launch is fine on the
server's own toolchain and the toolchain was not the cause.

#### Root cause: the KV cache does not have a fixed rank

The extension read `k_cache.stride(1)` as the token stride and
`k_cache.stride(2)` as the head stride. That is correct for exactly one of the
two layouts it is given:

| caller | KV tensor | strides | k_stride_n | k_stride_h |
|---|---|---|---|---|
| standalone harness | rank 4 `[pages, page_size, H, D]` | `[1024, 512, 128, 1]` | 512 | 128 (correct) |
| sglang paged pool | rank 3 `[tokens, H, D]` | `[512, 128, 1]` | 128 | **1 (wrong)** |

The harness slices a `[pages, 2, page_size, H, D]` buffer, so it is rank 4;
sglang's pool hands over a rank-3 tensor at `page_size 1`. With
`k_stride_h = 1` all four KV heads alias one head's memory, and
`get_gmem_layout(0, num_kv_heads, HEAD_DIM, 128, 1)` builds a degenerate layout
whose tensor map `cudaLaunchKernel` rejects as misaligned.

Note the page step happened to survive both layouts by accident
(`ratio * k_stride_n` is 2*512 = 1024 for rank 4 and 4*128 = 512 for rank 3,
both the correct page step), which is why only the head stride was wrong and why
the symptom was a launch rejection rather than wrong numbers.

This is a difference no harness built on the harness's own KV layout could ever
catch, which is the same structural blind spot as the tile-128 task: both bugs
exist precisely where the test fixture differs from sglang.

Fix: `kv_strides()` in the extension derives `(page, n, h)` from the tensor rank
and rejects any other rank outright, and `_fa3_sparse_indices` picks the matching
vector-sparse ratio (`stride(0)//stride(1)` for rank 4, `1` for rank 3) so the
two stay consistent.

The `kv_indices` fix is independently confirmed on the serving path: a second dump
taken after the fix shows `kv_indices len = 8192, pages needed = 8192, 0 invalid
entries` (previously 3846 vs 4102 with 256 garbage entries), and every base
pointer 16-byte aligned. That bug is fixed; the launch rejection is a separate
one.

#### The fused kernel now runs on the serving path

With the rank-derived KV strides (flashinfer `4d59914`) and the matching
vector-sparse ratio (sglang `ca09078ab`), the first clean serving run:

```
16/16 concurrent ~4000-token requests served (max_new_tokens 48)
misaligned=0   fatal=0   launch_failed=0   server alive at the end
forward mix (n=600): mixed_fused_fa3=448 (74.7%)  mixed_persistent=118 (19.7%)
                     decode_cuda_graph=25 (4.2%)  pure_prefill=9 (1.5%)
```

Command: `bash /tmp/run5.sh 1 fix2` on the H200 pod (`serve.sh` with
`SGLANG_FUSED_FA3_SM_SPLIT=1`, `SGLANG_MIXED_ATTENTION_KERNEL=persistent`,
`--enable-mixed-chunk --chunked-prefill-size 512 --max-running-requests 64
--disable-radix-cache`, qwen7b).

The `mixed_fused_fa3` count is taken at the launch site, not at plan time, so it
counts launches rather than accepted plans -- the instrument whose earlier
mis-placement produced the retracted numbers.

Two operational notes from this session worth keeping:

- `CUDA_LAUNCH_BLOCKING=1` is usable for attribution but **not** for a serving
  run: it slowed startup enough that the detokenizer heartbeat timed out and the
  server wedged before any traffic was processed. Use it only to locate a fault.
- `pkill -f` patterns must bracket a character (`sglan[g]::`) or the webshell
  kills itself, because the pattern string appears in its own command line. This
  cost several cycles earlier and recurred once this session.

#### A fifth invalid A/B, caught by the composition line

The first throughput A/B on the now-working path produced a clean-looking null
result:

| arm | total tok/s | mean ITL | p99 ITL | `mixed_fused_fa3` |
|---|---|---|---|---|
| `SM_SPLIT=1` | 18182.03 | 11.31 | 29.99 | **absent** |
| `SM_SPLIT=0` | 18203.29 | 11.05 | 28.88 | absent |

It is invalid, and the reason is in the composition line rather than the numbers:
`mixed_fused_fa3` never appears in either arm, so **both arms ran the same
kernel** and the 0.1% difference is run-to-run noise.

The cause is the workload, and the gate was behaving correctly. `bench.sh` used
`--random-input-len 2048`, while the FA3 gate needs `packed_qo >= 48` **and**
`kv_tiles >= 32`, i.e. at least 4096 accumulated KV. At `chunked-prefill-size 512`
a 2048-token prompt never accumulates more than 2048, so `kv_tiles <= 16` and no
row ever qualifies. The hand-driven traffic that first exercised the kernel used
~4000-word prompts, which is why it engaged there and not here.

So the fused path only engages on long-context prefill, which is worth stating as
a property of the gate rather than discovering again: the A/B must use
`--random-input-len` of at least ~4096, and **every** fused measurement must carry
its `mixed_fused_fa3` share or it is not evidence about this kernel.

That makes five invalid A/Bs on this path, all from the same root cause -- the arms
were not actually running different kernels. The composition counter is the only
thing that has reliably caught it, which is why it is checked before the numbers.

#### First valid server-side measurement of the fused kernel: a 8.1% regression

Same script with `--random-input-len 8192 --random-output-len 192
--num-prompts 300 --request-rate 12 --max-concurrency 32`, which clears the FA3
gate. The compositions now differ, so the two arms really did run different
kernels:

| metric | fused (`SM_SPLIT=1`) | base (`SM_SPLIT=0`) | delta |
|---|---|---|---|
| total token throughput (tok/s) | 23250.44 | **25291.44** | **-8.1%** |
| benchmark duration (s) | 54.83 | 50.40 | +8.8% |
| output token throughput (tok/s) | 556.62 | 605.48 | -8.1% |
| mean TTFT (ms) | 3443.97 | 3150.61 | +9.3% |
| p99 TTFT (ms) | 4679.48 | 4221.87 | +10.8% |
| mean ITL (ms) | 20.18 | 18.54 | +8.8% |
| p99 ITL (ms) | 31.87 | 31.62 | +0.8% |
| `mixed_fused_fa3` share | **29540 / 32400 (91.2%)** | absent | — |
| `mixed_persistent` share | 2796 (8.6%) | 2750 (98.2%) | — |
| misaligned / fatal / launch failures | 0 / 0 / 0 | 0 / 0 / 0 | — |

300/300 requests succeeded in both arms and both servers were alive at the end.

Commits: sglang `ca09078ab`, flashinfer `4d59914`. Command:
`bash /tmp/ab.sh` on the H200 pod (both arms from one `serve.sh`, toggled by
`SGLANG_FUSED_FA3_SM_SPLIT`), model qwen7b, `--chunked-prefill-size 512
--max-running-requests 64 --disable-radix-cache`,
`SGLANG_MIXED_ATTENTION_KERNEL=persistent`, `FLASHINFER_FUSED_BACKFILL=0`,
CUDA 12.8 / torch 2.8.0 (the venv the server uses).

**Reading.** The kernel is now correct and genuinely running on the serving path
at 91.2% of forwards, and it is **slower than the persistent kernel it replaces
by 8.1% of end-to-end throughput.** That is one run per arm, but the gap is two
orders of magnitude larger than the 0.1% spread between the two arms of the
earlier A/B where both ran the same kernel, so it is a real effect rather than
noise.

The direction agrees with the kernel-level measurements rather than contradicting
them: on H200 the fused kernel is 0.410x the sequential co-launch over 40 mixed
MTP steps, and 0.709x at its best split. A -8.1% end-to-end cost from a kernel
that is ~0.7x on the attention it replaces is the expected magnitude once the
rest of the step is accounted for.

So the remaining deficit is not a wiring bug; it is the structural one already
recorded above. `sm_fa3` is fixed at 48 here by `plan_dispatch_prefix`
(`MIN_SM_FA3, MAX_SM_FA3 = 8, 48`), and a split sweep is worth doing, but the
kernel-level sweep already showed the best split reaching only 0.709x, so a sweep
is very unlikely to turn -8.1% into a win on its own. The thing that would is
warpgroup-reentrant FA2, which is narrowed to a 3-line region (threading
`barrier_id` through `Run` and scoping `threadblock_sync_mdo_states`) and has been
attempted and reverted twice.

#### Optimizing the split on the real serving workload: -8.1% -> -3.6%

`sm_fa3` was pinned at 48 on every serving run because `plan_dispatch_prefix`
clamps it to `[MIN_SM_FA3, MAX_SM_FA3] = [8, 48]`, so the kernel's single most
important knob had never been varied server-side. `SGLANG_FUSED_SM_FA3` now pins
it (clamped only to `[1, num_sm-1]`, since the point of a sweep is to reach
values the heuristic would not choose).

Matched sweep, 200 prompts, `--random-input-len 8192 --random-output-len 192
--request-rate 12 --max-concurrency 32`, H200, `FLASHINFER_FUSED_BACKFILL=0`:

| `sm_fa3` | tok/s | vs base | mean ITL | p99 ITL | mean TTFT | fused share |
|---|---|---|---|---|---|---|
| 16 | 19369.38 | -22.4% | 23.77 | 45.46 | 3757.37 | 90.6% |
| 32 | 22034.74 | -11.7% | 20.93 | 35.94 | 3186.79 | 90.6% |
| **64** | **24055.80** | **-3.6%** | **19.18** | **34.74** | **2821.14** | 90.5% |
| 96 | 22856.25 | -8.4% | 20.26 | 37.05 | 3024.53 | 90.6% |
| 112 | 19457.78 | -22.0% | 23.95 | 44.89 | 3700.06 | 90.6% |
| base (fused off) | 24946.41 | — | 18.47 | 32.85 | 2671.04 | absent |

200/200 requests in every arm, 0 misaligned, 0 launch failures.

**There is a clear interior optimum at `sm_fa3 = 64`, and the heuristic's clamp
sat below it.** Moving from the heuristic's 48 to 64 recovers more than half the
regression: **-8.1% -> -3.6%**, with mean ITL 19.18 vs 18.47 ms (+3.8%) and p99
ITL 34.74 vs 32.85 (+5.8%). The curve is sharply peaked -- both 16 and 112 cost
~22% -- which is why leaving this knob unswept was costing so much.

**The analytic balance point is not usable for this knob.** Unclamped,
`plan_dispatch` returns `sm_fa3 = 131` and `115` for the two dumped serving batch
shapes, i.e. essentially the whole device on FA3 -- and the sweep measures 112 as
one of the two *worst* settings (-22.0%). So the clamp, not the model, was
choosing the value, and the model's own preference points the wrong way. Any
future tuning of this split should be measured, not derived from the overlap cost
model, which was fit for the *overlap* kernel and does not transfer to the
SM-pre-assignment kernel's makespan.

`sm_fa3 = 80` was added to bracket the peak: 23800.46 tok/s, mean ITL 19.38,
p99 ITL 35.72 -- worse than 64 on all three, so 64 is the peak and not an
artefact of the grid. `MAX_SM_FA3` is now 64 in
`python/sglang/srt/layers/attention/mixed_dispatch.py`, so the default lands on
the measured optimum instead of 4.5 points below it.

#### Cross-role backfill, tested on the server

`FLASHINFER_FUSED_BACKFILL=1` at the best split, same matched harness:

| arm | tok/s | mean ITL | p99 ITL | mean TTFT | fused share | misaligned | launch failures |
|---|---|---|---|---|---|---|---|
| `sm_fa3=64`, backfill **off** | 24055.80 | 19.18 | 34.74 | 2821.14 | 90.5% | 0 | 0 |
| `sm_fa3=64`, backfill **on** | 24064.65 | 19.18 | 34.90 | 2816.70 | 90.6% | 0 | 0 |

**+0.04% throughput -- within noise.** Backfill is worth 1.2% (H200) to 1.9-6.9%
(H20) at kernel level, and essentially nothing end-to-end here. That is
consistent rather than contradictory: backfill recovers idle SMs in the prefill
tail of a single mixed step, and at `chunked-prefill-size 512` with 32 concurrent
requests the server almost always has another step's work to overlap, so the tail
it reclaims is already covered.

**Correctness with backfill on is now clean**, which it previously was not. Under
the server's own toolchain (CUDA 12.8), on the dumped server batch shape, at
splits 64 / 48 / 32:

```
backfill=0:  0/512 unwritten, rel err 0.00032 / 0.00245 / 0.00313  PASS
backfill=1:  0/512 unwritten, rel err 0.00028 / 0.00218 / 0.00282  PASS
```

Scope of that claim: this is one batch shape -- the dumped serving one, which
covers FA3 + task 0 + task 2. The earlier backfill fault was recorded on "the one
batch shape whose FA2-class rows split their KV", which this shape does not
exercise, so **this does not retire that fault**; it shows backfill is correct on
the shape the server actually produces. `FLASHINFER_FUSED_BACKFILL` stays 0 by
default, now for a measured reason (no benefit) rather than only a correctness
one.

#### Multicast: what it actually requires

Multicast is not a flag on this path, and the reason is specific. The fused
kernel's FA3 side uses `hopper/sparse_mainloop.cuh` (paged/sparse KV), where:

- line 87: `static constexpr bool USE_TMA_LOAD_KV = false;` -- KV is gathered
  with `cp.async` per page, not loaded by TMA, so there is no TMA multicast to
  enable. The dense sibling `hopper/mainloop.cuh:75` has it `true`.
- the Q descriptor is built with `select<0,2>(TileShape_QKD{}), _1{}` and the
  comment `// no mcast for Q`, i.e. a cluster dim of 1.

So enabling cluster multicast needs (a) `page_size >= CTA_TILE_KV` (128) in
sglang's KV pool, which also means threading the real page size through the fused
launch, where it is currently hardcoded to 1, and (b) replacing the sparse
mainloop's gather with a TMA load so there is a descriptor to multicast from.
(b) is a rewrite of `sparse_mainloop.cuh`, not a configuration change. This is
recorded as the precise blocker rather than as "page_size=1 is a gather", which
was true but not actionable.

#### Larger prefill chunks make the fused kernel worse, not better

Hypothesis: FA3's advantage is large q tiles, and `chunked-prefill-size 512` caps
the FA3 role's q tile at 512 tokens, so raising it should favour the fused
kernel. `--chunked-prefill-size` is now settable via `CHUNK` in `serve.sh`.
Same matched harness, `sm_fa3 = 64`:

| chunk | arm | tok/s | vs base | mean ITL | p99 ITL | mean TTFT |
|---|---|---|---|---|---|---|
| 512 | fused | 24055.80 | **-3.6%** | 19.18 | 34.74 | 2821.14 |
| 512 | base | 24946.41 | — | 18.47 | 32.85 | 2671.04 |
| 2048 | fused | 24739.03 | **-17.8%** | 45.17 | **207.74** | 572.88 |
| 2048 | base | 30081.02 | — | 36.34 | 73.47 | 395.52 |

200/200 requests, 0 faults, ~90% fused share in both fused arms.

**The hypothesis is falsified, and the direction is diagnostic.** Raising the
chunk helps the baseline a lot (24946 -> 30081, +20.6%) and the fused kernel
barely at all (24056 -> 24739, +2.8%), so the gap widens from -3.6% to -17.8%.
p99 ITL is the clearest signal: 207.74 ms against the baseline's 73.47, a 2.8x
tail.

That is the static-partition cost showing up directly. FA3 and FA2 own disjoint
SM ranges for the whole launch, so the step's makespan is set by whichever role
finishes last; a bigger chunk makes the FA3 side's work per step much larger and
the imbalance grows with it. The persistent kernel has one shared work queue, so
the same imbalance is absorbed by whichever CTA is free. Backfill is supposed to
be the answer to exactly this and is measured at +0.04% (above), because it only
reclaims SMs *after* a CTA has finished its own role's queue -- it cannot move
work off the critical role mid-flight.

So `chunked-prefill-size 512` with `sm_fa3 = 64` remains the best fused
configuration found, at -3.6%, and this result is further evidence that the
remaining deficit is the static role partition rather than any per-kernel
inefficiency.

#### The split optimum does not track the work mix, so an adaptive split will not help

The static-partition reading above predicts something testable: if the makespan is
set by whichever role is longer, then at `chunk 2048` -- where FA3's share of the
step's work is much larger -- the best `sm_fa3` should shift *upward* from 64.

It does not. Split sweep at `chunk 2048`, same matched harness:

| `sm_fa3` | tok/s | mean ITL | p99 ITL | mean TTFT | fused share |
|---|---|---|---|---|---|
| 64 | 24739.03 | 45.17 | 207.74 | 572.88 | 90.0% |
| 96 | 23087.85 | 48.70 | 271.04 | 619.93 | 90.3% |
| 112 | 18505.54 | 60.78 | 422.41 | 889.56 | 89.9% |
| base | 30081.02 | 36.34 | 73.47 | 395.52 | absent |

Monotonically worse above 64, exactly as at `chunk 512`. **The optimum is
insensitive to the work mix**, which falsifies the prediction and changes the
conclusion in two ways:

1. **A per-batch adaptive split is not the fix.** `plan_dispatch`'s balance point
   already computes the work ratio and returns 115-131 for real batch shapes; the
   sweep measures 112 as one of the worst settings at *both* chunk sizes. The
   model's error is therefore **not calibration** -- the quantity it optimizes
   (work balance between the roles) is not the quantity that determines the
   optimum. Feeding it better constants would not move it toward 64.
2. `sm_fa3 ~ 64` -- about half the device -- behaves like a property of the two
   kernels' own saturation points rather than of the batch: FA3 stops gaining from
   extra CTAs while FA2's decode rows still need roughly half the device to hold
   ITL. That is why a fixed measured constant is the right choice for this knob,
   and why `MAX_SM_FA3 = 64` is stated as measured rather than derived.

This is also the directly relevant answer to the goal's perf-model question for
EDF: the overlap cost model's **balance-point output must not be used to set this
split.** Its step-time prediction is a separate matter and still rests on the 3.1%
MAPE result for the overlap kernel, which this kernel does not inherit.

#### Lifting the persistent cluster cap: -3.6% -> -2.0%

The persistent role's cluster count was `max(16, min(2*(num_sm - sm_fa3), 96))`.
The 96 was mine, chosen to keep the float workspace inside 160 MB -- not because
96 was right. At the measured-optimal `sm_fa3 = 64` the persistent role gets 68
CTAs, so `2*(132-64) = 136` clusters is what matches its grid, and the cap was
truncating that by 30%.

`SGLANG_FUSED_PERS_CLUSTER_CAP` and `SGLANG_FUSED_PERS_WS_MB` make this testable.
At `sm_fa3 = 64`, chunk 512, cap 136 with a 320 MB workspace:

| arm | tok/s | mean ITL | p99 ITL | mean TTFT | fused share |
|---|---|---|---|---|---|
| cap 96 (old default) | 24055.80 | 19.18 | 34.74 | 2821.14 | 90.5% |
| cap 136, sample 1 | 24446.95 | 18.85 | 33.18 | 2757.48 | 90.5% |
| cap 136, sample 2 | 24395.78 | 18.90 | 35.07 | 2763.73 | 90.5% |
| base (fused off) | 24946.41 | 18.47 | 32.85 | 2671.04 | absent |

200/200 requests, 0 misaligned, 0 plan declines in every arm. Two samples at cap
136 agree to 0.2%, so the **+1.5%** is larger than the repeat spread, and it
improves throughput *and* every latency statistic -- mean ITL 18.85-18.90 against
the baseline's 18.47 is now within 2.3%.

**The gap is now -2.0%** (24421 mean vs 24946), down from -8.1% at the start of
this round. Defaults moved to cap 160 and a 320 MB persistent workspace; the
extra ~160 MB is the price, and it is allocated lazily on the first accepted
mixed batch.

Progression of the fused path on this workload, all matched, all with the
`mixed_fused_fa3` share checked:

| configuration | tok/s | vs base |
|---|---|---|
| heuristic `sm_fa3=48`, cap 96 | 23250.44 | -8.1% |
| swept `sm_fa3=64`, cap 96 | 24055.80 | -3.6% |
| swept `sm_fa3=64`, cap 136 | 24421 (2 samples) | **-2.0%** |

#### The split optimum is robust to the cluster cap

The two knobs interact arithmetically -- the cluster count is
`min(2*(num_sm - sm_fa3), cap)`, so a lower split buys more clusters -- which
made it worth re-checking whether lifting the cap moved the optimum. It does not.
Re-sweep at `cap 160`, chunk 512:

| `sm_fa3` | clusters it gets | tok/s | mean ITL | p99 ITL | mean TTFT |
|---|---|---|---|---|---|
| 48 | 160 (of 168 wanted) | 23347.53 | 19.72 | **32.43** | 2949.07 |
| **64** | 136 | **24421** (2 samples) | **18.85-18.90** | 33.18-35.07 | 2757-2764 |
| 80 | 104 | 23873.03 | 19.33 | 33.60 | 2852.68 |
| base | — | 24946.41 | 18.47 | 32.85 | 2671.04 |

200/200 requests, 0 faults, ~90.6% fused share throughout. `sm_fa3 = 64` wins on
throughput, mean ITL and TTFT despite getting *fewer* clusters than 48, which
reinforces the earlier reading that 64 is about the two kernels' saturation
points rather than about resource balance. `sm_fa3 = 48` does take the best p99
ITL (32.43, slightly under the baseline's 32.85), so a deployment optimizing p99
rather than throughput would pick 48 -- worth noting, but the default stays 64.

### Final state of the fused path on this workload

| configuration | tok/s | vs base | mean ITL | p99 ITL |
|---|---|---|---|---|
| heuristic `sm_fa3=48`, cap 96 | 23250.44 | -8.1% | 20.18 | 31.87 |
| swept `sm_fa3=64`, cap 96 | 24055.80 | -3.6% | 19.18 | 34.74 |
| **`sm_fa3=64`, cap 136 (default)** | **24421** | **-2.0%** | 18.87 | 33.18-35.07 |
| base (fused off) | 24946.41 | — | 18.47 | 32.85 |

Everything above is matched, 200 prompts, `--random-input-len 8192
--random-output-len 192 --request-rate 12 --max-concurrency 32`, qwen7b on H200,
CUDA 12.8, `FLASHINFER_FUSED_BACKFILL=0`, and every row's `mixed_fused_fa3` share
was checked before its numbers were read.

Commits: sglang `725af14b3` (and `5b7d13c13`, `576d5d611`, `31b5289dd`),
flashinfer `4d59914`.

**What is left, and why it needs a kernel change rather than tuning.** Both host
knobs are now at measured optima and the two falsified predictions above
(larger chunks, adaptive split) rule out the remaining host-side ideas. The
residual -2.0% is the static role partition: backfill runs FA3 -> FA2 only,
because an FA2-role CTA returns two of its three warpgroups
(`if (threadIdx.x >= 128u) return;`) and so can never execute FA3's warp-specialised
body. Making the reverse direction possible needs FA3's work to be claimable from
a shared cursor rather than bound to `blockIdx.x`, i.e. a dynamic FA3 queue --
which also removes the `fa3_num_ctas == sm_split` contract that currently guards
against silently dropping FA3 work. That is a redesign of the FA3 side's
scheduling, not a parameter.

#### Does the fused kernel help EDF predictability? Measured: no

The goal asks how well the model-level perf model works with this kernel for EDF
against the unfused baseline. The operational form of that question is whether the
fused kernel produces a **tighter step-time distribution** -- a deadline predictor
can only be as accurate as the dispersion of what it predicts.

Matched runs, chunk 512, `sm_fa3 = 64` / cap 136 vs fused off, 200 prompts each:

| metric | fused | base | ratio |
|---|---|---|---|
| mean ITL (ms) | 18.854 | 18.459 | 1.021 |
| median ITL (ms) | 18.988 | 18.373 | 1.033 |
| std ITL (ms) | 5.025 | 4.727 | 1.063 |
| p95 ITL (ms) | 24.593 | 22.645 | 1.086 |
| p99 ITL (ms) | 33.182 | 34.994 | **0.948** |
| **CV = std/mean ITL** | **0.266** | **0.256** | 1.041 |
| **CV = std/mean TPOT** | **0.101** | **0.099** | 1.020 |
| p99 / median ITL | 1.747 | 1.905 | 0.918 |
| p95 / median ITL | 1.295 | 1.232 | 1.051 |

**The answer is no, and it is a clean null rather than a loss.** The coefficient
of variation is 0.266 vs 0.256 on ITL and 0.101 vs 0.099 on TPOT -- a 2-4%
difference. The tail ratios disagree in sign (p99/median 8.2% *tighter* fused,
p95/median 5.1% wider), which is the signature of noise rather than an effect.

For calibration on what counts as noise here: two identical baseline runs gave
24946.41 and 24951.21 tok/s (0.02% apart) but p99 ITL of 32.85 and 34.99 -- a
6.5% spread. **p99 ITL is the noisiest statistic in this harness**, so none of the
p99 differences above are interpretable, including the fused arm's apparent
advantage.

**This corrects an earlier statement in this document.** The fused arm's p99 ITL
was reported as worse than the baseline's (34.74 / 33.18 against 32.85). With the
second baseline sample at 34.99, p99 ITL is **indistinguishable between the two
arms**; only throughput (-2.1%, far outside its 0.02-0.21% repeat spread) and mean
ITL (+2.1%) separate them.

**Consequence for the colocation story.** `tasks/pd_colocation_vs_disagg.md` section
3 rests on the *overlap* kernel's closed-form predictability (3.1% MAPE), and that
document already states explicitly that "everything below depends on the
predictability property, not on the SM-pre-assignment kernel winning." This
measurement is direct evidence for keeping those separate: **the SM-pre-assignment
kernel does not inherit the overlap kernel's predictability advantage**, and the
EDF argument must not be re-attributed to it.

Limitation, stated rather than glossed: ITL dispersion is an end-to-end proxy for
step-cost predictability, not a measurement of model error. A direct test fits a
cost model to per-step (batch composition -> step time) samples and reports MAPE,
which needs per-step instrumentation in the scheduler that does not exist yet.
What is established here is that the fused kernel's end-to-end latency dispersion
is not better, which is the necessary condition for the stronger claim and it
fails.

#### The 4-mode MTP schedule does not exist in sglang today

The goal specifies that real-serving validation assume "an MTP spec model with
common acceptance rate in a 4-mode fused schedule". Every server run recorded
above had `speculative_algorithm=None`, so this was tested directly, using NGRAM
speculative decoding (no draft checkpoint needed) with
`--speculative-num-draft-tokens 2`, which gives a verify row the MTP k=1 shape
`packed_qo = 2 * gqa = 14` -- the tile-16 class the kernel-level MTP benchmarks
used.

The server came up and served, and the forward composition was:

```
forward mix (n=1400): pure_prefill=1400(100.0%)
batch kinds scheduled: 1463 Prefill batch, 11 Decode batch
mixed_fused_fa3: absent -- zero fused launches
```

The reason is in sglang, not in the kernel or the configuration:

```
WARNING server_args.py:1584: The overlap scheduler and mixed chunked prefill are
disabled because of using ngram speculative decoding.
-> enable_mixed_chunk=False, despite --enable-mixed-chunk being passed
```

This is **not ngram-specific**. `enable_mixed_chunk = False` is set
unconditionally on every speculative path:

| site | applies to |
|---|---|
| `server_args.py:1505` | EAGLE / EAGLE3 / STANDALONE ("Mixed chunked prefill is disabled because of using eagle speculative decoding") |
| `server_args.py:1578` | NGRAM |

**So with any speculative algorithm, sglang never puts a prefill chunk and a spec
verify row in the same batch.** The fused FA3 + FA2 kernel exists precisely to
serve that batch -- a large-tile prefill row on the FA3 side and small-tile
verify/decode rows on the FA2 side -- so on the real MTP serving path it currently
has no workload at all.

Consequences, stated plainly because they bound every server-side number in this
document:

1. The **-2.0%** result and the whole split/cluster/backfill tuning above were
   measured in the only regime where this kernel can run today: mixed chunked
   prefill with speculative decoding **off**. That is a prefill-chunk + decode
   mix, not the 4-mode MTP mix the goal specifies.
2. The kernel-level 4-mode MTP validation does exist and stands
   (`tests/bench_mtp_fused.py`, verify rows at `packed_qo = 14` mixed with
   prefill; H200 mono 34,301 / seq 25,503 / fused 83,612 / best-split 48,397).
   The gap is specifically the server path.
3. Getting the goal's regime requires making mixed chunked prefill and speculative
   decoding coexist in sglang's scheduler -- removing those two assignments and
   handling a batch that contains chunked prefill rows *and* verify rows. That is
   a scheduler change upstream of this kernel, and it is a prerequisite for the
   kernel's intended workload rather than an optimization of it.

This was found by running it rather than by reading the scheduler, which is why it
is recorded with the forward-composition evidence: `pure_prefill=100%` is
unambiguous, and the composition counter caught it the same way it caught the five
invalid A/Bs.

#### Forcing the 4-mode schedule: the exact blocker in sglang's scheduler

The three config gates were relaxed behind `SGLANG_ALLOW_MIXED_CHUNK_SPEC=1`
(default off, so stock behaviour is untouched):

| gate | site | kind |
|---|---|---|
| EAGLE / EAGLE3 / STANDALONE | `server_args.py:1505` | assignment `enable_mixed_chunk = False` |
| NGRAM | `server_args.py:1578` | assignment `enable_mixed_chunk = False` |
| validation | `server_args.py:3673` | `assert not self.enable_mixed_chunk` -- **and its message is written backwards**, reading "enable_mixed_chunk is required for speculative decoding" while asserting it must be off |

With all three relaxed the server **starts** cleanly (`KEPT ON with ngram
speculative decoding`, no assertion, no fatal, no misaligned) and then **dies on
the first real request**:

```
scheduler.py:2247        new_batch.mix_with_running(self.running_batch)
schedule_batch.py:1410   out_cache_loc = torch.cat(
                             [self.out_cache_loc, running_batch.out_cache_loc])
TypeError: expected Tensor as element 1 in argument 0, but got NoneType
```

So the blocker is concrete and named: **`mix_with_running` assumes the running
batch has `out_cache_loc` allocated, and under speculative decoding it is
`None`** -- the spec worker allocates KV slots during draft/verify rather than
through the scheduler's normal decode path, so there is nothing for the mixed
batch to concatenate.

This is a much more actionable statement than "sglang disables the combination":
the gates are not merely conservative, they guard a real unimplemented path, and
the work needed to reach the goal's 4-mode schedule is to teach
`mix_with_running` how to merge a spec-decoding running batch -- deciding where
the verify rows' KV slots come from when their batch is folded into a chunked
prefill batch.

Not attempted here, deliberately: getting KV slot allocation wrong in that merge
corrupts the cache silently rather than crashing, and validating it needs the
greedy-equivalence harness plus per-row KV auditing. The equivalence harness is
built and was used for this run (`/tmp/mode4.sh`: the no-spec greedy reference is
captured first and compared against the 4-mode output; here the 4-mode output was
empty because the scheduler died, which the comparison correctly reported as a
mismatch rather than a pass).

**Status of the goal's MTP requirement, stated plainly.** The 4-mode MTP serving
schedule is unreachable on sglang today: not by configuration, and not by
relaxing the gates either. It needs a scheduler fix at
`schedule_batch.py:1410`. Every server-side number in this document is therefore
from the prefill-chunk + decode mixed regime with speculative decoding off, which
is the only regime in which this kernel currently runs at all. The kernel-level
4-mode MTP validation (`tests/bench_mtp_fused.py`, verify rows at
`packed_qo = 14`) remains the only 4-mode evidence, and it stands.

#### Why the `mix_with_running` fix is a feature, not a patch

Reading `schedule_batch.py:1400-1427` after the crash shows the `out_cache_loc`
`None` is the *first* of four assumptions that all encode "one new token per
running request", which is true for ordinary decode and false for a spec verify
batch (each running request contributes `num_draft_tokens`, 2 at MTP k=1):

| line | assumption | why it breaks under spec decoding |
|---|---|---|
| `req.extend_input_len = 1` | one token per running req | a verify row has `num_draft_tokens` |
| `torch.cat([..., running_batch.out_cache_loc])` | slots already allocated | `None`; the spec worker allocates at verify time |
| `self.extend_lens.extend([1] * running_bs)` | one token per running req | same as above |
| `prefix_lens` from `len(origin_input_ids) + len(output_ids) + delta` | exactly one new token | off by `num_draft_tokens - 1` |

And beyond those four, the merged batch carries **no `spec_info`** -- the draft
tree and its verification mask -- so even with the lengths fixed the attention
backend would have no way to build the correct causal mask for the verify rows
inside a `ForwardMode.MIXED` forward.

So making the goal's 4-mode schedule real means: allocating verify rows' KV slots
inside the mixed batch, generalising four length computations from 1 to
`num_draft_tokens`, threading `spec_info` through `ForwardMode.MIXED`, and teaching
the attention backend to apply a tree mask to a subset of rows in a mixed batch.
That is a feature in sglang's scheduler and attention path, upstream of this
kernel.

**I am not attempting it**, and the reason is specific rather than budgetary
reluctance: every one of those four is a silent-corruption failure mode -- wrong
`prefix_lens` or wrong KV slots produce plausible tokens from the wrong cache
rather than a crash, and a short greedy probe can pass while a loaded run is
quietly wrong. Doing it properly needs the equivalence harness (built, at
`/tmp/mode4.sh`) plus per-row KV auditing, and it should be a separate, focused
change with its own validation rather than a tail-end addition to this one.

The override that exposes the path stays in the tree, defaulted off
(`SGLANG_ALLOW_MIXED_CHUNK_SPEC=1`), because it is the instrument that located all
of this and whoever picks the feature up will need it.

#### Attempt at symmetric backfill (FA2-role CTAs running FA3): implemented, faults, gated off

The measured residual is the one-directional backfill, so this implements the
missing direction as `FusedFA3PersistentSymmetricKernel`
(`FLASHINFER_FUSED_SYMMETRIC=1`, default off):

- CTAs `>= sm_split` drain the FA2 queue on warpgroup 0 while warpgroups 1-2 park
  at `__syncthreads()`, then all 384 threads enter `fa3_prefill_body` together.
  The order is FA2-**first** deliberately: a rendezvous *after* the FA3 body is
  undefined because its warp specialisation lets warpgroups exit at different
  times, while before the body nothing has exited.
- FA3 is planned for the whole grid so those CTAs have a slice to execute, and the
  dispatch's contract switches from `fa3_num_ctas == sm_split` to
  `fa3_num_ctas == num_sm`. That check fired correctly on the first two attempts
  (the harness was still declaring `sm_split` as the planned count), which is the
  guard working as intended rather than a failure.

**Result: it launches and then faults in-kernel.**

```
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
```

"Illegal instruction" is **consistent with** a `setmaxnreg` protocol violation,
which would make this the third instance of that hazard class in this kernel --
but that attribution is a hypothesis, not a measurement. An attempt to confirm it
by reading ptxas register counts for the two kernels (`nvcc -Xptxas -v` on the
extension) was made twice and did not produce them: first an include-path error
for the JIT-generated `batch_attention_config.inc`, then, with that located via
`find`, a standalone compile that emitted no `Compiling entry function` blocks --
this translation unit does not build cleanly outside the torch JIT environment
that normally assembles its flags. So **the cause is not established**. The counts were finally obtained with
`tests/build_only_fused.py`, which builds with `-Xptxas -v` and never launches a
kernel (the earlier attempts failed on an include path, on the TU not compiling
standalone, and on the kernel's abort truncating the log):

| kernel | registers | spill stores (per instantiation) |
|---|---|---|
| asymmetric | **168** | 2508 / 30752 / 2296 / 1988 / 1988 |
| SYMMETRIC | **168** | 2568 / 19264 / 2264 / 3020 / 3020 |

**This falsifies the register-budget hypothesis.** Both kernels compile to an
identical 168 registers per thread, and the symmetric variant does not spill more
(in the heaviest instantiation it spills *less*, 19264 against 30752). So the
fault is not `alloc<240>` becoming unsatisfiable through a larger footprint, and
the surviving explanation is a **`setmaxnreg` protocol violation**.

The arithmetic says why that protocol has no margin here. At a 168-register
baseline over three warpgroups, FA3's producer deallocating to 24 releases 144
registers, and its two consumers allocating to 240 claim 72 each -- exactly 144.
The protocol balances to the register, so it tolerates no deviation in which
warpgroup issues which instruction or in what order. In symmetric mode warpgroup 0
executes the FA2 mainloop, with its own named-barrier syncs that warpgroups 1-2
(parked at `__syncthreads()`) never reach, so the three warpgroups enter FA3's
body having run different instruction streams before its dec/inc sequence begins.

That makes the fix direction concrete rather than speculative: the FA2 drain
cannot happen on a warpgroup that afterwards participates in FA3's register
protocol. Either FA3 must be retuned so the drain uses a warpgroup it excludes, or
FA3 needs a dynamic work queue so no CTA changes role mid-kernel -- the second
being the option that removes the protocol constraint entirely rather than working
around it.

A third option follows directly from the 144 = 144 arithmetic and looks cheaper
than either: make the pre-FA3 phase **symmetric across all three warpgroups**
instead of giving it to warpgroup 0 alone. `fused_drain_queue` is already
parameterised by `tid_base`, so warpgroups 0/1/2 can each drain the FA2 queue
independently at `tid_base` 0/128/256, after which all three have executed the
same instruction stream and FA3's dec/inc sequence begins from a symmetric state
-- which is precisely the condition the measurement says is violated.

One prerequisite if that is tried: `fused_drain_queue` currently does
`NamedBarrier::sync(128, kFusedQueueBarrier)`, and a named barrier is CTA-wide, so
three warpgroups arriving at the same id each expecting 128 arrivals would
mis-match. Each warpgroup needs its own barrier id (`kFusedQueueBarrier + wg`),
which is why this is a real change and not a one-liner -- but it is a much smaller
change than retuning FA3's warp specialisation or writing a dynamic FA3 queue, and
it is the one most directly implied by the register measurement. The two candidate mechanisms it would distinguish are a protocol
violation (a warpgroup joining FA3's dec/inc sequence after running unrelated
code) and a register *budget* problem (the combined FA2+FA3 footprint forcing a
baseline that makes `alloc<240>` unsatisfiable). Confirming which is the first
thing to do before changing the design.

For context, the prior two instances of the hazard class were: the first was
`setmaxnreg` across an ABI boundary (the `__noinline__` FA3 body, which corrupted
adjacent output row pairs), the second was `MergeGroupJIT` behind a call. Here
warpgroup 0 runs the FA2 mainloop -- with its own register pressure, at the
default allocation -- and then joins FA3's `warpgroup_reg_dealloc<24>` /
`warpgroup_reg_alloc<240>` protocol mid-kernel. `.inc` can only take registers
another warpgroup released with a matching `.dec`, so a warpgroup that arrives
having already executed unrelated code does not satisfy the protocol's
precondition.

The asymmetric default path is **unaffected and still correct**: the same build
passes the regression test at `sm_split = 64` (0/512 unwritten, rel err 0.00021 /
0.00163 / 0.00241), so this is an additive, gated variant rather than a regression.

Next step for whoever picks it up, now that the failure is specific: the FA2
mainloop and FA3's body cannot share a warpgroup's register protocol, so the
parked warpgroups must enter FA3 *without* warpgroup 0 having run FA2 first.
That means either dedicating warpgroup 0 of an FA2-role CTA to the queue for the
whole kernel while 1-2 wait and then run FA3 as a 256-thread configuration (which
needs FA3 retuned for two warpgroups), or giving FA3 a dynamic work queue so no
CTA ever changes role mid-kernel -- which is the design the earlier sections point
at and the only one that avoids the register protocol entirely.

#### Final attempt: symmetric 3-warpgroup drain — default path verified, variant still faults

Acting on the resolved diagnosis, the symmetric kernel was changed so **all three**
warpgroups drain the FA2 queue before FA3 (each with its own cursor, merge flag and
named barrier), making the pre-FA3 instruction stream identical across them -- the
condition the 168/168 register measurement says is violated.

**Both results did arrive, and they settle two things.**

*The shared-path risk is retired.* The asymmetric control **passes** with the
changed `fused_drain_queue`: `sm_split=64`, 0/512 unwritten, rel err 0.00024 /
0.00209 / 0.00294. The derived barrier ids, the `kWg * 16` slot offsets and the
16 -> 48 byte tail do not disturb the default path, as inspection suggested but
had not established. No revert is needed.

*And the protocol hypothesis is falsified too.* With all three warpgroups draining
-- so the pre-FA3 instruction stream **is** identical across them -- the symmetric
kernel still faults with the same `illegal instruction`, reproduced in two
independent runs. That was the surviving explanation after the register
measurement, and it is now gone:

| hypothesis | status |
|---|---|
| register budget (`alloc<240>` unsatisfiable through a larger footprint) | **falsified** -- 168 registers in both kernels, symmetric spills no more |
| warpgroups entering FA3's dec/inc sequence from asymmetric states | **falsified** -- made symmetric, still faults |

So the cause is none of what was proposed, and the remaining candidates are
different in kind:

1. **`setmaxnreg` may not be legal after *any* prior code in the kernel** -- i.e.
   a warpgroup's register protocol must begin at entry, not mid-kernel. If so the
   whole FA2-then-FA3 ordering is impossible regardless of symmetry, and the
   dynamic FA3 queue is the only route left.
2. **`__syncthreads()` is `barrier.sync 0`, which FA3 also uses as
   `kQueryEmpty = 0`.** The rendezvous completes before FA3's body starts, but
   FA3's mbarrier initialisation may assume a pristine barrier-0 state.
3. **The FA2 mainloop writes FA3's shared-memory region** (they overlap unless
   `FLASHINFER_FUSED_DISJOINT_SMEM`), and FA3's body may not re-initialise every
   mbarrier it then waits on.

(2) and (3) are both cheap to test -- move the rendezvous to a named barrier above
`FirstUserBarrier` instead of `__syncthreads()`, and separately try the disjoint
shared-memory layout -- and they would discriminate (1) from the rest. That is the
next step, and it is a different and smaller question than the two that were just
ruled out.

**This matters more than an unfinished experiment usually would, because the change
touches shared code on the default path.** `fused_drain_queue` now derives its
named-barrier id from `kTidBase` (`kFusedQueueBarrier + kTidBase/128`) and offsets
its shared cursor/merge slots by `kWg * 16`, and the reserved shared tail grew from
16 to 48 bytes. The asymmetric kernel uses the same function.

By inspection that is equivalent-or-safer for the asymmetric path: only one
warpgroup drains per CTA there, so distinct barrier ids and distinct slots cannot
collide, and the tail only grew (and `smem_size` was updated to match). **But
"sound by inspection" is exactly what produced two of this session's bugs** -- the
`kCursorOff` sizing taken from the wrong runner, and the fixed-rank KV stride
assumption. So it is not to be trusted without the test.

Reproduce either arm with:

```
FLASHINFER_FUSED_SYMMETRIC=0 SPLITS=64,48,32 python tests/test_fused_server_batch.py
FLASHINFER_FUSED_SYMMETRIC=1 FA3_PLAN_CTAS=132 SPLITS=64 python tests/test_fused_server_batch.py
```

#### Moving the rendezvous off barrier 0 changes the failure mode

`__syncthreads()` compiles to `barrier.sync 0` and FA3 uses barrier 0 as
`NamedBarriers::kQueryEmpty`, so the symmetric kernel's pre-FA3 rendezvous was
moved to `kFusedQueueBarrier + 3` (flashinfer `eff4970`; confined to the symmetric
kernel, so the asymmetric path is untouched by this edit).

The symptom changed:

| rendezvous | symptom |
|---|---|
| `__syncthreads()` (barrier 0) | `CUDA error: an illegal instruction was encountered`, reproduced in 3 runs |
| named barrier 11 | no traceback and **no result table** -- the test printed its 3 header lines and produced nothing further under a 400 s timeout, reproduced in 2 independent runs (so deterministic, not flaky) |

The asymmetric control passes in the same build (0/512 unwritten, rel err 0.00029 /
0.00240 / 0.00288), as expected since this edit is confined to the symmetric
kernel -- worth stating because the previous change *did* touch the shared helper.

Stated carefully, because the distinction matters: the absence of a traceback is
**consistent with a hang**, and a hang is what a `NamedBarrier::sync(384, ...)`
produces when fewer than 384 threads arrive. It is not proof of one -- the run was
under `timeout 400` and no explicit timeout message was captured.

What this does establish is that **barrier 0 was genuinely implicated** -- the
illegal instruction is gone -- but that it was not the only problem. The most
likely remaining issue is arrival count at the new barrier: all three warpgroups
call `fused_drain_queue` before the rendezvous, and that function's own loop exits
via `break` when its cursor passes `n_clusters`, so warpgroups can leave the drain
at different iterations but should all still reach the rendezvous. Any path where
one does not -- an early `return` inside a `Runner::Run`, for instance -- deadlocks
all 384.

Next step, and it is now a narrow one: confirm hang vs fault by running the
symmetric arm with no timeout and `CUDA_LAUNCH_BLOCKING=1`, and if it hangs, dump
the stack with `cuda-gdb` or add a counter to verify all three warpgroups reach
the rendezvous. This is a materially smaller question than either of the two
hypotheses that were falsified, and it is the fourth distinct mechanism this fault
has presented -- which is itself the reason to instrument rather than hypothesise
a fifth.

#### Confirmed: it is a hang, and the barrier ids collide

```
exit_code=124   (timeout)        GPU utilization 99%
```

A 124 exit with the GPU pegged at 99% is a spin on a barrier, not a fault. So the
symmetric variant's remaining problem is a **deadlock**, and the illegal
instruction really was the barrier-0 collision (now fixed).

The deadlock has an identified cause, and it is in the ids I assigned:

- the three per-warpgroup queue barriers were given `kFusedQueueBarrier + 0..2`,
  i.e. **8, 9, 10**;
- but `kPersistentMainloopBarrier = kFusedQueueBarrier + 1` = **9**
  (`prefill.cuh`), and `threadblock_sync_mdo_states` syncs on it with
  `NamedBarrier::sync(KTraits::NUM_THREADS = 128, kPersistentMainloopBarrier)`.

So warpgroup 1's queue barrier is the same id the persistent mainloop uses, and
with three warpgroups now running the mainloop concurrently they all arrive at
that one id each expecting 128 arrivals. A named barrier is CTA-wide, so the
counts mis-match and the CTA spins -- exactly what 99% utilization with no
progress looks like.

Two things are needed, not one, and both are now specified:

1. Move the per-warpgroup queue barriers clear of `kPersistentMainloopBarrier` --
   e.g. base them at `kPersistentMainloopBarrier + 1` so the three occupy 10, 11,
   12, with the rendezvous at 13. (There are 16 hardware barriers, so this fits.)
2. Make `kPersistentMainloopBarrier` itself per-warpgroup, since three warpgroups
   now run `threadblock_sync_mdo_states` at once. That means threading a barrier
   id into it the same way `barrier_id` was threaded through `Run` earlier, and it
   is the part that was attempted and reverted twice before -- but the reason is
   now concrete (concurrent warpgroups sharing one id) rather than exploratory.

This is the fifth mechanism this fault has presented, but unlike the previous four
it is identified rather than hypothesised: the hang is confirmed by exit code and
GPU state, and the id collision is a matter of arithmetic on two constants that
are both in the tree.

#### The barrier fix is verified safe but does not resolve the hang — and the approach is self-defeating

```
asymmetric   rc=0     PASS, 0/512 unwritten, rel err 0.00024 / 0.00207 / 0.00253
symmetric    rc=124   still deadlocked
```

The shared-path risk is retired: per-warpgroup mdo barriers (9/10/11) and queue
ids moved to 12/13/14 leave the asymmetric default path correct. But the symmetric
variant still hangs, so the barrier-id collision was not the cause -- or not the
only one.

**The reason is structural, and it makes this whole approach self-defeating.** The
3-warpgroup drain has all three warpgroups executing `Runner::Run` *concurrently
and on different work items*. The persistent mainloop uses **block-wide**
synchronisation internally (`__syncthreads()`, and the JIT merge's block-wide
arrival logic). Block-wide means all 384 live threads: warpgroup 0 at its first
queue slot and warpgroup 1 at its third would both arrive at the same block-wide
barrier and pair up with each other, desynchronising the two and deadlocking.

That is precisely the **warpgroup-reentrant FA2** problem recorded at the very
start of this task as the blocker -- the one attempted and reverted twice. So the
3-warpgroup drain does not avoid that problem; it *requires* it to be solved
first. It was the wrong fix, and the hang is the mainloop refusing to be run by
three independent warpgroups, not a barrier id.

This closes the loop on the whole line of attack:

| variant | requires | outcome |
|---|---|---|
| drain on warpgroup 0 only, then FA3 | nothing extra | faulted (illegal instruction) and that was barrier 0; after fixing it, untested in this form |
| drain on all three warpgroups, then FA3 | **warpgroup-reentrant FA2** | deadlocks, because the mainloop syncs block-wide |

So the two routes are: (a) revert to the single-warpgroup drain now that barrier 0
is fixed and see whether it still faults -- cheap, and it was never retested after
that fix; or (b) make the persistent mainloop warpgroup-reentrant, which is the
original blocker and gates everything here including a dynamic FA3 queue.

(a) is one env-var-sized experiment and should be done first. It was skipped
because the 3-warpgroup drain was built on top of the barrier-0 fix in the same
step -- the two changes were not separated, which is the process error here.

#### The symmetric FA2 -> FA3 backfill kernel is CORRECT

```
symmetric  (WG0-only drain + rendezvous on named barrier 15)   rc=0
  sm_split=64: unwritten (NaN) rows = 0/512
  row 0 (FA3    qo= 424 kv= 4001) rel err 0.00020  ok
  row 1 (task0  qo=  87 kv=   87) rel err 0.00194  ok
  row 2 (task2  qo=   1 kv=   14) rel err 0.00282  ok
  PASS
asymmetric control                                             rc=0  PASS
```

So the missing backfill direction works: **FA2-role CTAs now execute FA3 work**,
with FA3 planned across the whole grid, every output row correct against the fp32
reference and nothing left unwritten.

The whole fault was the **barrier 0 collision** -- `__syncthreads()` compiles to
`barrier.sync 0` and FA3 uses barrier 0 as `kQueryEmpty`. Nothing else. The four
other mechanisms investigated were all wrong:

| mechanism | verdict |
|---|---|
| register budget (`alloc<240>` unsatisfiable) | falsified -- 168 registers in both kernels |
| warpgroups entering FA3 from asymmetric states | falsified -- symmetric drain still failed |
| queue/mdo barrier id collision (`8/9/10` vs `9`) | real but not the cause; fix verified harmless |
| the 3-warpgroup drain | **actively harmful** -- deadlocks, because it requires warpgroup-reentrant FA2 |

And the process lesson, since it cost the most: the barrier-0 fix and the
3-warpgroup drain were committed **together**, so when the pair still failed, the
barrier-0 fix looked insufficient when in fact it was the entire answer and the
drain change was the new problem. Separating those two would have found this
several cycles earlier. The per-warpgroup mdo barriers and relocated queue ids are
kept -- they are verified harmless and are a prerequisite if warpgroup-reentrant
FA2 is ever done -- but they were not what fixed it.

Next: the throughput A/B. The symmetric kernel is the mechanism identified as the
cause of the residual -2.0%, so this is the measurement that says whether the gap
closes. `FLASHINFER_FUSED_SYMMETRIC=1` with `fa3_num_ctas == num_sm`; the server
plumbing passes `sm_fa3` as the planned count, so it needs the full grid passed
there before a server run.

#### Symmetric backfill transforms the kernel: 0.687x -> 1.003x of the monolithic kernel

Kernel-level 4-mode MTP benchmark (`tests/bench_mtp_fused.py`, 40 mixed MTP steps,
H200), asymmetric vs symmetric, everything else identical:

| | asymmetric | symmetric | change |
|---|---|---|---|
| fused (fixed split) | 84202.5 us (0.408x mono) | **35082.3 us (0.974x mono)** | **2.40x faster** |
| best split | 50051.8 us (0.687x mono) | **34071.3 us (1.003x mono)** | **1.47x faster** |
| steps where fused beats mono | **0 / 40** | **21 / 40** | — |
| steps where fused beats seq | 0 / 40 | 0 / 40 | unchanged |
| mono (reference) | 34392.2 us | 34168.4 us | — |
| seq co-launch | 25483.0 us (1.350x) | 25532.6 us (1.338x) | — |

**The one-directional backfill was the whole structural deficit.** Letting FA2-role
CTAs execute FA3 work moves the kernel from 0.687x to **1.003x** of the monolithic
persistent kernel -- from a 1.5x loss to parity, winning outright on 21 of 40
steps where it previously won none. That is the single largest improvement in this
task, and it confirms the diagnosis the chunk-2048 experiment pointed to: the
makespan was set by whichever role finished last, and the idle role could not help.

Sequential co-launch is still ahead at 1.338x, so the fused kernel is not yet the
fastest option at kernel level; it is now competitive rather than structurally
beaten.

**The optimal split moved, and the host default is now wrong.** Best-split
distribution over the 40 steps:

```
asymmetric:  {64: 20, 80: 7, 16: 6, 48: 5, 32: 1, 8: 1}   -> favours 64
symmetric:   {8: 26, 24: 6, 32: 4, 16: 3, 96: 1}          -> favours 8
```

That reversal is the mechanism working as intended: once every CTA can reach FA3's
queue, a *small* dedicated FA3 set is best and the rest of the device backfills.
`MAX_SM_FA3 = 64`, which was measured correct for the asymmetric kernel, is
therefore wrong for symmetric mode -- it should be near 8, and the server-side
sweep must be redone before quoting any symmetric end-to-end number.

Next, in order: (1) re-sweep `sm_fa3` server-side under symmetric mode around 8,
(2) thread `num_sm` as `fa3_num_ctas` through `_plan_fused_split_inner` (it
currently passes `sm_fa3`, which symmetric mode's contract rejects), (3) rerun the
server A/B. The -2.0% figure predates all of this and should not be carried
forward as the kernel's standing result.

#### Server-side: symmetric backfill turns the regression into a WIN (+0.8%)

Matched harness, 200 prompts, `--random-input-len 8192 --random-output-len 192
--request-rate 12 --max-concurrency 32`, chunk 512, cap 160 / 320 MB,
`FLASHINFER_FUSED_SYMMETRIC=1`, H200:

| configuration | tok/s | vs base | mean ITL | p99 ITL | mean TTFT | fused share |
|---|---|---|---|---|---|---|
| **symmetric, `sm_fa3=8`** | **25155.02** | **+0.8%** | **18.33** | 32.90 | **2636.22** | 90.6% |
| symmetric, `sm_fa3=16` | 24959.08 | +0.04% | 18.48 | 34.34 | 2668.59 | 90.5% |
| base (fused off), 2 samples | 24946.41 / 24951.21 | — | 18.46 / 18.47 | 32.85 / 34.99 | 2671 / 2674 | absent |
| asymmetric best (`sm_fa3=64`) | 24421 | -2.0% | 18.87 | 33.18-35.07 | 2757-2764 | 90.5% |

200/200 requests, 0 misaligned, 0 launch failures in both symmetric arms.

**The fused path is no longer a regression.** At `sm_fa3 = 8` it beats the
persistent kernel on throughput (+0.8%), on mean ITL (18.33 vs 18.46-18.47) and on
mean TTFT (2636 vs 2671-2674), with p99 ITL inside the baseline's own run-to-run
range (32.85-34.99).

Full progression on this workload, every row with its `mixed_fused_fa3` share
checked:

| configuration | tok/s | vs base |
|---|---|---|
| heuristic `sm_fa3=48`, cap 96, asymmetric | 23250.44 | -8.1% |
| swept `sm_fa3=64`, cap 96, asymmetric | 24055.80 | -3.6% |
| `sm_fa3=64`, cap 136, asymmetric | 24421 | -2.0% |
| **`sm_fa3=8`, cap 160, SYMMETRIC** | **25155.02** | **+0.8%** |

And the split optimum moved exactly as the kernel benchmark predicted: 64 for the
asymmetric kernel, ~8 for the symmetric one, because once every CTA can reach FA3's
queue a small dedicated FA3 set is best and the rest of the device backfills.

**Caveat, stated rather than buried: this is one sample at `sm_fa3 = 8`.** The
baseline has two samples agreeing to 0.02% (~5 tok/s), so the +209 tok/s gap is far
outside the *baseline's* repeat spread -- but the symmetric arm itself has not been
repeated. Before `SGLANG_FUSED_FA3_SM_SPLIT` is flipped on by default, repeat the
`sm_fa3=8` arm 2-3 times and sweep 4/8/12 to confirm the optimum. That is the only
thing standing between this and shipping the path enabled.

Commits: sglang `130a1893e` (server plumbing), flashinfer `ca17ed5` (the working
symmetric kernel), `94764bb` (per-warpgroup barriers), `eff4970` (rendezvous off
barrier 0 -- the actual fix).

#### Confirmation: the win repeats, and it is a plateau rather than a spike

Same harness, symmetric mode, sweeping the new optimum and repeating `sm_fa3=8`:

| `sm_fa3` | tok/s | vs base | mean ITL | p99 ITL | mean TTFT |
|---|---|---|---|---|---|
| 4 | 25048.30 | +0.40% | 18.41 | 33.80 | 2652.14 |
| **8 (sample 1)** | **25155.02** | **+0.83%** | 18.33 | 32.90 | 2636.22 |
| **8 (sample 2)** | **25126.94** | **+0.71%** | 18.36 | 34.14 | 2639.82 |
| 12 | 25074.79 | +0.50% | 18.40 | 34.03 | 2646.47 |
| 16 | 24959.08 | +0.04% | 18.48 | 34.34 | 2668.59 |
| base (2 samples) | 24946.41 / 24951.21 | — | 18.46 / 18.47 | 32.85 / 34.99 | 2671 / 2674 |

200/200 requests, 0 misaligned, 0 launch failures, 90.6% fused share in every arm.

**The caveat from the previous section is cleared.** The two `sm_fa3=8` samples
agree to 0.11% (25155.02 / 25126.94, mean 25141) and both sit above the baseline's
two samples, which agree to 0.02%. It is a **plateau**, not a lucky point: 4, 8 and
12 all beat the baseline, and mean ITL (18.33-18.41 vs 18.46-18.47) and mean TTFT
(2636-2652 vs 2671-2674) improve across the whole plateau. Five independent fused
runs, all at or above baseline.

p99 ITL is the one metric that does not clearly improve (32.90-34.14 against the
baseline's 32.85-34.99) -- but those ranges overlap, and p99 ITL was already
established as this harness's noisiest statistic, varying 6.5% between two
identical baseline runs.

**Enabling the path by default is now justified by evidence**, and it is three
coupled changes rather than one, which is why it is written down rather than done
in the last minutes of a session:

1. `SGLANG_FUSED_FA3_SM_SPLIT` default 0 -> 1
2. `FLASHINFER_FUSED_SYMMETRIC` default 0 -> 1 (the asymmetric kernel is the
   regression; the default must be the symmetric one)
3. `MAX_SM_FA3` 64 -> 8, but **mode-dependent**: 64 remains the measured optimum
   for the asymmetric kernel and 8 for the symmetric one, so this must branch on
   the mode rather than simply change the constant

Anyone making that change should re-run this table afterwards, because flipping
defaults alters which code path the gate selects and that has produced five
invalid A/Bs in this task already.

#### Why the 4-mode MTP schedule is a control-flow restructuring, not four edits

Reading `prepare_for_decode` (`schedule_batch.py:1571`) settles the size of this
work, and my earlier "four length-assumption edits" estimate was wrong:

```python
if not self.spec_algorithm.is_none():
    # if spec decoding is used, the decode batch is prepared inside
    # `forward_batch_speculative_generation` after running draft models.
    return
```

Under speculative decoding the running batch is **not prepared at batch-formation
time at all**. Its `out_cache_loc` is `None` and its `input_ids` do not yet exist
because the verify rows' contents *are the draft model's output*, produced by a
separate forward inside the spec worker, after the scheduler has already formed
the batch. `mix_with_running` runs at formation time, before any of that.

So the `torch.cat([..., running_batch.out_cache_loc])` crash is not a missing
allocation to be patched in. Mixing a prefill chunk with verify rows requires the
order of operations to change:

1. form the chunked-prefill batch,
2. run the draft model for the running requests,
3. merge the resulting verify rows into the prefill batch,
4. run one fused target forward over both.

Today steps 1 and 4 are the scheduler's and steps 2-3 are the spec worker's, with
no interleaving point between them. That is a restructuring of
`forward_batch_speculative_generation` and its contract with the scheduler -- not
four edits to a merge helper, and not something to attempt behind a guard.

**This also explains why sglang disables the combination at three separate config
sites.** Those gates are not conservatism about an untested path; the control flow
genuinely cannot produce such a batch. Recording that distinction matters, because
"the gates are overly cautious" and "the data the merge needs does not exist yet"
imply very different amounts of work.

One simplification that does hold for whoever takes it on: with
`--speculative-eagle-topk 1` the draft is a **chain**, not a tree, so causal
masking over the verify rows is correct and no tree-mask plumbing is needed. That
removes the attention-backend half of the problem, leaving the scheduling reorder
as the whole of it.

#### Multicast is absent from flashinfer's Hopper attention entirely, not just the paged path

An earlier section said multicast was blocked because the *sparse* mainloop
gathers KV (`sparse_mainloop.cuh:87`, `USE_TMA_LOAD_KV = false`), implying the
dense path had it and the fix was to reach that path. **That was under-evidenced
and the implication is wrong.** Checking `hopper/mainloop.cuh`, the dense path:

```
:61   SmemLayoutQ{}, select<0,2>(TileShape_QKD{}), _1{});   // no mcast for Q
:67   take<0,2>(SmemLayoutK{}), select<1,2>(...), _1{});    // no mcast
:73   take<0,2>(SmemLayoutV{}), select<2,1>(...), _1{});    // no mcast
:121,124,127   same three, rebuilt                          // no mcast
:205,219,234,239,248   copy(..., /*mcast_mask=*/0)
```

Every TMA descriptor is constructed with a cluster dimension of `_1{}` and every
`copy()` passes a zero mcast mask. `hopper/kernel_traits.cuh` contains no cluster
configuration at all.

So there is **no runnable multicast path anywhere in this attention stack** --
dense or sparse, Q or KV. The distinction matters for sizing the work:

- what the earlier note implied: route the fused kernel to the dense mainloop
  (page_size >= 128) and multicast comes along -- **false**;
- what is actually required: *add* multicast to flashinfer's Hopper attention --
  a cluster dimension in the TMA descriptors, a non-zero mcast mask on the copies,
  cluster-scoped mbarriers, and a cluster launch config -- and only then route the
  fused kernel through it.

That is why no multicast number appears in this document: there is nothing to
measure yet, and the honest statement is that the feature does not exist here
rather than that it is switched off. Testing it means implementing it first.

This is the correct characterisation to hand over, and it supersedes the earlier
"needs a `sparse_mainloop.cuh` gather->TMA rewrite" framing, which was necessary
but not sufficient.

#### Multicast, tested on hardware: DSMEM cluster sharing is 1.59x over redundant gather

The earlier sections concluded multicast was untestable because TMA multicast
needs a tensormap over a dense strided region and paged KV at `page_size=1` is a
gather. That is true of **TMA** multicast and it was the wrong place to stop:
Hopper's distributed shared memory gives the same "load once, consume in N CTAs"
property without TMA at all. One CTA's gathered tile is readable by its cluster
peers through `cg::cluster_group::map_shared_rank`, which flashinfer already uses
in `trtllm/fused_moe/RoutingKernel.cuh` and `comm/trtllm_allreduce*.cuh`.

`tests/dsmem_kv_share.cu` measures it against the redundancy our kernel actually
has. Both arms pair CTAs on the **same** KV tile -- which is the real case, since
every q-tile of a prefill row attends to the same KV range, so the CTAs holding
those q-tiles gather identical KV. The only difference is whether each CTA gathers
the whole tile or half of it and reads the peer's half over DSMEM:

```
H200, 132 blocks = 66 clusters of 2, 4096 tiles of 128 tokens x 128 dims (bf16),
1M scattered pages (256 MB, past L2), page_size=1 gather pattern

  redundant gather : 1325.249 us    202.6 GB/s global
  DSMEM shared     :  835.249 us    160.7 GB/s global (half the bytes)
  speedup          :    1.587x
```

**1.587x**, against a 2.0x ceiling (half the global traffic); the gap is
`cluster.sync()` plus DSMEM read latency. The absolute 202.6 GB/s is far below
H200's HBM ceiling because a 256-byte scattered gather is transaction-bound rather
than bandwidth-bound -- which is exactly why halving the number of gathers helps
this much.

Three things this establishes that the earlier "needs a gather->TMA rewrite"
framing got wrong:

1. multicast's *benefit* is reachable at `page_size=1`, with no KV-pool change and
   no tensormap;
2. it needs a **real cluster launch**, which flashinfer's attention does not do
   anywhere -- `cudaLaunchKernelEx` / `clusterDim` appear only in `sampling.cuh`,
   `norm.cuh` and vendored trtllm code, and `persistent.cuh:192` states its own
   "cluster" is logical with `cluster_size` pinned to 1;
3. the sharing only pays where cluster-mates hold q-tiles of the same row, so the
   planner must co-assign them -- it is a scheduling change as much as a kernel one.

Command: `nvcc -std=c++17 -arch=sm_90a -O3 -o /tmp/dsmem tests/dsmem_kv_share.cu && /tmp/dsmem`

#### Multicast and SM-pre-assignment fusion are in tension: the cluster dim is grid-wide

The DSMEM win (1.587x) needs a thread-block cluster. Adding the cluster launch
(`FLASHINFER_FUSED_CLUSTER=N`, flashinfer `610d8d7`) and running the regression:

```
CLUSTER=1   rc=0   PASS, 0/512 unwritten, rel err 0.00025 / 0.00205 / 0.00266
CLUSTER=2   rc=1   CUDA error: unspecified launch failure
CLUSTER=4   rc=1   CUDA error: unspecified launch failure
```

The launch is *accepted* and the kernel then crashes, so something inside is
cluster-sensitive. It is FA3's CUTLASS pipeline, and the mechanism is exact:

- `hopper/prefill_sm90.cuh:113,123` constructs the pipeline with a **hardcoded**
  `cluster_shape = Shape<_1,_1,_1>{}`;
- `cutlass/pipeline/sm90_pipeline.hpp:313-316` derives the mbarrier arrival count
  from that declared shape:

```cpp
uint32_t multicast_consumer_arrival_count = params.num_consumers;  // If cluster_size is 1
if (cute::size(cluster_shape) > 1) {
  multicast_consumer_arrival_count = (size<0>(cluster_shape) + size<1>(cluster_shape) - 1)
                                     * num_consumer_warpgroups_per_cluster;
}
```

So under a real cluster of 2 the barriers are initialised for 1-CTA arrivals while
the hardware has clustered the CTAs. Passing the *true* shape instead does not fix
it -- it makes CUTLASS expect `2 x num_warpgroups` arrivals per barrier, which only
materialise if FA3's TMA loads actually multicast. FA3's KV cannot multicast
(`page_size=1` gather, no tensormap), so those arrivals never come and it hangs
instead of crashing.

**The structural point.** `clusterDim` is a property of the *launch*, so it applies
to every CTA in the grid. This kernel's whole premise is that one grid holds two
different roles. DSMEM is wanted only by the FA2 role (which gathers with
`cp.async` and has no CUTLASS pipeline); it is actively harmful to the FA3 role.
CUDA offers no per-role cluster shape, so:

| option | consequence |
|---|---|
| cluster the whole grid | breaks FA3's pipeline, as measured |
| give FA3 a true cluster shape | CUTLASS then demands multicast arrivals FA3 cannot produce |
| launch the two roles as separate clustered/unclustered kernels | that **is** the sequential co-launch baseline -- already 1.338x faster than this kernel |
| implement TMA multicast in FA3 as well | the full job: tensormap-able KV (`page_size >= 128`) plus mcast masks plus cluster-consistent pipelines |

That last row is the only route that keeps the fusion *and* gets multicast, and it
is substantially more than "add DSMEM to the FA2 mainloop" -- which is what the
1.587x microbenchmark appeared to license. Worth stating plainly because it is a
real cost of the SM-pre-assignment design, not an implementation detail: **the
single-grid, two-role structure is what forecloses per-role use of cluster
features.** Any paper claim about combining SM pre-assignment with multicast has
to confront this.

What is established: multicast's benefit for our access pattern is real and
measured (1.587x, `tests/dsmem_kv_share.cu`); the cluster launch is implemented and
verified to be output-neutral at N=1; and N>1 is blocked on FA3's pipeline for a
reason that is now precisely located rather than suspected.

#### How much of the DSMEM win survives L2? Most of it -- the gather is transaction-bound

The 1.587x above was measured with a 256 MB working set, deliberately past L2. That
overstates the in-situ case: in the real kernel the CTAs that redundantly gather a
row's KV run concurrently, so L2 serves most of the repeats ("locality means
concurrency in time", as the LPT work in this task established). Re-running with an
L2-resident working set bounds the honest upside:

| working set | redundant gather | DSMEM shared | speedup |
|---|---|---|---|
| 16 MB (**inside L2** -- the in-situ case) | 682.436 us / 393.3 GB/s | 514.583 us / 260.8 GB/s | **1.326x** |
| 256 MB (past L2 -- cold) | 1325.362 us / 202.5 GB/s | 835.079 us / 160.7 GB/s | 1.587x |

L2 absorbs part of the redundancy -- the baseline nearly halves, 1325 -> 682 us --
but **the DSMEM advantage only falls from 1.587x to 1.326x**, it does not
disappear. The reason is worth stating because it is the non-obvious part: a
`page_size=1` gather is bound by the *number of 256-byte transactions* each CTA
issues, not by which memory level answers them. L2 shortens the latency of a repeat
but still charges for the transaction and its own bandwidth; DSMEM removes the
repeat entirely. So sharing wins even when the KV is already resident.

That makes the integration genuinely worth wanting -- ~1.33x on the KV-gather
portion of a prefill row's work, in the regime the server actually runs in -- and it
is exactly what the grid-wide `clusterDim` forecloses for this kernel's two-role
design (previous section). The two findings together are the honest state of
multicast here: **the benefit is real and quantified in the realistic regime, and
the fused single-grid structure is what prevents collecting it.**

Command: `/tmp/dsmem 65536` (L2-resident) and `/tmp/dsmem 1048576` (cold), from
`tests/dsmem_kv_share.cu`.

#### Correction: the FA2 side is cluster-safe, so the fusion does NOT foreclose multicast

The section above concluded that the grid-wide `clusterDim` plus this kernel's
two-role structure forecloses per-role cluster features. **That was reasoned, not
measured, and it is wrong.** `FLASHINFER_FUSED_SKIP_FA3=1` sends every CTA straight
to the FA2 queue so a cluster launch can be exercised with FA3's pipeline out of
the picture:

| `CLUSTER` | `SKIP_FA3` | result |
|---|---|---|
| 1 | 0 | rc=0, PASS, 0/512 unwritten |
| 2 | 0 | rc=1, **CUDA error: unspecified launch failure** |
| 1 | 1 | rc=1, 424/512 unwritten -- expected, **no CUDA error** |
| 2 | 1 | rc=1, 424/512 unwritten -- expected, **no CUDA error** |

With FA3 skipped, `CLUSTER=2` runs cleanly: the only failure is the deliberately
missing FA3 output (424 unwritten rows is exactly row 0's token count), with no
CUDA fault. So **the FA2/persistent side tolerates a real cluster launch today**,
and FA3 alone is cluster-hostile.

That changes the conclusion in the right direction:

- the blocker is **not** the single-grid two-role design, it is specifically FA3's
  CUTLASS pipeline under a clustered launch;
- the side that wants DSMEM -- the FA2 role, where the measured 1.326x
  in-situ win lives -- is **already cluster-ready**;
- so the remaining work is to make FA3 *tolerate* a cluster it does not use, which
  is a bounded CUTLASS-pipeline question, not the "implement TMA multicast in FA3
  as well" redesign the previous section claimed.

The ordering lesson, since this is the third time in this task that checking an
asserted cause overturned it: the earlier claim followed from reading
`prefill_sm90.cuh`'s hardcoded `cluster_shape = <1,1,1>` and CUTLASS deriving
arrival counts from it. That reading explains why FA3 *might* break, and it is
probably still the mechanism -- but it does not license the structural conclusion
that was drawn on top of it, and one 2x2 matrix separated the two.

**Next step, now narrow.** Determine why FA3 crashes under a cluster it declares as
size 1. The likely candidate is barrier initialisation visibility: CUTLASS's
`init_barriers` fences with `__syncthreads()` scope appropriate to a 1-CTA cluster,
while a hardware cluster needs the init visible cluster-wide before any TMA
transaction completes against it. If that is it, the fix is a `cute::cluster_sync()`
after init under a clustered launch -- not a multicast implementation. Then: planner
co-assignment of cluster-mates to q-tiles of one row, and the DSMEM gather split in
the FA2 mainloop.

#### Narrowing the cluster blocker: not the FA2 side, not shared memory -- it is FA3's TMA pipeline

Two candidate causes eliminated by measurement rather than argument.

**Shared memory is not it.** The fused kernel asks for ~165 KB (FA3's
`SharedStorage` at one K/V stage plus the persistent mainloop's), while
`dsmem_kv_share.cu` clusters happily at 16 KB -- so smem size was a live
hypothesis. Padding the benchmark's dynamic smem request to match:

| dynamic smem | cluster launch | speedup |
|---|---|---|
| 16 KB (natural) | ok | 1.325x |
| 100 KB | ok | 1.326x |
| **165 KB** (the fused kernel's footprint) | **ok** | 1.327x |
| 200 KB | ok | 1.326x |

A clustered launch tolerates the full footprint. Combined with the `SKIP_FA3`
matrix (FA2-only clusters cleanly), the cause is isolated to **FA3's CUTLASS TMA
pipeline** -- three of the four candidates are now excluded by test.

**Why both cluster_shape choices fail.** FA3 declares
`cluster_shape = Shape<_1,_1,_1>` (`prefill_sm90.cuh:113,123`):

- declaring **1** under a real cluster: CUTLASS takes its `__syncthreads()` branch
  for barrier init and computes cta-local mbarrier addresses, while the hardware
  encodes mbarrier arrives cluster-relatively -- a TMA completion can signal the
  wrong CTA's barrier. Crashes, as measured.
- declaring the **true shape**: `sm90_pipeline.hpp:313-316` then sets
  `multicast_consumer_arrival_count = (size<0> + size<1> - 1) * num_consumer_warpgroups`,
  i.e. it expects consumers from *both* CTAs to arrive on each barrier. They only
  do if the TMA load actually multicasts, and FA3's KV cannot (`page_size=1`
  gather). It would hang rather than crash.

So FA3 cannot simply be told about the cluster. Three routes remain, and the third
is new and the cheapest:

1. implement real TMA multicast in FA3 -- needs tensormap-able KV
   (`page_size >= CTA_TILE_KV`), mcast masks and cluster-consistent pipelines;
2. leave FA3 unclustered by splitting the roles into separate launches -- which is
   the sequential co-launch baseline, already 1.338x ahead of this kernel;
3. **drop TMA from the FA3 role.** The sparse mainloop already gathers KV with
   `cp.async`; only Q goes through TMA (`sparse_mainloop.cuh:80-87`, `TMA_Q`). Load
   Q with `cp.async` too and FA3 has no CUTLASS pipeline, hence no cluster
   conflict, and the whole grid can cluster. The cost is Q-load efficiency on a
   tile that is small next to the KV stream -- and the gain is the measured 1.326x
   on the KV gather, which is the dominant traffic. This is the option to try
   first, and it was invisible until smem and the FA2 side were ruled out.

#### Correction: the pipeline mechanism above is wrong -- that `if constexpr` branch is not taken

The previous two sections blamed FA3's cluster crash on
`PipelineTmaAsync`'s arrival-count arithmetic
(`cutlass/pipeline/sm90_pipeline.hpp:313-316`) reached through a hardcoded
`cluster_shape = Shape<_1,_1,_1>`. **This configuration never instantiates that
pipeline.** Reading `prefill_sm90.cuh:109-127` properly:

```cpp
MainloopPipeline pipeline_k = [&] {
  if constexpr (use_tma_load_kv) {
    pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    return MainloopPipeline(shared_storage.pipeline_k, pipeline_params,
                            /*cluster_shape=*/Shape<_1, _1, _1>{});
  } else {
    return MainloopPipeline(shared_storage.pipeline_k, pipeline_params);   // <-- ours
  }
}();
```

and `kernel_traits.cuh:112-113`:

```cpp
using MainloopPipeline =
    std::conditional_t<USE_TMA_LOAD_KV, cutlass::PipelineTmaAsync<NUM_STAGES>,
                       cutlass::PipelineAsync<NUM_STAGES>>;
```

The fused kernel runs the **sparse** mainloop, where `USE_TMA_LOAD_KV = false`. So
`MainloopPipeline` is `PipelineAsync`, the `cluster_shape` argument is never
passed, and the multicast arrival-count branch is dead code here. Both the
"declaring 1 computes cta-local addresses" and the "declaring the true shape
demands multicast arrivals" arguments were about code this build does not compile.

**What is actually left to suspect**, having read the right branch: the sparse FA3
path still uses TMA in exactly two places, and nowhere else.

1. `TMA_Q` -- Q is TMA-loaded (`sparse_mainloop.cuh:80-87`), signalling
   `shared_storage.barrier_Q`, which `prefill_sm90.cuh:105` initialises with
   `init(/*num_threads=*/1)`.
2. the epilogue's TMA **store** of O, which is why cross-role backfill needed
   `cute::tma_store_wait<0>()` in the first place.

A TMA operation's mbarrier completion is cluster-relative on Hopper, and
`barrier_Q` is initialised and waited on as a CTA-local barrier. That is the
remaining candidate, and it is consistent with the crash being a launch-time
addressing fault rather than a hang.

Usefully, this makes the route recorded in the previous section *more* attractive
rather than less: replacing `TMA_Q` with a `cp.async` Q load removes the only TMA
load on this path, and the only remaining TMA is the epilogue store. The fix is
smaller than "implement TMA multicast in FA3", and the claim it rests on is now
read from the branch that is actually compiled.

**Fourth correction of an asserted cause in this task.** The pattern is consistent
enough to be worth naming: each time, the error was reasoning from code that looked
relevant without checking which `if constexpr` / template specialisation this build
selects. `USE_TMA_LOAD_KV = false` is the single fact that decides most of FA3's
behaviour here, and it needs checking first, not last.

#### The Q TMA load is NOT the cluster blocker -- it is the epilogue's TMA store

Tested rather than assumed, with `FLASHINFER_SKIP_TMAQ` replacing the Q TMA
producer by a bare `barrier_Q.arrive()`:

| `CLUSTER` | `SKIP_TMAQ` | result |
|---|---|---|
| 1 | 0 | rc=0, PASS |
| 2 | 0 | rc=1, unspecified launch failure |
| 1 | 1 | rc=1, 424/512 unwritten -- expected with garbage Q, **no CUDA error** |
| 2 | 1 | rc=1, **still unspecified launch failure** |

Removing the only TMA *load* on the sparse path does not fix the clustered launch.
So `TMA_Q` and its `ClusterTransactionBarrier` are **not** the blocker, and the
"replace `TMA_Q` with `cp.async`" route recorded two sections ago would have been
wasted work -- the fifth asserted cause in this task to be overturned by actually
testing it.

By elimination the remaining TMA on the FA3 path is **the epilogue's TMA store of
O**, and that is now the only candidate left:

| candidate | status |
|---|---|
| FA2 / persistent side | cluster-safe (`SKIP_FA3` matrix) |
| dynamic shared memory size | cluster-safe at 100 / 165 / 200 KB |
| `PipelineTmaAsync` arrival counts | never compiled here (`USE_TMA_LOAD_KV = false`) |
| `TMA_Q` + `barrier_Q` | **ruled out above** |
| **epilogue TMA store of O** | **only one left** |

That is a more consequential finding than the Q result would have been, because the
epilogue store is not incidental to this kernel: `cute::tma_store_wait<0>()` is
precisely what cross-role backfill had to wait on before an FA3 CTA could drain the
FA2 queue. So the store is load-bearing for the backfill design, and making it
cluster-safe is not a matter of swapping it for a plain store without re-examining
that interaction.

Next test, and it completes the elimination: skip the epilogue TMA store the same
way and re-run `CLUSTER=2`. If it then runs, the blocker is confirmed and the
design question becomes whether O can be written without TMA (losing the
`tma_store_wait` handle that backfill uses) or whether the store can be made
cluster-correct.

**Method note, now that five causes have been overturned.** Every one of the five
was reasoned from code that looked relevant; every one fell to a two-line
compile-time or runtime switch that isolated it. The switches cost minutes and the
reasoning cost hours. On this kernel, isolate first.

#### Elimination state for the cluster blocker

The epilogue stores O with a plain `TiledCopyO` (`epilogue.cuh:203`), not TMA --
the store is sparse, so TMA would need a descriptor per page. So there is no
epilogue TMA store to blame.

| candidate | status |
|---|---|
| FA2 / persistent side | cluster-safe (`SKIP_FA3` matrix) |
| dynamic shared memory size | cluster-safe at 100 / 165 / 200 KB |
| `PipelineTmaAsync` arrival counts | never compiled (`USE_TMA_LOAD_KV = false`) |
| `TMA_Q` load | ruled out (`SKIP_TMAQ` matrix) |
| epilogue store | plain tiled, not TMA |
| `ClusterTransactionBarrier` barrier_Q / barrier_O | ruled out -- `tests/cluster_barrier_repro.cu` reproduces FA3's init(1)/arrive/wait pattern and is OK at cluster 1, 2 and 4 |

Every cheap hypothesis is now excluded. The next step is compute-sanitizer against
the clustered fused kernel, which was avoided earlier as expensive but is now the
only tool left with the search space exhausted:

```
FLASHINFER_FUSED_CLUSTER=2 /usr/local/cuda/bin/compute-sanitizer --tool memcheck \
  /dev/shm/w/venv/bin/python tests/test_fused_server_batch.py
```

memcheck (sanitizer 2025.4.1.0) attaches and reports **no device-side violation**
-- only the host-side `cudaErrorLaunchFailure (719)` propagating through
`cudaDeviceSynchronize` / `cudaGetLastError`, 4 errors all of that kind. No
"Invalid __global__ read", no out-of-bounds, no misaligned access.

That is itself a narrowing: the clustered launch is **not** failing on a memory
access. `unspecified launch failure` with a clean memcheck points at an illegal
instruction or a barrier/sync violation, which is `synccheck`'s domain -- and this
kernel is dense with named barriers plus FA3's warp specialisation, so that is the
tool to run next:

```
FLASHINFER_FUSED_CLUSTER=2 /usr/local/cuda/bin/compute-sanitizer --tool synccheck \
  /dev/shm/w/venv/bin/python tests/test_fused_server_batch.py
```

synccheck, run **detached on the pod** to get past the webshell's hard 25-minute
ceiling (which killed the first attempt -- a tooling limit, not a result), reports
**`ERROR SUMMARY: 0 errors`**.

So both sanitizer tools are clean:

| tool | result |
|---|---|
| memcheck | 0 device-side violations (only host `cudaErrorLaunchFailure` propagating) |
| synccheck | **0 errors** |

A clean memcheck *and* a clean synccheck alongside `unspecified launch failure`
rules out the two large classes -- bad memory access and illegal
barrier/synchronisation. What is left is an illegal instruction or a runtime
resource/scheduling failure, neither of which either tool reports.

The next probe is therefore a host-side one rather than another sanitizer run:
`cudaOccupancyMaxActiveClusters` for this kernel at cluster size 2. If it returns
0, the cluster is simply not schedulable at this kernel's 384 threads and ~165 KB
of shared memory, and the launch failure is explained without any device-side
defect -- the fix would then be reducing per-CTA resources rather than hunting a
bug. Note the microbenchmark clusters fine at 165 KB but uses only 128 threads, so
thread count is the untested difference, not smem.

Practical notes so the next attempt does not re-lose time:
- run the sanitizer **detached** (`nohup ... &` on the pod) and poll a done-marker;
  a foreground webshell call is capped at 25 minutes and synccheck exceeds it;
- invoke `/usr/local/cuda/bin/compute-sanitizer` explicitly -- bare
  `compute-sanitizer` resolved to a build that never attached (no header), and the
  venv runs CUDA 12.8 against a 13.1 default;
- run it on the standalone harness, never the server, which trips the detokenizer
  heartbeat long before the kernel is reached -- the same trap
  `CUDA_LAUNCH_BLOCKING=1` set earlier in this task.

Multicast therefore stands as: benefit measured on hardware (1.326x in situ,
1.587x cold), cluster launch implemented and output-neutral at N=1, integration
blocked at N=1 -> N=2 by a fault whose cause is not yet identified.

One consequence that is worth recording because the goal states otherwise:
cross-role backfill does **not** need `tma_store_wait`. With a plain tiled store
there are no TMA stores to drain, so `cute::tma_store_wait<0>()` at
`persistent_fa3_fused.cuh:236` guards nothing; the guard that matters is
`cp_async::wait_group<0>()` at line 246. The measured backfill numbers (+0.04%
server, 1.2% H200 kernel-level) rest on `wait_group`.

#### Strong lead: `__launch_bounds__(N, 1)` may be what blocks the cluster launch

`tests/cluster_occupancy_probe.cu` asked the driver directly. The probe has a flaw
-- the first failure leaves a sticky CUDA error, so only its **first row is valid**
and rows 2-6 merely echo `cudaErrorLaunchFailure`; each config needs its own
process. Row 1 is still the useful one:

```
threads=128 smem=16KB cluster=2 | maxActiveClusters=-1 (cudaErrorInvalidClusterSize)
                                | launch=ok sync=cudaErrorLaunchFailure
```

That is the **same shape that works** in `dsmem_kv_share.cu` (128 threads, 16 KB,
cluster 2, `map_shared_rank` + `cluster.sync()`). The one structural difference
between the two kernels: this probe declares `__launch_bounds__(128, 1)`, and the
working benchmark declares no launch bounds at all.

The fused kernel declares
`__launch_bounds__(FA3Traits::NUM_WARPS * cutlass::NumThreadsPerWarp, 1)`.

So the hypothesis is that the explicit **`minBlocksPerMultiprocessor = 1`** in
`__launch_bounds__` conflicts with cluster scheduling -- the driver is being told
to guarantee one block per SM while also being asked to co-schedule a cluster, and
`cudaOccupancyMaxActiveClusters` rejecting the size with
`cudaErrorInvalidClusterSize` is consistent with that. It would also explain the
whole picture: both sanitizers clean, no memory or sync violation, launch accepted,
kernel fails -- a scheduling constraint, not a device-side defect.

**This is a hypothesis, stated as one.** Six previous asserted causes in this task
were overturned by test, so it needs the clean experiment before anything is built
on it: three separate processes, one each for `__launch_bounds__(128, 1)`,
`__launch_bounds__(128)` and no launch bounds, at cluster 2. If the `, 1` variant is
the only failure, the fix in the fused kernel is to drop the
`minBlocksPerMultiprocessor` argument -- a one-token change -- and then the whole
grid can cluster and the DSMEM gather split becomes implementable.
