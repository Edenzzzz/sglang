# Kernel results from captured two-mode and four-mode serving

## What was captured

These are **actual model Q/K/V tensors**, not synthetic attention fixtures.
Both runs use Qwen2.5-7B on one H20, layer 14, BF16, head dimension 128,
28 query heads / 4 KV heads, page size one, and 2048-token prefill chunks.

- Two-mode: prefill plus one-token decode rows, without speculation.
- Four-mode: the real trained EAGLE3 head, three draft steps / four verify
  tokens, with prefill and verify rows sharing the target forward. Actual
  weighted tokens per verify are 2.013 for chat and 3.542 for agents; no
  simulated acceptance. The latter uses generated agent prompt contents.
- Workload: eight TraceLab-derived sessions, two turns each, program
  concurrency four, 32–64K contexts, plus 480 natural ShareGPT chats at
  8 requests/s with chat templates. TraceLab supplies session/round shapes,
  not original prompt contents; recorded inter-turn delays are omitted.
- Each run completes all 16 agent and 480 chat requests without errors.
  The existing four-mode eligibility gate remains 8K context / four queued
  requests; no EDF or routing changes are introduced.

Capture starts **after the workload barrier**, excluding warmup. Four
first-eligible buckets per mode cover short/cached prefill prefixes and
small/large prefill query counts. A shape census is retained. This is a
shape-coverage sample, not a random or throughput-weighted workload sample.
The replay restores original physical KV page IDs, full cache shapes and
strides; unused slots are poisoned. Live K/V are not repacked to improve
locality. The server is stopped before kernel timings.

## Fresh paired validation

The later **SM-grid baseline check** at the end of this note supersedes
cached-case comparisons where a stronger FA3 control was found. In
particular, the Qwen2.5 two-mode 32–64K cached/small case is now a regression,
not a win for v2. The subsequent packed-GQA estimator correction (v3),
documented below, removes this regression. Earlier raw measurements remain
preserved as historical controls.

Times below are milliseconds, CUDA-graph replay, averaged over two opposite
arm orders. Each order has three timing samples of 20 executions. Settings
are selected in an earlier tuning pass, then independently remeasured on
the same captures. These are **per-shape tuned microbenchmarks**, not a
deployed dispatch policy or held-out workload generalization.

Single-FA3 controls include default and SplitKV 8/32 with automatic/forced
GQA packing. Dual-FA3 controls use two streams and row SplitKV 0/8/32.
The serial native control uses the **same selected FA3 and FA2 split plan**
as PDL, including the FA3 reduction when needed.

| Mode / capture | Single FA3 | Dual FA3 | Native serial | Native PDL | Speedup vs single / dual |
|---|---:|---:|---:|---:|---:|
| Two, short-prefix / small query | 0.3443 | 0.2025 | 0.1499 | **0.1344** | **2.56× / 1.51×** |
| Two, cached-prefix / small query | 1.2504 | 1.1254 | 0.9724 | **0.9309** | **1.34× / 1.21×** |
| Two, short-prefix / large query | 0.8186 | 0.7405 | 0.7428 | 0.7291 | 1.12× / 1.02× |
| Two, cached-prefix / large query | 1.5836 | 1.5851 | 1.7185 | 1.5863 | 1.00× / 1.00× |
| Four, short-prefix / small query | 0.2222 | 0.1261 | 0.1332 | **0.1226** | **1.81× / 1.03×** |
| Four, cached-prefix / small query | 0.9744 | 0.6506 | 0.5598 | **0.5214** | **1.87× / 1.25×** |
| Four, short-prefix / large query | 0.8179 | 0.7385 | 0.7508 | 0.7419 | 1.10× / 1.00× |
| Four, cached-prefix / large query | 1.5840 | 1.5803 | 1.6734 | 1.5905 | 1.00× / 0.99× |

The two-mode short-prefix/small-query batch is 84 prefill tokens plus 47
one-token rows, including agent contexts of about 35K, 38K and 58K. The
four-mode cached-prefix/small-query batch is 71 prefill tokens over a 39,800
token prefix plus seven four-token verify rows, including 45K, 51K and 66K
contexts. The large-query captures have 2046 prefill tokens and two running
rows. Full per-request lengths are in the raw metadata, not replaced with
these rounded descriptions.

**Eager measurements agree:** the two-mode short-prefix/small-query case
is 0.3467 / 0.2085 / 0.1380 ms for single FA3 / dual FA3 / PDL; the four-mode
cached-prefix/small-query case is 0.9770 / 0.6501 / 0.5242 ms. Thus the
representative gains are not an artifact of graph-only replay.

There is **no universal gain**: large-prefill cases are approximately tied
with dual FA3, including slight regressions. PDL's incremental gain over
matched native serial is only about 1–12%; role-specific attention and
SplitKV planning account for most of the larger advantage over FA3.

## What changed in the kernel path

Actual heterogeneous batches exposed two weaknesses absent from the small
synthetic mixes: unsplit FA3 has too little parallel work for short queries
over deep cached prefixes, and FA2's automatic split plan leaves long rows
behind many short rows. The native path now supports FA3 SplitKV plus its
post-PDL reduction, explicit FA2 KV chunk sizes, and an explicit scratch
budget. The tuning pass tests FA3 splits 1/2/4/8 and FA2 automatic/1024/4096/
8192-token chunks. No scheduler, routing, or request-order change is involved.

The full tuning matrix has 800 records. All **128 fresh validation records**
pass full-output FP32 causal-attention checks, with maximum per-head relative
error **0.006534**, below the pre-existing 0.02 limit. Matched native serial
and PDL outputs are bitwise identical on all eight actual captures. Four GPU
regressions additionally cover SplitKV reductions, changed-input graphs,
poisoned workspaces and the downstream full-output consumer.

## Reproduction and limits

- Fresh results, selected settings, source and logs:
  `../../pd_colocation/results_20261023/captured_pair_validation/`.
- Complete tuning results:
  `../../pd_colocation/results_20261023/captured_tuning_final/`.
- Requests, measured speculative counters, server logs and protocol:
  `../../pd_colocation/results_20261023/captured_workload_requests.tgz`.
- All eight raw tensor captures are downloaded and SHA-256 verified at
  `/opt/tiger/captured_mixed_inputs_20261023/` (1,772,160,348 bytes).
- `validate_captured_pair_20261023.py` replays the frozen selected settings;
  `bench_captured_mixed_20261023.py` restores physical layouts and computes
  the full FP32 reference. GPU time excludes capture I/O and once-per-forward
  planning; no end-to-end throughput, TTFT or ITL improvement is inferred.

The abandoned first capture included warmup inputs and is not used here.
The initial four-mode attempt incorrectly equated all small-query kernel
rows with verify rows; tiny prefill rows can also belong to FA2. That capture
check is corrected and covered by three CPU tests, and the four-mode workload
is rerun successfully. Completed two-mode captures are reused unchanged.
The larger 128–256K and approximately 394K captured cases are now measured
in `geometry_policy_evaluation_20261023.md`. That follow-up also expands
the dual-FA3 prefill split/packing controls; the earlier table here must not
be described as using that newer control grid. In particular, some cached
prefill advantages shrink. No synthetic long-context fixture substitutes
for the newly captured cases.
Main task-document structure and large TODOs are unchanged.

## Updated comparison against expanded FA3 controls

The v2 development regression remeasures the best previously selected
single/dual-FA3 control against one geometry-only native policy on all 21
captures. The expanded control grid tunes prefill splits and GQA packing
as well as row splits; row packing stays automatic. Representative graph
times from that matched remeasurement are:

| Captured mode / context / bucket | Selected FA3 (ms) | Native serial (ms) | Native PDL (ms) | FA3 / native PDL |
|---|---:|---:|---:|---:|
| Two / 32–64K / fresh small | 0.2016 | 0.1508 | 0.1350 | **1.49×** |
| Four / 32–64K / cached small | 0.5634 | 0.5609 | 0.5255 | **1.07×** |
| Two / 128–256K / cached small | 2.7972 | 2.2206 | 2.1904 | **1.28×** |
| Four / 128–256K / cached small | 5.4609 | 3.0710 | 3.0634 | **1.78×** |
| Two / ~394K / fresh small | 0.9334 | 0.5544 | 0.5360 | **1.74×** |
| Four / ~394K / fresh small | 0.9355 | 0.7656 | 0.7510 | **1.25×** |
| Four / ~394K / heterogeneous cached large | 7.1534 | 6.7521 | 6.6027 | **1.08×** |
| Four / 32–64K / cached large | 1.5883 | 1.6316 | 1.6243 | **0.98×** |

Full 21-case / 336-record evidence, including eager measurements and both
arm orders, is in `../../pd_colocation/results_20261023/geometry_v2_regression/`.
Maximum full-output head-relative error against FP32 is 0.007802; matched
native serial and PDL outputs are exact. These are actual captured tensor
replays, not end-to-end gains. V2 was developed on these captures, the ~394K
workload is cold-start only, and prefill-heavy captures can still regress.
Most of the large gains come from heterogeneous attention and its split
plans, not PDL alone. Model-level emitted-token equivalence remains open;
see `serving_output_gate_followup_20261023.md` for the independent FP32
adjudication rather than treating the kernel tolerance as serving parity.

## FP32-partial precision ablation

A separately named native FA2 operator now retains partial outputs and
SplitKV merging in FP32, then converts the row result once to BF16. Q/K/V,
tensor-core inputs, native Q32 geometry, FA3 prefix work and the v2 split
policy are unchanged. The final conversion is **included** in the timings.
The native module privately specializes the merge dispatcher to the already
required 128-dimensional heads; it does not modify installed FlashInfer.
The existing public BF16 plan already reserves FP32-sized partial storage;
the executor verifies those bounds rather than changing plan offsets.

Five tests pass on each of H20 and H200, covering unchanged BF16 inputs,
public-plan reuse, exact original-tile unsplit BF16 outputs, FP32 reference
checks, changing inputs, poisoned scratch, graph replay and a downstream
BF16 consumer. Matched native serial/PDL outputs are bitwise identical.

All 21 unchanged captures complete another **504 eager/graph records**,
with both arm orders and full-output FP32 checks (maximum head-relative
error 0.007802). Native-row relative L2 error decreases in every capture:
**1.9–40.8%, median 14.1%** relative to the BF16-partial native path. It is
lower than the selected FA3 control on 20/21 captures, not universally.
These error summaries are unweighted across the selected capture buckets.

| Captured mode / context / bucket | Selected FA3 (ms) | BF16-partial PDL (ms) | FP32-partial PDL (ms) | FA3 / FP32 PDL |
|---|---:|---:|---:|---:|
| Two / 32–64K / fresh small | 0.2015 | 0.1343 | 0.1388 | **1.45×** |
| Four / 32–64K / cached small | 0.5634 | 0.5255 | 0.5297 | **1.06×** |
| Two / 128–256K / cached small | 2.7880 | 2.1763 | 2.1775 | **1.28×** |
| Four / 128–256K / cached small | 5.4170 | 3.0433 | 3.0491 | **1.78×** |
| Two / ~394K / fresh small | 0.9444 | 0.5416 | 0.5453 | **1.73×** |
| Four / ~394K / fresh small | 0.9345 | 0.7483 | 0.7661 | **1.22×** |
| Four / ~394K / heterogeneous cached large | 7.1544 | 6.6039 | 6.6139 | **1.08×** |
| Four / 32–64K / fresh small | 0.1254 | 0.1226 | 0.1291 | **0.97×** |

The additional precision costs a median **0.54%** graph latency across
captures, up to **5.28%** on the smallest four-mode batch. It retains the
large-context advantages but can erase a small existing win. No automatic
promotion or new accuracy threshold follows from this ablation. Identical
same-input serving outputs still require a separate gate.

Complete sources, failed-build logs, successful tests, hashes, every timing
sample and per-capture precision metrics:
`../../pd_colocation/results_20261023/fp32_partials_captured/`.
H200 build/test evidence:
`../../pd_colocation/results_20261023/fp32_partials_h200_gates/`.

## Additional FA3 row-packing controls

The baseline check now also varies row GQA packing (automatic/off/on) and
row SplitKV (0/1/2/4/8/16/32/64), rather than leaving row packing automatic.
It starts from the earlier best dual-FA3 prefill setting, chooses row settings,
then retunes prefill splits 0/1/2/4/8/16/32 and packing for that row setting.
A separate full-batch FA3 search covers those seven split counts and three
packing choices. This is coordinate tuning, not an exhaustive cross-product.

All **1,386 search records and 588 fresh validation records** pass the full
FP32 gate (maximum head-relative error 0.007802). Validation uses eager and
graph execution in opposite arm orders and retains the earlier best control
as a separate arm, so a poor new selection cannot erase a stronger baseline.
The main representative graph advantages survive: **1.45×** for two-mode
32–64K fresh/small, **1.72×** for two-mode ~394K fresh/small, **1.78×** for
four-mode 128–256K cached/small and **1.22×** for four-mode ~394K fresh/small.
Four-mode 32–64K fresh/small and cached/large remain roughly 3–4% slower.

One new dual-FA3 selection on the two-mode 128–256K cached/small case times
at 2.73 ms during graph search and 2.74 ms in eager validation, but 4.35 ms
after graph recapture in validation. The retained full-FA3 control is 2.81 ms
and native is 2.18 ms. Do not use the slower recaptured dual arm alone to
inflate the gain: versus the fastest observed FA3 timing the gain is about
1.25×, rather than treating the 4.35 ms result as the baseline. The raw
recapture discrepancy is retained, not explained away as normal noise.

Every selected setting, timing sample and source is archived in
`../../pd_colocation/results_20261023/fa3_row_packing/`.

## Second-model captures: Qwen3-8B

The same native policy and executor were frozen before new Qwen3-8B
captures. This changes the head geometry to 32 query / 8 KV heads, uses the
trained AngelSlim EAGLE3 draft head, and changes the agent selection seed to
20261023. Query head dimension remains 128, BF16, page one, layer 14 and
2,048-token chunks. Qwen3's original 40,960-position context is explicitly
recorded in its YaRN extension; this is not an unmodified native-context test.

Each mode completes 16 TraceLab-derived agent requests at concurrency four
and 480 natural ShareGPT chats at 8 requests/s, without request errors.
Actual four-mode tokens per verify are **3.137 agent / 1.821 chat**. Eight
after-barrier captures preserve physical KV indices and cache strides.
The same fixed policy is evaluated without tuning native parameters on
Qwen3; FA3 controls use the expanded row-packing/split and full-batch search.

| Qwen3 capture | Best tested FA3 (ms) | FP32-partial native PDL (ms) | Speedup |
|---|---:|---:|---:|
| Two / fresh small | 0.3967 | 0.2523 | **1.57×** |
| Two / cached small | 1.7425 | 1.4767 | **1.18×** |
| Two / fresh large | 0.6265 | 0.5751 | **1.09×** |
| Two / cached large | 1.6273 | 1.5805 | **1.03×** |
| Four / fresh small | 0.4081 | 0.2853 | **1.43×** |
| Four / cached small | 3.5977 | 2.4306 | **1.48×** |
| Four / fresh large | 0.3914 | 0.3654 | **1.07×** |
| Four / cached large | 1.4041 | 1.3720 | **1.02×** |

All 528 new control-search records and 224 fresh eager/graph validation
records pass FP32 checks, maximum head-relative error **0.005405**. Matched
native serial/PDL outputs are exact. With Qwen3's GQA ratio four, the
four-token rows use the original Q16 path rather than the Q32 override;
these gains are not evidence that Q32 specifically improves this model.
This is new-model/new-seed evidence for the frozen policy, still based on
first-eligible shape buckets rather than a workload-weighted sample.

Metadata, requests, complete controls, policy freeze hashes and logs:
`../../pd_colocation/results_20261023/qwen3_captured/`.
All eight raw captures are SHA-256 verified locally under
`/opt/tiger/captured_mixed_inputs_20261023/qwen3/`, totaling 3,281,874,012 bytes.

## Fused merge conversion

The FP32-partial path no longer needs a separate FP32-to-BF16 copy launch
when FA2 uses SplitKV. A separate native operator keeps the same FP32
attention partials and reduction, but writes the reduction's result directly
as BF16 into the final mixed-output row slice. Unsplit plans retain the
previous FP32-output-plus-conversion path; the direct operator explicitly
rejects unsplit metadata. FA3 prefix execution and its reduction are unchanged.

Four H20 tests pass, including Qwen2.5, Qwen3 and GLM TP4 head geometries,
changed-input graphs, poisoned workspaces, a downstream BF16 consumer and
the unsplit fallback. All **29 actual Qwen2.5/Qwen3 captures** pass **464 new
eager/graph records**. Direct output is **bitwise equal** to the prior
separate-conversion path and to matched serial execution on every capture;
maximum head-relative error versus FP32 is 0.004554.

The gain is small and measured as such: graph latency falls a median
**0.23%**, up to **1.90%** on Qwen2.5's smallest four-mode capture. That case
improves from 0.1290 to 0.1266 ms but still trails its 0.1255 ms FA3 control.
Qwen3 fresh/small improves to **0.2503 ms two-mode / 0.2821 ms four-mode**,
versus FA3 0.3986 / 0.4098 ms (**1.59× / 1.45×**). These are complete measured
steps including the final output, not conversion-only timing estimates.

Source, private generated headers, binary hash, tests and full records:
`../../pd_colocation/results_20261023/native_direct_output/`.
The option is `SGLANG_NATIVE_PDL_DIRECT_BF16_OUTPUT=1` together with FP32
partials; it remains separate from the earlier preserved serving results.

## SM-grid baseline check: strongest current controls

The final control pass adds FA3 prefill margins 0/8/…/72 on the H20's 78
SMs. Each nonzero margin is tested with either unrestricted row work or a
complementary row-grid budget, keeping the selected row/prefill split and
packing choices. These are grid caps, not exclusive physical SM assignment.
All **551 search records and 348 fresh eager/graph records** across 29
captures pass the FP32 gate, maximum head-relative error **0.004554**.

Most principal gains survive: Qwen2.5 two-mode fresh/small is **1.47× at
32–64K / 1.74× at ~394K**; four-mode cached/small at 128–256K is **1.78×**
and fresh/small at ~394K is **1.23×**. Qwen3 remains **1.59× two-mode
fresh/small**, **1.45× four-mode fresh/small** and **1.48× four-mode cached/small**.

**A stronger baseline reverses one earlier win:** Qwen2.5 two-mode
32–64K cached/small improves from FA3's earlier 1.11 ms to **0.8236 ms**
with SM-grid tuning, while native is **0.9343 ms**, a **13.4% regression**.
It must not be presented using the earlier ~1.19× advantage. Prefill-heavy
32–64K captures also remain small regressions. The two-mode 128–256K
cached/small dual-FA3 control now recaptures at **2.743 ms**; native is
**2.188 ms**, confirming the conservative **1.25×** comparison noted above.

The complete plot and machine-readable table include every capture, not
just favorable examples:
`../../pd_colocation/results_20261023/captured_sm_controls/captured_native_all29.svg`
and `captured_native_all29.csv` in the same directory. The accompanying
JSON hashes the raw source records. Full tests, settings and measurements
are in `../../pd_colocation/results_20261023/captured_sm_controls/`.

## V3 corrects the cached-prefill regression

The FA3 executor uses packed GQA but the v2 split estimator counted unpacked
query tiles. Correcting the count changes only three of the 29 prior
captures. In fresh measurements, Qwen2.5 two-mode 32–64K cached/small improves
from **0.9348 to 0.7358 ms**, versus the strong FA3 control's **0.8235 ms**:
the former regression becomes a **1.12×** gain. Qwen3 four-mode cached/small
improves from **2.4290 to 2.1576 ms**, versus FA3 **3.5974 ms (1.67×)**.
No FA2 chunk or serving-scheduler setting changes.

All 580 fresh records pass FP32 checks; matched native serial/PDL outputs
are exact. Four unchanged prefill-heavy/small four-mode cases still have
small regressions. This is development validation on existing captures;
the v2 plot and its adverse result are not replaced or erased. Full cause,
formula, role decomposition and results are in
`geometry_policy_evaluation_20261023.md` and
`../../pd_colocation/results_20261023/geometry_v3_regression/`.
