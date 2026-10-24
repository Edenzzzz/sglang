# Fixed geometry policy: held-out context evaluation

The earlier captured 32–64K timings use per-shape tuning. They demonstrate
kernel headroom but do not provide an online policy. `geometry_v1` is now
implemented and frozen before capturing the 128–256K and 300–400K workloads.
The policy reads only exact host query/KV lengths, head counts and SM budget;
it does not inspect activations, request identity, measured latency or SLOs.

- For at most 512 prefill tokens over at least an 8K cached context, select
  FA3 SplitKV from 1/2/4/8 by minimizing a simple CTA-wave × KV-tile estimate.
  Other prefill shapes stay unsplit. Ties select fewer splits.
- FA2's chunk is the power of two at or below longest-row-KV / 32, bounded
  to 1024–8192 tokens. This gives long rows enough partitions even when
  many short rows make the default batch-wide plan too coarse.
- Use a 512-MiB FA2 scratch buffer. Choices are stored per forward, not
  mutated global split counts. Pure/unrecognized modes retain native FA3.

This is a heuristic developed from the earlier small-context captures,
not a claim of optimality. It is an opt-in serving setting:
`SGLANG_NATIVE_PDL_SPLIT_POLICY=geometry_v1`. Both kernel and serving evidence
must pass before promoting it. No EDF, routing or admission rule changes.

The frozen source hash and definition are recorded in
`../../pd_colocation/results_20261023/native_geometry_policy_20261023.json`.
Five host tests include the policy's short/long-context choices. The GPU gate
adds sequential replanning across different context regimes with one shared
executor, checking FP32 attention and exact native serial/PDL agreement.

`captured_long_policy_h20_20261023.sh` gates the implementation, then captures
and benchmarks two-mode and real four-mode serving at 128–256K TraceLab-derived
session lengths and at 393,216-token natural-text fan-out prefixes. Original
physical KV layouts are retained, warmup inputs excluded, and all request
outputs validated. The server is stopped before each kernel matrix. Both
the frozen policy and its matched serial executor are compared against
default/tuned single FA3, tuned dual FA3 and the original untuned native path.

The long-context protocol uses Qwen2.5-7B with YaRN factor 16 / 524,288-token
context and real trained EAGLE3 where applicable; that extension and the
TraceLab loader's generated contents remain explicit limitations.

## Completed long-context captures

Both modes completed in both ranges: 13 additional actual captures and 624
paired eager/graph timing records, all passing FP32 checks (maximum head
relative error 0.005972). Each TraceLab-derived arm completed eight agent
and 480 chat requests; each 394K fan-out arm completed two agent requests
and 1,000 chats, with no request errors. **The fan-out runs completed only
round zero:** their 180-second program window expired during cold prefill.
These are valid cold-start captures, not evidence about warm multi-turn
long-context serving. Missing capture buckets are not filled synthetically.

Raw requests, protocols and logs are in
`../../pd_colocation/results_20261023/captured_long_policy_records.tgz`.
All 13 tensors (8,269,118,466 bytes) are downloaded and SHA-256 verified under
`/opt/tiger/captured_mixed_inputs_20261023/`; see
`../../pd_colocation/results_20261023/captured_long_inputs_manifest.json`.

## Expanded FA3 controls

The earlier dual-FA3 comparison tuned only the row splits. A stronger matrix
now also varies **prefill splits 0/1/2/4/8/16/32 and prefill GQA packing
auto/off/on**, crossed with row splits 0/8/32 (row packing remains automatic).
All 3,150 opposite-order graph records over all 21 captures pass the same
FP32 gate, maximum relative head error 0.008223. Results and complete settings:
`../../pd_colocation/results_20261023/captured_fa3_role_controls/`.

Against these expanded controls, the frozen rule's short-prefix captures
at 394K retain **1.63–1.74× two-mode** and **1.24–1.25× four-mode** speedups.
Some earlier cached-prefill advantages narrow: the 32–64K four-mode cached
case is 0.529 ms versus the stronger dual-FA3 control's 0.565 ms (1.07×),
not the older comparison against 0.650 ms. These remain per-capture kernel
times, not workload-weighted serving improvements or exhaustive FA3 tuning.

**V1 is not promoted.** A heterogeneous 394K four-mode capture regresses:
14.40 ms versus 7.13 ms for the best tested FA3 control. Its 2,047 prefill
tokens include a 133-token query over a 394,107-token cached prefix and
many cheap fresh-prefill queries. The aggregate-token guard hides the costly
small cached query and incorrectly leaves all prefill work unsplit.

## Revised selector: development regression only

An explicit split diagnostic on that unchanged capture reduces native time
from 14.40 to 6.58 ms with 16 prefill splits; all 32 diagnostic records pass
FP32 checks. `geometry_v2` therefore identifies the compute-dominant prefill
row using query-length × KV-length, rather than classifying by aggregate
query count, and extends the candidate split counts through 16/32.

All 21 existing captures pass the v2 development regression: 336 fresh
eager/graph records, maximum head-relative error 0.007802, and exact matched
native serial/PDL outputs. The former failing capture becomes **6.60 ms**
versus the selected FA3 control's **7.15 ms**. The four-mode 151K cached
capture improves from v1's **4.23 to 3.06 ms**, versus **5.46 ms** for FA3.
Large-prefill 32–64K cases still have small regressions of roughly 2%.

This uses the previously measured captures to develop the fix; it is **not
held-out validation of v2** and does not establish serving gains. The v1
source and manifest are preserved in `geometry_v1_source_snapshot.tgz` under
the result directory; its failed result is not rewritten. V2 records are
in `../../pd_colocation/results_20261023/geometry_v2_regression/` and the
isolated split diagnostic is in `../../pd_colocation/results_20261023/deep_prefill_diagnostic/`.

## Packed-GQA work-count correction

The stronger SM-grid FA3 baseline exposed a 13.4% regression on the
Qwen2.5 two-mode 32–64K cached/small capture. Role decomposition shows
FA2 rows are not the problem: native rows take 0.125 ms versus FA3's
0.189 ms. The native prefill takes 0.835 ms versus FA3's 0.741 ms. A
controlled 72-setting split/SM-budget diagnostic finds the complete native
step falls from 0.937 to 0.739 ms with 32 prefill splits instead of eight;
all 32 decomposition/validation records and 72 search records pass FP32.

The estimator's tile count was inconsistent with the compiled kernel.
The actual FA3 mainloop, epilogue and dynamic scheduler all use packed GQA
with 128 packed-query rows per tile. V2 counted
`sum(ceil(query / 128)) * query_heads`, which is the unpacked layout.
For 79 query tokens at 28/4 heads that predicts 28 tiles instead of the
actual `ceil(79 * 7 / 128) * 4 = 20`. This changes the wave estimate and
selects too few splits. The active H20 build's include path and source
confirm `PackGQA=true`; this is not inferred from timing alone.

`geometry_v3` corrects that count to
`sum(ceil(query * (query_heads / kv_heads) / 128)) * kv_heads`.
The dominant-query guard, candidate split counts, FA2 chunk rule and SM
budget are unchanged. Ten host tests include ragged packed rows and GQA
ratio one. The new version is explicit; v1/v2 and their evidence are preserved.

All 29 existing captures pass **580 fresh eager/graph records**, with exact
native serial/PDL outputs and maximum head-relative FP32 error **0.004554**.
Only three captures change their split counts:

| Capture | V2 → V3 splits | V2 ms | V3 ms | Strong FA3 ms | FA3 / V3 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 two, 32–64K cached/small | 8 → 32 | 0.9348 | **0.7358** | 0.8235 | **1.12×** |
| Qwen2.5 four, 32–64K cached/small | 8 → 32 | 0.5289 | **0.5143** | 0.5607 | **1.09×** |
| Qwen3 four, 32–64K cached/small | 4 → 16 | 2.4290 | **2.1576** | 3.5974 | **1.67×** |

The previously regressing cached/small case now beats the stronger FA3
control. Four unchanged 32–64K captures still trail slightly; this is not
a universal winner. These 29 captures are a development regression, not
held-out v3 validation. A new 64–128K/new-seed capture pass freezes the
policy before collecting inputs and will be reported separately.

Raw diagnosis: `../../pd_colocation/results_20261023/diagnose_cached_prefill/`.
Complete v3 records: `../../pd_colocation/results_20261023/geometry_v3_regression/`.

## Frozen v3: new 64–128K cohort

The policy source is frozen before capturing a new 64–128K context band
with agent selection seed 20261024. Both two-mode and real four-mode
serving complete 16 agent requests at concurrency four and 720 natural
ShareGPT chats at 6 requests/s, without request errors. Actual four-mode
tokens per verify are 3.360 for agents and 1.987 for chat. Capture starts
after the benchmark barrier; native parameters are not retuned on this cohort.

The first attempt is rejected, not reused: the source trace contains 514
identical duplicate round records, and the selected cohort hit one of them.
The session-concurrency validator rejected duplicate logical round IDs.
The rerun retains the same seed and uses a separately hashed copy that
removes only identical complete records under `(provider, session, round)`;
conflicting identities would fail. All 357,161 original records are audited,
356,647 unique records retained, with no length clipping or round renumbering.
The original dataset is unchanged. The workload launcher now propagates
every client failure instead of letting a successful chat client mask an
agent failure. The failed attempt remains archived alongside the rerun.

All eight new captures pass 160 fresh eager/graph validation records after
the full FA3 split/packing/SM-grid control search. Maximum head-relative
error is 0.004172; native serial/PDL outputs agree exactly.

| Capture | Strong FA3 (ms) | Frozen v3 (ms) | Speedup |
|---|---:|---:|---:|
| Two / fresh small | 0.4145 | 0.2679 | **1.55×** |
| Two / cached small | 4.6733 | 4.3126 | **1.08×** |
| Two / fresh large | 0.6015 | 0.5536 | **1.09×** |
| Two / cached large | 1.4731 | 1.4244 | **1.03×** |
| Four / fresh small | 0.3200 | 0.2846 | **1.12×** |
| Four / cached small | 4.2246 | 3.8941 | **1.08×** |
| Four / fresh large | 0.6040 | 0.5872 | **1.03×** |
| Four / cached large | 1.4733 | 1.4480 | **1.02×** |

For the new two-mode cached/small case v3 chooses **eight splits instead
of v2's 32**, improving 4.4098 → 4.3126 ms. Thus the correction is not a
blanket increase to 32 splits. All other plans are unchanged on this cohort.
This is held-out context/cohort evidence, still a first-eligible bucket
sample and TraceLab-derived lengths rather than original agent text.

Requests, freeze timestamps/hashes, duplicate audit, rejected-attempt logs,
all controls and timings: `../../pd_colocation/results_20261023/geometry_v3_holdout/`.
All eight raw captures are locally SHA-256 verified at
`/opt/tiger/captured_mixed_inputs_20261023/geometry_v3_holdout/`, totaling
4,046,615,388 bytes. No end-to-end v3 serving gain is inferred from this pass.
