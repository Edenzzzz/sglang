# Native FA3 + native FA2 on the FA3 serving backend

This is an isolated integration overlay, not a change to the main task document
or a claim that the kernel serving target is closed.

## Implemented path

`native_fa3_fa2_pdl_20261023.py` connects the validated stock-lineage FA3
producer and native FA2 row executor directly to the **native FA3 backend**.
The producer uses a bounded persistent grid, releases its PDL successor early,
and completes its TMA stores. Independent FA2 rows execute through the native
Q16/Q32 path and wait for the producer before allowing the downstream consumer.
The two roles write disjoint, contiguous slices of the final attention output.
No gather/scatter or second stream is used in this path.

The first supported serving configuration is causal BF16, head dimension 128,
page size one, chain speculation. The Q32 override requires 17–32 packed query
rows and a compatible original FA2 plan. Other small-row plans retain their
native tile. Unmixed, unsupported, pure-decode, and pure-verify forwards retain
native FA3. GLM TP4's four-token verify rows do not satisfy the small-row gate.

`apply_native_pdl_fa3_backend_20261023.py` installs the opt-in seam in a separate
copy of the frozen Python tree. Original native FA3 backend SHA-256:
`53489175aace7b97498efb10ae8b01c489d35e91058842eee02b19aec18f1a3c`.
Enable with `SGLANG_NATIVE_PDL=1`; `SGLANG_NATIVE_PDL_VERIFY=1` compares every
selected layer against full-batch native FA3 and is diagnostic, not timed.

## Combined GPU gate

On the existing H20 worker-4, all three combined GPU tests passed:

- Three head/GQA geometries, three FA3 grid budgets, serial and PDL execution.
- Six changed-query/key/value CUDA-graph replays, poisoned output/workspace,
  full-output projection, and bitwise comparison against the serial executor.
- Two 393,216-token row contexts, one- and four-token rows, and full FP32
  causal references. Worst relative head error in the timeline fixtures: 0.004375.

Device timestamps prove FA2 starts before FA3 finishes for a 512-token chunk
over a 32K prefix beside the long verify rows. The 128-token fresh-prefill
fixtures did **not** overlap: FA3 finished before FA2 started. Thus PDL support
does not imply useful overlap for every workload. The FA2 end timestamp is
the earliest CTA completion, not a grid-wide end; only the start-before-FA3-end
relation is used as overlap evidence.

Logs and binary/source hashes:
`../../pd_colocation/results_20261023/native_pdl_combined_h20_gate.json`.
Four host partition/metadata regressions also pass.

## Serving comparison

The first two serving attempts completed requests but were **rejected for
missing native-kernel activation**. The native runner normalizes `MIXED` to
`EXTEND` before initializing attention metadata (`runner/eager_runner.py`),
so checking only the original mode silently bypassed the new executor.
The integration now recognizes normalized extend batches and partitions by
their exact host query lengths; it still requires real merged verify-token
evidence before accepting the four-mode serving gate.

Original attempts are preserved under `/dev/shm/w/native_pdl_serving_20261023`
and `/dev/shm/w/native_pdl_serving_v2_20261023`. The corrected comparison uses
`/dev/shm/w/native_pdl_serving_v3_20261023`: real Qwen2.5-7B + trained EAGLE3,
three draft steps/four verify tokens, natural-text foreground requests and
long-running background decodes, same schedule/backend for all arms. It
retains complete generated token sequences and a repeated baseline; per-layer
numerical verification is enabled only for the candidate. Full output
agreement is reported separately from runtime and attention checks.

The corrected run completed all 16 requests in each arm with no request
failures or server tracebacks. Actual native PDL calls on four-token verify
rows are logged through call 1,201; every selected layer is checked against
native FA3 with a strict 0.02 per-head relative-error limit. The existing
correctness driver's printed request count includes metadata fields; the
audited request count is 16, not its printed 34.

**Full generated-output equality is not established.** The baseline repeats
match completely on 8/16 requests; candidate versus the two baselines matches
6/16 and 7/16. For foreground requests, minimum common prefix is 52 tokens
between baselines versus 11 for the candidate. Background responses can
diverge immediately even between baselines, but the candidate also introduces
additional early divergences. This is not dismissed as noise or converted
into a full serving-output pass. A same-input full-target shadow comparison
is being used to separate kernel numerical changes from scheduling effects:
two native-FA3 reference forwards, then the candidate, with no intervening
sampling or verify-slot allocation. It separately records reference-repeat
equality, full-logit errors, argmax changes, and their top-logit margins.

Complete outputs/logs: `../../pd_colocation/results_20261023/native_pdl_serving_h20_gate.tgz`;
comparisons: `../../pd_colocation/results_20261023/native_pdl_serving_comparison.json`.
No throughput or ITL gain is claimed from these correctness runs.

## Attention timing with strong controls

All 190 attention-only timing records pass changed-query/full-FP32-output
checks (maximum relative head error 0.004340). Both eager and graph execution
are measured. Controls tune FA3 SplitKV (8/32) and GQA packing, include two
co-running FA3 calls, two-stream FA3+native-FA2, and the same native kernels
launched serially. Selected eager times, milliseconds:

| Fixture | Best single FA3 | Best dual FA3 | Best native PDL | Native two-stream |
|---|---:|---:|---:|---:|
| Qwen7, fresh q128 + 2 × 393K verify | 1.5774 | 0.9332 | 0.7775 | 0.7827 |
| Qwen7, q512@4K + 2 × 393K verify | 1.7822 | 1.2365 | 1.0859 | 0.9944 |
| Qwen7, q512@4K + 786K verify | 1.7802 | 1.2314 | 1.0713 | 1.0446 |
| Qwen7, q2048@32K + 2 × 393K verify | 8.5881 | 8.1855 | 8.0262 | 8.0270 |
| Qwen7, q512@4K + 2 × 393K decode | 1.7830 | 1.2381 | 0.8539 | 0.8581 |
| Qwen3, q512@4K + 393K verify | 1.7872 | 1.2503 | 0.9259 | 0.8516 |

The useful gain is principally **role-specific FA2 execution**, not PDL
itself: PDL improves the same native serial pair by only about 1–6%, and
often loses to the two-stream native control. GLM TP4 verify remains
ineligible; its native-FA2 two-stream control loses to tuned dual FA3.

Reserving 16 SMs for FA2 increases the large-prefill fixture from 8.0262 to
10.3575 ms; reserving 32 raises it to 12.8419 ms. The default FA3 grid budget
therefore uses all available SMs, with early release/hardware backfill rather
than a forced reservation. Explicit smaller budgets remain available for
controlled experiments. This is a dispatch choice supported by attention
measurements, not an end-to-end performance claim.

Raw records: `../../pd_colocation/results_20261023/native_pdl_long_mix_h20.json`.

## Opt-in FP32 partials

`SGLANG_NATIVE_PDL_FP32_PARTIALS=1` selects the separate
`native_fa2_q32_fp32` operator, loaded from
`NATIVE_PDL_FA2_FP32_LIBRARY`. This retains FA2 partial outputs and their
merge in FP32, followed by one BF16 conversion into the normal row slice.
Q/K/V and tensor-core probability operands remain BF16; this is not the
independent full-FP32 attention reference. The original BF16-partial path
remains the default, and both precisions have separate executor caches.

The public planner already reserves FP32-sized partial storage. Explicit
ABI/offset/workspace checks guard its reuse. The build changes only a
private copy of the merge dispatcher, restricting it to the supported
128-dimensional head rather than instantiating an unsupported FP32/512-head
copy. Installed FlashInfer and the baseline native module are unchanged.
`load_native_fp32_partials_20261023.py` generates the isolated variant with
checked source anchors. `build_fp32_partials_h200_20261023.sh` builds and gates
it; `fp32_partials_captured_h20_20261023.sh` repeats the real-capture matrix.
Results, precision improvements, latency costs and limitations are recorded
in `captured_kernel_results_20261023.md`; this option is not promoted yet.

`SGLANG_NATIVE_PDL_DIRECT_BF16_OUTPUT=1` additionally uses
`NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY` for SplitKV plans. The merge writes BF16
directly into the output slice while retaining FP32 partials and reduction,
removing the separate conversion launch. Unsplit plans use the previous
conversion implementation. Both variants have distinct operator namespaces
and executor caches. All 29 captured inputs are bitwise equal between them;
the measured launch-elimination benefit is modest, not a new large speedup.

## Same-batch full-model audit

**Metric correction:** the original comparator used the first `topk(2)` index
as an argmax. These differ on tied logits, and EAGLE's greedy verifier uses
`torch.argmax`. The historical count of 38 below is therefore a **top-k winner
change count, not a verified greedy-argmax count**. Raw records are retained;
the comparator now uses true argmax and has two CPU regressions. The original
records do not contain enough logits to reconstruct every corrected count.

The initial full-model shadow stopped at a 0.02844 relative-logit discrepancy,
despite zero argmax changes in that first step. It is preserved, not counted
as a pass. The follow-up collects the full cohort while retaining the 0.02
diagnostic threshold as a failed result. It additionally executes the same
native FA3+FA2 pair **serially** before PDL, on identical inputs and allocated
verify slots. The native default uses the measured full-SM grid budget; its
three combined GPU gates were rerun and passed before this serving audit.

Across **125 mixed target forwards / 4,187 logit rows**:

- Repeated stock-FA3 references are bitwise identical on every forward.
- Native serial and native PDL logits are bitwise identical on every forward.
- Native versus stock FA3 changes 38 top-k winners, all in verify rows; no
  prefill top-k winner changes. Positions 0/1/2/3 account for 1/4/17/16 changes.
- Flipped rows' reference top-two margin has median 0 and maximum 0.25;
  33/38 candidate argmaxes are among the reference top two.
- Worst relative logit error is 0.14733; maximum absolute difference is 2.3125.
  119/125 forwards exceed the diagnostic 0.02 relative-logit limit, so the
  report finishes with exit status **3**, not an accepted full-model gate.

This isolates the sampled discrepancy to the native arithmetic path rather
than PDL overlap/order. It does **not** establish that the changes are harmless:
accepted-position attribution and probability-space comparison are still
missing. The 0.02 full-model diagnostic is also a different metric from the
per-head attention limit; passing one cannot establish the other. Further
attribution should separate the custom FA3 producer from native FA2 rows,
and measure probability/log-probability changes before serving-performance
claims or promoting this executor.

All requests in the follow-up completed without server tracebacks. Full
records, source snapshots, logs, original failed attempt and repeated GPU
gate are in `../../pd_colocation/results_20261023/native_pdl_model_shadow_h20.tgz`;
the aggregate is `../../pd_colocation/results_20261023/native_pdl_model_shadow_summary.json`.
