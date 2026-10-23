# Serving-output gate: actual committed tokens

The historical `38` count was a top-k winner count, not a true greedy-argmax
count. The corrected comparator uses `torch.argmax`; tied-logit regressions
cover the distinction. Raw prior records remain unchanged and are not
retroactively assigned a corrected count.

The next isolated native-policy gate adds two checks beyond raw logits:

1. Compare full softmax distributions using reference-to-candidate KL and
   total variation, plus centered-logit error. Raw relative-logit error is
   retained, but a constant logit shift changes that metric while leaving
   both probabilities and outputs unchanged. A CPU regression demonstrates
   this directly; the earlier 0.02 cutoff was not a semantic output criterion.
2. Run the production greedy EAGLE sampler on cloned reference/serial/native
   logits with the same draft tree and live, read-only penalty snapshot.
   Compare emitted token sequences, accepted lengths and consumed positions,
   not unused verify positions. Check the native probe against the actual
   committed verify result. Prefill is similarly checked with cloned logits
   through the production preprocessing/sampler, deliberately omitting
   `ModelRunner.sample`'s n-gram-state update; the actual prefill result must
   agree with that probe too.

Probes require greedy requests without grammar or active sampling observers.
They never allocate verify slots, commit KV, update n-gram history, or replace
production outputs. Probe/production disagreement is a failure, not evidence
that the candidate is wrong or right. Baseline self-repeat and native
serial/PDL logits must still match bitwise. Every native attention layer is
also checked against the full-batch FA3 reference.

The serving-output verdict requires identical same-input **emitted tokens**
for both prefill and verify, with independent checks that the probes equal
the actual production samplers. The old raw-logit cutoff is printed as a
separate diagnostic; exceeding it is not silently erased. Probability
metrics are reported without inventing a new quality threshold.

`native_geometry_serving_shadow_h20_20261023.sh` enables the frozen
`geometry_v1` policy in this diagnostic. Seven CPU tests cover true argmax,
probability shift invariance, changed distributions, accepted-index padding,
prefill-probe isolation and verification of both actually committed halves.
The worker patch compiles and is idempotent against the frozen snapshot.

## Completed H20 diagnostic

The expanded gate completed 125 mixed forwards / 4,183 logit rows and
1,011 verify request-steps. Every native sampler probe agrees with the
actual committed prefill/verify result; baseline self-repeats and native
serial/PDL logits are bitwise identical throughout.

The corrected result is **33 raw greedy-argmax changes and 11 actual verify
token-sequence disagreements** against same-input native FA3, with no
prefill-token changes. All 11 disagree within the common emitted prefix,
not merely in acceptance length. Thus they are not unused verify positions
or the earlier top-k tie-counting error. The serving-output gate remains
open and exits with status 3.

Weighted mean reference-to-candidate KL is 0.000574 nats; maximum is 0.015819.
Mean total variation is 0.00804; maximum is 0.07620. The original raw-logit
metric still exceeds 0.02 on 118 forwards, with maximum 0.14707. These
measurements do not establish that the changed greedy outputs are harmless.
The next accuracy step is FP32 attention adjudication through the full
target forward, rather than lowering a threshold or blaming PDL ordering.

Sources, all per-step sampling traces, actual output sequences and summary:
`../../pd_colocation/results_20261023/native_geometry_serving_shadow/`.
V2 has not yet passed this full serving-output gate.

## H200 FP32 adjudication and original-row control

The independent reference now replaces attention at every one of the 28
target layers with FP32 QK, softmax and PV, using the same actual paged KV.
TF32 is disabled. Attention outputs are rounded to BF16 and GEMMs remain
unchanged: this is not a whole-model FP32 or re-encoded-history reference.
Every disputed emitted-token step and every sixteenth mixed forward is
adjudicated. Two reference forwards must agree exactly in logits and hidden
states; a subsequent native forward restores KV before production sampling
and must reproduce both native outputs exactly. These checks all pass.

The first completed H200 run has 125 mixed forwards / 4,179 logit rows,
33 raw argmax changes and eight actual verify emission disagreements, with
no prefill changes. FP32 adjudication covers 15 forwards / 517 rows; it
agrees with FA3 on five of the eight disputed decisions and native Q32 on
three. Mean KL from the FP32 reference is 0.000637 for FA3 and 0.000727 for
native. Neither kernel is uniformly more accurate. The output gate remains
open; the terminal status 3 is the measured parity failure, not a crash.

A second run adds the original FA2 row tile (`query_tile=0`, normally Q64)
with the same native SplitKV plan. It completes 125 forwards / 4,179 rows,
with 21 adjudicated forwards / 742 rows. Q32 and the original tile each have
nine actual verify-emission changes against FA3; their logits are bitwise
equal on only two forwards and have 36 argmax differences from each other.
For Q32's nine disputed verify decisions the reference agrees with FA3 on
four, Q32 on four, and neither on one. Mean reference-to-candidate KL over
this selected sample is 0.000621 for FA3, 0.000658 for Q32 and 0.000571 for
the original tile. Q32 changes arithmetic, rather than simply rescheduling
bitwise-identical FA2. Different runtime batch composition prevents treating
the eight-versus-nine counts across runs as an accuracy trend.

All actual sampler probes, native serial/PDL comparisons, reference repeats,
and native-state restorations pass in both runs. The original-tile control
does not close the serving-output gate. The next isolated kernel experiment
keeps BF16 Q/K/V but retains FA2 SplitKV partial outputs and merging in FP32,
with one final BF16 conversion. This tests an identified rounding site;
it is not yet established as the cause or a performance improvement.

Complete logs, sources, per-step decisions and summaries are archived in
`../../pd_colocation/results_20261023/fp32_adjudication_h200/` and
`../../pd_colocation/results_20261023/fp32_original_row_h200/`.

## FP32-partial serving ablation

The FP32-partial candidate completes all 16 requests and 125 mixed target
forwards / 4,163 logit rows on H200. Each forward includes a same-input
BF16-partial native control, two FA3 references and a matched FP32-partial
native serial control. All repeats, sampler probes, serial/PDL comparisons
and native-state restorations pass. There are **14 verify emission changes
against FA3, versus 11 for the BF16-partial control**, with no prefill change.
The terminal status is therefore still 3, not a serving-output pass.

FP32 attention adjudicates 23 forwards / 764 logit rows, selected by either
native variant's emission disagreement plus periodic controls. On exactly
this common sample:

| Candidate | Mean KL from FP32 reference | Verify emissions differing from FP32 |
|---|---:|---:|
| Native FA3 | 0.0006266 | 6 |
| Native FA2 BF16 partials | 0.0006191 | 11 |
| Native FA2 FP32 partials | 0.0005948 | 8 |

FP32 partials improve both metrics relative to the matched BF16-partial
path, but FA3 still has fewer changed emitted decisions despite its higher
mean KL. For the 14 FA3-versus-FP32-partial disputes, the independent
reference agrees with FA3 eight times and native six times. This supports
the precision change, not universal accuracy superiority or harmlessness
of all changed outputs. The sample is disagreement-selected, not a held-out
quality benchmark. Attention microbenchmarks retain their separate scope.

Sources, launch flags, all logits/decision summaries, completed outputs and
logs: `../../pd_colocation/results_20261023/fp32_partial_serving_h200/`.

## Direct BF16 merge-store validation

The final-conversion fusion passes all four native tests on H200 as well
as H20. A new 125-forward / 4,163-logit-row target shadow uses the previous
FP32-partial **separate-conversion** implementation as its serial control,
and the direct BF16 merge-store operator as the production PDL candidate.
All target logits are bitwise equal between those two implementations;
all prefill/verify production sampling probes also agree. The optimization
therefore does not introduce another arithmetic change on this gate.

Comparison against FA3 still yields 14 verify emission differences and no
prefill differences. Full-FP32 attention adjudicates 19 forwards / 620 rows,
agreeing with FA3 seven times and native seven times on those disputes.
Mean KL is 0.000614 from FP32 to FA3 versus 0.000572 to native. All FP32
repeats and native-state restorations pass. The exact-emission gate remains
open, with terminal status 3; the bitwise conversion-fusion result is not
mislabelled as equality to FA3.

Sources, raw reports, launch flags and explicit separate-conversion
equivalence audit: `../../pd_colocation/results_20261023/direct_output_shadow_h200/`.
H200 build/tests: `../../pd_colocation/results_20261023/native_direct_output_h200_gates/`.

## Packed-GQA v3 serving check

The corrected split estimator also completes 125 mixed forwards / 4,187
logit rows on H200. The direct and separate-conversion v3 implementations
remain bitwise identical at full-target logits, with all production sampler
probes agreeing. Actual v3 plan activation is present in the worker log.
The FA3 comparison has 16 verify emission differences and no prefill
differences. Independent FP32 attention adjudicates 22 forwards / 763 rows,
agreeing with FA3 on seven disputed decisions and native on nine; mean KL
is 0.000646 versus 0.000617. Repeats and native-state restoration all pass.

This remains an exact-emission parity failure (status 3), not a crash or
PDL race. Counts across independent runs must not be interpreted as an
accuracy trend because batch composition changes. Sources, per-step data
and launch flags: `../../pd_colocation/results_20261023/geometry_v3_shadow_h200/`.

## Downstream task-quality gate: PASS (2026-10-23, Claude, picking up from the Codex session)

The criterion the user chose: FP32 attention checks (passed on every captured record) plus a downstream task-quality
comparison, not identical greedy tokens. Run on the H200 (`34c7bd503d461dc1`) with the frozen tree and the exact
kernel environment of the audited v3 serving run (`geometry_v3`, FP32 partials, direct BF16 output); only
`SGLANG_NATIVE_PDL` differs between arms. `python quality_gate_20261023.py <out>`; raw summary in
`../../pd_colocation/results_20261023/quality_gate_h200/summary.txt`.

Arms: FA3, native, FA3 again (the second FA3 run is the noise floor: batched greedy decoding is not bitwise
reproducible). Tasks: all 1,319 GSM8K test questions (8-shot, concurrency 128, so prefills share mixed-chunk steps
with decodes) and 200 passkey retrievals at ~32K / ~60K tokens (concurrency 16, so deep-prefix chunks share steps
with other requests' decode rows -- the shape the native path serves). Pass rule, fixed before running: native within
|FA3a - FA3b| + a binomial 2-sigma band, native logged NATIVE_PDL_CALL > 0, FA3 arms logged none, zero request errors.

| mode / task | FA3 a | FA3 b | native | delta vs FA3 mean | band | native calls | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| two / GSM8K | 76.57% | 76.57% | 76.50% | -0.08 pts | 3.30 pts | 606 | PASS |
| two / passkey | 100% | 100% | 100% | 0 | -- | 606 | PASS |
| four / GSM8K | 76.88% | 75.44% | 76.19% | +0.04 pts | 4.76 pts | 393 | PASS |
| four / passkey | 100% | 100% | 100% | 0 | -- | 393 | PASS |

Native task quality is indistinguishable from FA3, in both two-mode and four-mode (trained EAGLE3), with the native
path confirmed to fire. Limit: passkey is at ceiling in every arm, so it can only catch gross corruption of
deep-context attention; GSM8K is the discriminating task. Qwen2.5-7B only.
