# Three-model kernel screen and two-model follow-up

The user's selection plan is to screen all three models and carry the two
with the strongest reproducible gains into the full serving/scaling matrix.
The kernel screen now favors **Qwen2.5-7B and Qwen3-8B**. GLM-4.5-Air remains
an explicitly reported negative control, not a missing result. Qwen3's
matched H200 serving screen is running; kernel selection alone does not
establish its end-to-end or SLO gains.

## Comparable scope, not identical shapes

Every kernel comparison replays the same actual Q/K/V tensors, physical KV
indices and strides against both implementations. All models use BF16,
128-dimensional heads, page one, 2,048-token prefill chunks and real trained
speculation. The controls tune FA3 full/dual calls, SplitKV, GQA packing and
SM-grid caps. Native uses v3's packed-GQA geometry policy, FP32 partials and
the fused final BF16 store.

Across models, this is a common **32–64K workload class**, not identical
live batch shapes or a common-cohort causal estimate. Qwen2.5's original
capture seed is 20261014; Qwen3 uses 20261023; GLM uses 20261023 on the
duplicate-normalized trace, which changes the selected sessions/windows.
All runs retain the actual per-request lengths. No workload-weighted average
or cross-model absolute-throughput ranking is inferred from these samples.

| Model | Source serving configuration | Per-rank query/KV heads | Four-token native row tile |
|---|---|---:|---:|
| Qwen2.5-7B | One H20, trained EAGLE3 | 28 / 4 | Q32 |
| Qwen3-8B | One H20, trained AngelSlim EAGLE3 | 32 / 8 | Q16 |
| GLM-4.5-Air | Four H20s at TP4, own NEXTN layer | 24 / 2 | Original Q64, opt-in |

GLM captures only rank zero to avoid concurrent writers. Its kernel timings
measure that rank's attention on one idle H20 after serving stops; they do
not include TP collectives or claim four-GPU model speedups. The Q64 row
class has changed-input/graph/consumer gates on H20 and H200; it is not
silently substituted with a Q32 kernel that cannot cover 48 packed rows.

## Screening evidence

Qwen2.5 has meaningful two-mode and long-context four-mode advantages,
including 1.47× at 32–64K two-mode fresh/small and 1.78× at 128–256K
four-mode cached/small. Its prefill-heavy 32–64K cases retain small
regressions. Qwen3's eight 32–64K captures range from 1.03–1.59× two-mode
and 1.02–1.67× four-mode. These are representative kernel gains, not
end-to-end gains.

GLM completes 16 agent and 480 chat requests in each mode, zero errors,
with actual four-mode tokens per verify **2.936 agent / 2.588 chat**. Seven
eligible captures are observed; four-mode cached/small is **not observed**
and is not filled synthetically. All 140 fresh v3 validation records pass
the FP32 gate, maximum head-relative error 0.003941, and native serial/PDL
outputs are exact.

| GLM capture | Best tested FA3 (ms) | Native v3 (ms) | FA3 / native |
|---|---:|---:|---:|
| Two / fresh small | 0.1089 | 0.0809 | **1.35×** |
| Two / cached small | 0.5447 | 0.5420 | 1.01× |
| Two / fresh large | 0.6019 | 0.6047 | 1.00× |
| Two / cached large | 1.3312 | 1.3326 | 1.00× |
| Four / fresh small | 0.0843 | 0.1335 | **0.63×** |
| Four / fresh large | 0.5340 | 0.5572 | **0.96×** |
| Four / cached large | 1.2623 | 1.2863 | **0.98×** |

All three observed GLM four-mode cases regress, by roughly 2–58%. Its
single sizeable two-mode win is insufficient for the required four-mode
story. The normal 32-packed-query eligibility limit remains unchanged;
the Q64 path is opt-in and is not promoted for GLM serving. These results
do not prove GLM can never benefit on unmeasured shapes.

## Artifacts and next criterion

- All 23 common-band capture points, including regressions and missing
  buckets: `../../pd_colocation/results_20261023/model_screening/three_models_32k.svg`
  with CSV and source-hashed JSON alongside it.
- GLM requests, source freeze, Q64 tests, all controls and numerical checks:
  `../../pd_colocation/results_20261023/glm_captured/`.
- All seven GLM tensor captures are locally SHA-256 verified under
  `/opt/tiger/captured_mixed_inputs_20261023/glm/` (759,313,034 bytes).
- Qwen2.5 serving attribution is in `native_kernel_serving_attribution_20261023.md`.
  The earlier long-context pair clears 10% agent throughput, not the full
  latency/SLO target, and uses the older v2 policy.

Proceed with Qwen2.5/Qwen3 serving confirmation and full workload/rate/SLO
evaluation, retaining GLM's negative screening table. Exact FA3 emitted-token
parity remains open and must not be conflated with passing attention or
serial/PDL correctness checks. The main document and its big TODOs are unchanged.
