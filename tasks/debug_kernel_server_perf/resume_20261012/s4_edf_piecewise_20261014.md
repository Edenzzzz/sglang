# Mode-aware EDF step-model refit and serving evaluation

The first fit shares coefficients between graph decode and prefill-containing
steps, and minimizes squared error in milliseconds. Its48.28%/39.95% held-out
MAPE motivates a small controlled refit, not an assumption that the split is
automatically easier to predict. A six-variant screen separates mode handling
from loss weighting; all variants retain nonnegative physical work terms.

The selected candidate fits decode and prefill-containing steps separately,
weighting each training residual by the inverse observed step duration. The
runtime selects the corresponding coefficients from the known prefill-token
count. It preserves the existing predictor interface and rejects malformed or
nonphysical mode coefficients. The base `coef` field represents decode-only
cost; prefill inference requires the mode-aware reader.

| Model / method | Overall MAPE | DECODE MAPE | EXTEND MAPE |
|---|---:|---:|---:|
| Shared coefficients, unfused | 48.28% | 52.49% | 13.71% |
| Shared coefficients, split | 39.95% | 38.64% | 50.40% |
| Mode-aware relative fit, unfused | 16.30% | 15.85% | 20.00% |
| Mode-aware relative fit, split | 13.89% | 14.58% | 8.33% |

These are chronological second-half errors within each of six runs per
method. The relative-error objective trades worse unfused EXTEND error for
better typical-step accuracy; it is not a uniform win for that method.
Trace populations differ between schedules. The same holdouts were inspected
during model selection, so these are exploratory results, not independent
post-selection validation or proof of a causal modeling advantage.

Leaving out entire context ranges remains difficult: unfused errors are
33.96%/41.55%/21.51% when holding out128/32/64 respectively; split errors are
26.96%/41.08%/20.09%. Do not claim generalization to unseen context ranges or
other GPUs. All evaluated serving ranges are represented in the final refit.

Reproduction:

```
.venv/bin/python drivers/serving/fit_step_matrix.py --root /dev/shm/s4_trace_complete_20261014 --out /dev/shm/s4_piecewise_local_models_20261014 --strategy piecewise-relative
```

The complete raw checkpoint is `s4_trace_checkpoint_12_20261014.tar.gz`.
`s4_linear_report_20261014.json`, `s4_piecewise_report_20261014.json` and
`s4_piecewise_refit_screen_20261014.json` preserve the fits, cross-context tests
and screen. Fourteen fitter tests and eleven runtime tests pass; the runtime
tests also pass on the trial. The source overlay is80,497 bytes, SHA256
`df5ff8d137bf36f738df847db8f1b992d07f71de0b5b76219a8de61dfef23665`.

## Runtime gates and timed serving

The mode-aware gates completed at06:55:20, with positive EDF queue/chunk/token
observation counters and no errors/fallbacks. Unfused/split long-prompt output
prefix minima are18/11 tokens; the inherited numerical-output criterion does
not imply bitwise generation equality. The unchanged EDF-disabled stock
control is reused from the immediately preceding successful runtime gate.

`s4_edf_serving_20261014.sh` started as PID726987 at06:56:01 on the same H200.
There are24 fresh timed arms: unfused/split, each with/without request-EWMA EDF,
at chat16/24 per second across32–64K,64–128K and128–256K, plus agent0.1/s.
Each backend uses its own fitted model. Tracing is disabled for performance;
source/model hashes, execution counters, complete requests and zero errors
are required. This first sweep is a screen, not repeated-noise-floor evidence.

Controller TTFT is length-scaled from a2000ms base and its per-token ITL budget
is50ms. The reporting CSV retains the existing fixed chat2000/50 and
agent10000/80 thresholds; these are not identical SLO definitions. No EDF
throughput, latency or SLO improvement is claimed before the serving results.

All seven runtime/piecewise raw gate logs, outputs, fitted models, reports and
source/model hashes are preserved in `s4_edf_gates_complete_20261014.tar.gz`,
234,355 bytes, SHA256
`e664abe21355f708bf0cc2308cc7cfd09eae50279ba5db1d7efca19598915d63`.
The archive and each of its seven server-log hashes were rechecked locally.

The first timed sweep subsequently stopped on its first EDF arm at07:05:58,
before any EDF performance result could be accepted. The adapter passed a
prefill-only cap into an API that subtracts decode reservations from a total
budget, producing a negative prefill extent. The root correction, regression
tests and fresh V2 run are recorded in `s4_edf_budgetfix_20261014.md`; the
original failed run is preserved. The fitted model weights are unchanged.
