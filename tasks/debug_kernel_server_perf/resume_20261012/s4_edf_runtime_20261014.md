# EDF runtime integration for the current serving tree

The migrated `up_s4_trace_v2/python` tree contains `edf_scheduler.py`, but a
call-site audit finds no callers for deadline assignment, model-based chunk
control or acceptance observation. The benchmark driver sets EDF environment
variables and priority CLI flags; those alone do not activate the controller.
No EDF policy comparison has been run on this tree. The ongoing non-EDF trace
collection is unaffected.

The opt-in port restores the missing runtime hooks:

- Load the step predictor during scheduler policy initialization.
- Assign TTFT deadlines on queue entry and retain urgent retracted-decode deadlines.
- Limit prefill chunks using current decode KV, actual cached prefix, per-request
  acceptance EWMA and the existing SLO controller; preserve page alignment.
- Observe newly committed tokens after both ordinary and mixed verify results,
  updating the request's last-step timestamp and acceptance estimate.
- Emit queue, chunk-decision and token-observation counters so a model load is
  not mistaken for actual policy execution.

New request objects also lack the old `last_tic` attribute. The deadline helper
now uses recorded EDF arrival/completion timestamps first, with a compatible
fallback. Default EDF-disabled behavior is retained. Changes target a private
`up_edf_runtime` tree; the running trace tree and main TODOs are not modified.

Nine CPU tests execute the actual extracted scheduler methods and real EDF
controller, covering disabled behavior, predictor loading, current-schema
requests, retraction, page-aligned chunk caps, overdue decode and post-commit
EWMA updates. The correctness driver additionally enables the same priority
CLI flags as the EDF benchmark; its off/on flag check passes.

`s4_edf_runtime_gate_20261014.sh` waits for the specific trace-matrix process
and its successful completion marker, takes the GPU lock, verifies source
hashes and fits only a complete matched trace matrix. It then runs five fresh
output gates: stock, unfused, split, unfused+EDFR and split+EDFR. EDF gates
require positive queue/chunk/observation counters, loaded models, graph decode
and no errors/fallbacks. A later policy-performance comparison is still required.

The linear model uses the full traced prefix domain rather than the old
4K-prefix MLP projection. Cumulative server acceptance is not substituted for
request EWMA: exported step-cost models omit the acceptance coefficient, while
the controller still uses request EWMA to convert per-token SLO to step budget.

Patch: `s4_edf_runtime_20261014.patch`. Source/test/driver/fitter overlay:
`edf_runtime_overlay_20261014.tar.gz`,83,693 bytes, SHA256
`c2edd14331f3add2ac6f0b7bb69d718898ebc4dc932100ee78102db59c6c7b17`.
GPU validation is pending; no EDF benefit is claimed.
Baseline source hashes and all nine CPU tests passed again on the trial.
PID690685 was verified live at05:18:35, waiting for trace PID682264; it does not
touch the GPU or runtime trees until that predecessor completes successfully.

All five GPU/serving gates completed successfully at06:42:41. Baseline arms
have no EDF counters; each EDFR arm records positive queue, chunk and token
observation counters, with the final logged observation count4000. Unfused/split
EDFR long-prompt common-prefix minima are39/11 tokens under the existing gate.
This verifies active integration and the output gate, not predictor accuracy
or an EDF latency win. The initial global linear fit is superseded by the
mode-aware candidate documented in `s4_edf_piecewise_20261014.md`.
