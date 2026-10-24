# EDF step-model trace collection

The completed paired-serving and unpacked-query experiments do not displace
the plain split. The next required stage compares model predictability of
that serving kernel against the unfused four-mode schedule.

Claude's latest handoff and last100 session messages were reviewed after the
user resumed Codex. Claude added per-worker acceptance EWMA to the CPU router
and changed its pending S4 script to use the split. It did not launch GPU
work. The old S4 script points at a missing remote fitter and does not fail
on an individual serving arm's failure; it is not launched as-is.

`s4_trace_matrix_20261014.sh` started as PID676094 at04:43:56 +08:00. It holds
`a2_kernel.lock` on the existing H200. Fresh output gates compare stock MTP,
unfused four-mode and split four-mode first. Trace arms then alternate order:

- 32–64K: chat8/16/24 per second.
- 64–128K: chat8/16 per second.
- 128–256K: chat12 per second.

All use agent0.1/s, EAGLE3 q4, graph decode and the race-fixed serving tree
without a per-step barrier. There are12 traced arms. Each must complete every
request, report mixed steps, have no errors/fallbacks, contain at least100
asynchronous-event trace rows, and execute the expected split/non-split path.
Source hashes and effective environment are captured before the gates.

This is the data-collection phase, not a predictor or EDF improvement claim.
The next analysis must preserve run boundaries, measure held-out error by
mode/context, compare acceptance-aware and acceptance-free fits, and validate
the resulting scheduling policy in fresh serving arms. Main TODOs are unchanged
by this worker; Claude's independent documentation/commits are left intact.

The fresh gates passed at04:47:33: minimum long-prompt common prefixes are18
tokens for unfused four-mode and11 for the split, under the existing8-token
criterion. The short-prompt minimum is0 and1 respectively; these gates are not
bitwise generation-equivalence claims. The first trace arm then started.

`drivers/serving/fit_step_matrix.py` adds the held-out analysis, with12 CPU
tests in `test/srt/test_fit_step_matrix.py`. It keeps launch intervals inside
individual runs, excludes idle gaps, holds out each run's second half, and
separately leaves out complete context ranges. Fits use nonnegative physical
coefficients and report error by mode, context and run for launch intervals
and asynchronous GPU durations. A complete, matched two-method matrix is
required before producing deployment candidates.

The trace's `acc` field is a cumulative pre-step server acceptance ratio,
not the EDF controller's per-request EWMA. Acceptance-feature fits are therefore
reported only as an ablation; initial deployment candidates omit `c_acc` rather
than silently substituting a differently defined runtime feature. EDF still
uses its existing request-EWMA tokens-per-step estimate for per-token budgets.
No held-out performance claim is available until the trace matrix finishes.

## V1 rejected before fitting; trace schema fixed

The first unfused arm completed all requests with2451 mixed steps, but the
instrumentation gate stopped the matrix at04:54:43. All25,711 STEPTRACE records
reported zero prefill rows, including2594 EXTEND records. This is not usable
model-training data. No fit or EDF performance claim was produced from it.

Root cause: the current request schema uses `extend_len`; the trace hook still
looked up the absent `extend_input_len` and defaulted silently to zero. The
fix uses the actual per-row `batch.extend_lens`, validating its shape first.
It changes only tracing, in a private Python tree; attention and scheduling
behavior are unchanged. Five host tests execute the real extracted trace
method, covering current request fields, stale request attributes, mixed
verify context, decode rows and missing metadata.

The preserved patch is `s4_steptrace_features_20261014.patch`. The replacement
runner `s4_trace_matrix_v2_20261014.sh` repeats fresh correctness gates with
tracing enabled and requires positive mixed prefill/decode feature rows before
starting its12-arm matrix. The fitter also now rejects all-zero prefill traces.
The original stopped run remains untouched, not overwritten or resumed.

`s4_trace_v2_overlay_20261014.tar.gz` is67,574 bytes, SHA256
`8a3929193f7b305d808e075c5447e5c0d1c74aa5a5415e3289b98e4dbadf0fa6`.
V2 started as PID682264 at05:00:16 after both baseline/candidate scheduler
hash checks passed. The invalid V1 trace is preserved as
`s4_v1_invalid_steptrace_20261014.log.gz`, SHA256
`0f225c0e65300bd8ab4aaa97d1706e76a18ab7ee8a07d74a8ba345b974322599`.

V2 passed all fresh gates at05:03:54. Stock/unfused/split feature checks cover
888/857/844 trace records respectively and now include the expected prefill
and mixed-row features. Unfused/split minimum long-prompt output prefixes are
11/52 tokens. The first corrected matrix arm started immediately afterward;
PID682264 and its GPU server were confirmed live at05:04:52.

The first corrected arm, mix32_r8_MTPMIX_LPT, completed at05:11:04 with2451
mixed steps and25,850 trace records. Its source/server hashes and feature
coverage passed. A local parser/fitter smoke check accepts the real records;
25,749 within-run launch intervals remain after idle-gap filtering. This is
only a one-arm sanity check, not the completed cross-method/context comparison.

The subsequent EDF call-site audit and opt-in runtime port are documented in
`s4_edf_runtime_20261014.md`. They do not modify the ongoing trace collection.

## Completed matrix

All12 corrected arms finished successfully at06:36:30. Each passed its request,
execution, graph and trace-feature audits. There are59,129 usable within-run
launch intervals for unfused four-mode and75,026 for the split. Raw trace and
execution-audit checkpoint hashes were independently verified locally.

`s4_trace_checkpoint_12_20261014.tar.gz` is2,561,937 bytes, SHA256
`41a2be7ad6a9398d06898b2c44de161e78caea1819fbfdfb62f440a73f7d00fd`.
The earlier six-arm checkpoint is1,886,102 bytes, SHA256
`d405e5b1d0c49dc5ed58068b79f30eed79780433329ab930f3718061bf88b82e`.

Initial shared-coefficient NNLS models have held-out launch-time MAPE48.28%
unfused and39.95% split. They are not accurate enough to label this stage a
successful predictor result. The refit and its limitations are recorded in
`s4_edf_piecewise_20261014.md`.
