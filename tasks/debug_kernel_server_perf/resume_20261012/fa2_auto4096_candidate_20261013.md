# A4 candidate: cap automatic FA2 KV chunks, 2026-10-13

The main task file and its todo structure are unchanged. This is an opt-in
kernel-planner candidate, not a replacement for the selected native-FA2 split.

## Recovered evidence

Claude's `fa2_chunk_floor_20261013` experiment actually completed at 08:04.
All 30 arms were recovered and independently checked for fp32 correctness,
graph-replay correctness, correct unfused controls, and successful final verdicts.
Artifacts: `fa2_chunk_floor_complete_20261013.tar.gz` and the adjacent audit JSON.
Archive SHA256: `5085b7bb4a8bbcaa0c61db6d11a9f9f3b7bc5bbe46cf5cbca7237fd04509c505`.

| Fixture | Auto ms | Fixed 4096 ms |
|---|---:|---:|
| 1 row, q8, KV4096 | 0.0220 | 0.1021 |
| 1 row, q4, KV16384 | 0.0280 | 0.0989 |
| 16 rows, q8, KV4096 | 0.0944 | 0.1629 |
| 8 rows, q4, KV8192 | 0.0727 | 0.1190 |
| 46 rows, q4, KV16384 | 0.8541 | 0.6621 |
| 16 rows, q4, KV131072 | 1.7382 | 1.7441 |

A global minimum of 4096 would hurt small batches and cannot explain the large
batch's improvement: the current automatic rule selects **32768**, not less than
4096, for 46 q4 rows at KV16384. The original fixture log also reports zero merge
rows, consistent with no KV splitting when the selected chunk exceeds KV length.
At 46 q4 rows and KV49152 the rule selects 131072; adding a FA3 prefill row makes
it 262144. These are host-rule outputs, not new GPU performance measurements.

## Implemented candidate

`FUSED_FA2_KV_CHUNK=auto4096` uses `min(existing_auto_chunk, 4096)` in
`/opt/tiger/flashinfer/port07/include/flashinfer/attention/fused_plan.cuh`.
Existing `auto`, fixed chunks, FA3 planning, tile classes and defaults are unchanged.
Small-batch choices remain 256, 512 or 2048 rather than being raised to 4096.

Eight new host cases cover the small fixtures, the large serving verify fixtures,
a real 512-prefill + 46-verify mixture, and restoration of uncapped `auto` behavior.
The complete existing host suite passes:
`bash /opt/tiger/flashinfer/port07/tests/run_host_checks.sh /opt/tiger/flashinfer/include`.

The remote baseline header was checked against SHA256
`bfbcf997f6f6b0b6cafad03231ec0c25b922f05d145570e323e290558ff9834d` before staging.
Only the candidate header was overlaid on a copy of the remote `port07` tree;
other local dirty headers were not deployed. Candidate header SHA256:
`37c20d6b135264013148f77242b8dc98bf9aaca63ed7a4821da9e165872ea64b`.
The running serving tree, `port07`, and its compiled caches are untouched.

## Completed GPU validation

`fa2_auto4096_validate_20261013.sh` is queued as PID `395036` at 13:12:36 +08:00,
behind the middle-context comparison PID `390462`. It uses isolated
`fi_auto4096`, `merged_auto4096`, and `torch_ext_auto4096` paths.
Twelve fixtures compare auto/capped-auto twice each, interleaved: 48 runs, each
with 100 graph timing iterations, 20 warmups, fp32/replay/unfused-control gates.
The parser passes seven local rejection/success tests. All 48 GPU cases completed
successfully at 14:00:28 +08:00, including fp32, graph-replay and unfused-control
checks. Mean fused times over the two interleaved repeats:

| Fixture | Auto ms | Capped-auto ms | Change |
|---|---:|---:|---:|
| 46 q4 rows, KV16384 | 0.85350 | 0.66380 | -22.23% |
| 46 q4 rows, KV49152 | 2.53700 | 1.84010 | -27.47% |
| 512 prefill + 46 q4 rows, KV49152 | 3.64465 | 2.92545 | -19.73% |
| 16 q4 rows, KV131072 | 1.73375 | 1.73255 | -0.07% |

The small single-row q4 fixture regresses 3.46% (0.0289 to 0.0299 ms), despite
retaining the same chunk choice. Other small fixtures are roughly neutral; this
is not evidence of improvement at every shape. Completed logs and audits use the
`fa2_auto4096_complete_20261013` prefix. The candidate remains opt-in.

The best split uses stock nonpersistent FA2 and does not consume this custom
planner. Even a successful candidate does not establish a serving gain or close
A4: the fixed-4096 co-launch still trailed native FA2 on the recovered fixture.

## Serving follow-up

`fa2_auto4096_serving_20261013.sh` started as PID `430924` at 14:41:23 +08:00,
after all KV-sharing kernel cases completed. It compares the actual co-launch
`MTPMIXFUSED` under auto/capped/auto/capped at mix32 chat24/s + agent0.1/s.
The dispatch rule, source tree, graphs and synchronization settings are identical;
this isolates the cap rather than combining it with A3b. Both policies must pass
the existing q4 serving-output gate before timing. Actual fused calls, both roles,
mixed steps, full request completion and graph decode are required per arm.
Six local execution-audit tests pass. Serving results are pending.

The output gates pass the unchanged eight-token long-prompt criterion, but there
is an important difference: AUTO matches all96 tokens of all four long requests;
CAPPED matches96 on three requests and only11 on `l1`. Neither has a traceback
or fallback. Do not treat this as output equivalence or promote the cap solely
from timing. Real-input attention verification / divergence analysis is still
needed. The first AUTO timing arm completed cleanly at14:52:08, and CAPPED_1 was
running at14:59. The running benchmark was not restarted or modified.

All four serving arms finished successfully at15:16:00. Mean-of-repeat changes
versus the uncapped co-launch (not versus the faster native-FA2 split):

| Tenant | Output throughput | Mean TTFT | Mean TPOT | Mean of per-run P99 ITL |
|---|---:|---:|---:|---:|
| chat | +9.37% | -53.63% | -14.59% | -15.29% |
| agent | +15.38% | -42.83% | -15.20% | -14.99% |

Acceptance lengths are essentially stable: chat2.0495 vs2.049; agent3.5925 vs3.5985.
The native split remains faster. This screen covers mix32 at24/s only; it does not
establish gains at all ranges, rates or workload families.

`fa2_auto4096_realcheck_20261013.sh` then ran as PID441589 from15:37:25 to15:40:23.
Both policies passed real-input attention checks with output-buffer snapshot
protection and independent fp32 adjudication of disputed heads. Each logged121
layer0 checks with maximum relative head error0.012052; other-layer failures would
abort execution. Neither produced a failure dump or traceback. No timing claims
use these instrumented runs. The shorter prefix is not unique to the cap: this
run gives AUTO a minimum long prefix11 and CAPPED96, reversing the first gate.
This supports numerical validation within the configured tolerance, not bitwise
generation equivalence or a proven cause for the token divergence.

Four audit-parser tests pass for the verifier runner. Completed serving and
real-input evidence uses the `fa2_auto4096_serving_complete_20261013` and
`fa2_auto4096_realcheck_complete_20261013` prefixes. The default remains unchanged.

## Wider context results

`fa2_auto4096_wide_20261013.sh` completed eight further audited arms at mix64 and
mix128, again auto/capped/auto/capped at chat24/s + agent0.1/s. All arms completed
their requests without fallbacks or tracebacks. Mean-of-repeat changes:

| Range / tenant | Throughput | Mean TTFT | Mean TPOT | Mean of per-run P99 ITL |
|---|---:|---:|---:|---:|
| 64–128K / chat | +17.82% | -28.68% | -14.31% | -17.04% |
| 64–128K / agent | +15.29% | -23.95% | -18.22% | -17.64% |
| 128–256K / chat | +2.66% | -10.59% | -3.56% | +4.62% |
| 128–256K / agent | +2.67% | -5.59% | +1.96% | -2.57% |

The largest range is not a uniform win: chat P99 ITL and agent mean TPOT regress.
Agent acceptance length also changes (mix64:3.144 to2.9965; mix128:3.466 to3.411),
so these whole-run results are not isolated attention timings. The cap remains
opt-in and does not establish the full performance goal. Complete CSV/log/audits
use `fa2_auto4096_wide_complete_20261013`. The next A4 prototype targets the native
nonpersistent FA2 family directly instead of treating the remaining bandwidth gap
as solved by chunk selection.
