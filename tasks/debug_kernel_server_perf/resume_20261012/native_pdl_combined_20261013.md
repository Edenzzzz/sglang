# Dedicated FA3 entry plus compact native-PDL planning

This candidate combines the measured dedicated FA3 entry with the compact
primary-planning API. It does not include either KV-sharing prototype. Its two
FA3 headers are byte-identical to the completed dedicated-entry experiment;
its backend, compact C++ entry and harness match the compact-planning variant.
The combined local CUDA build and all four CPU glue tests pass.

The dedicated-entry kernel experiment improves its four measured fixtures by
9.2–23.0%, but those timings do not establish an end-to-end gain. The preceding
native-PDL serving matrix showed no mean-performance advantage over the stock
two-launch split. This combined candidate must earn its own serving result.

`native_pdl_combined_chain_20261013.sh` is queued as PID 526973. Validation waits
for the bulk-KV experiment PID 506950 to terminate and for exclusive GPU access.
It records that experiment's exit status without requiring success, because its
sources are independent. Successful dedicated-entry and compact-planning gates
are required. Validation uses private combined source/Python/cache paths and
requires eager and graph useful-overlap proof, ten split-KV numerical/graph/
unfused-control cases, and the real four-mode attention and token-output gate.

Only after successful validation does the chain launch
`native_pdl_combined_serving_20261013.sh`: fresh SPLIT/NATIVE comparisons at
32–64K, 64–128K and 128–256K, chat 24/s plus agent 0.1/s, EAGLE3 q4 and CUDA graphs.
Each range uses SPLIT_1/NATIVE_1/NATIVE_2/SPLIT_2 to balance arm order. Ordinary
output gates run without attention-check instrumentation before timing. Native
arms require both native-FA2 and compact-primary execution counters; completed
request counts, mixed composition, graph decode and zero errors/fallbacks are
mandatory. Default dispatch remains unchanged in both arms.

The source patch is relative to native-PDL v3 and includes both changes. The
remote overlay applies the two FA3 headers to the separately validated compact
tree and verifies before/after hashes of the headers, C++ entry, backend and
harness. Archive: 38,956 bytes, SHA256
`a2fa9bdd30eae69c20a26a8df3fd6cbec68147439133735e555b7c18f130b47c`.
No default is changed, no serving gain is claimed yet, and the main task
document and large TODOs remain unchanged.

## Validation passed; serving comparison started

Combined validation completed at 21:09:39 +08:00 on 2026-10-13. Both eager and
graph traces show useful FA2 work completing during FA3 execution. All ten
split-KV cases passed. The real four-mode gate exercised compact-primary calls
(last logged 3401), with zero fallbacks/tracebacks, 127 layer0 attention checks
and maximum relative error 0.012052. The existing output gate passes with
minimum long prefix 11; this is not bitwise generation equivalence.

`native_pdl_combined_gate_complete_20261013_audit.json` preserves those checks,
raw-log hashes, overlap evidence and the output comparison. The chain then
started serving PID 546077, initially running its ordinary output gates before
the 12-arm timing matrix.

## Completed serving comparison

All 12 arms completed at 22:52:49 +08:00 on 2026-10-13, with the required
execution counters, complete request counts, CUDA-graph decode, zero fallbacks
and zero tracebacks. Two-repeat mean changes against the contemporaneous
page-1 SPLIT control are:

| Context | Tenant | Output throughput | Mean TTFT | Mean TPOT | P99 ITL |
|---|---|---:|---:|---:|---:|
| 32–64K | chat | -0.02% | +2.28% | +1.27% | -0.32% |
| 32–64K | agent | -1.18% | +2.51% | +0.52% | -1.08% |
| 64–128K | chat | +1.74% | -2.53% | -1.72% | -15.91% |
| 64–128K | agent | +9.65% | -4.98% | +5.54% | -2.31% |
| 128–256K | chat | -0.13% | +0.80% | -0.05% | -0.57% |
| 128–256K | agent | -0.80% | +0.65% | +2.08% | +1.04% |

The 64–128K agent throughput result is confounded by acceptance: SPLIT averages
2.753 tokens per verify versus 3.177 for the candidate. Individual repeats are
2.731/2.774 for SPLIT and 3.240/3.114 for the candidate. It is not an isolated
kernel-speed measurement. At the other ranges throughput is essentially flat
or slightly worse; the requested mean-latency/throughput target is not met.
The candidate is not promoted over the two-launch control.

The audit, complete runner log, CSV, source manifest, environment and output
comparisons use the prefix `native_pdl_combined_serving_complete_20261013`.
