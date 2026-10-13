# Native PDL: split-KV support and serving integration

The v2 numerical/overlap gates and 48-case matrix are complete, but its unsplit
FA3 restriction prevented general serving use. V3 removes that restriction by
planning only the FA3 requests in the primary workspace, launching native FA2
independently, then running the existing FA3 reduction after both executors finish.
No FA2 partials are reduced from an unexecuted legacy plan.

`fused_primary_finish` validates an FA3-only plan before reducing its partials.
The native FA2 family's own split reduction is unchanged. The harness now covers
short/odd FA3 queries on long KV, concurrent FA3/FA2 splits, and multiple FA3 rows
with different query/context lengths, without `FUSED_NOSPLIT`.

## Serving path

The isolated backend adds `SGLANG_FUSED_NATIVE_PDL=1` without changing defaults.
The FA2 wrapper remains planned once per step. FA3-only metadata is also prepared
once; a copy-completion event protects reusable pinned staging. All token gathers
and output allocations occur before the primary launch. Native FA2 follows it
immediately, then deferred FA3 reduction and any necessary output scatters.

Contiguous prefixes, contiguous suffixes and noncontiguous row partitions are
supported. Three CPU glue tests verify the correct output-row mapping and reject
tensor indexing/allocation between the two PDL launches. Full-batch reference
checking, output snapshots and independent fp32 adjudication are retained for
instrumented serving validation. Native execution has its own counter in addition
to the existing fused counter; an old path cannot satisfy the native gate silently.

The bridge is currently bf16, head128, causal attention, without softcap or the
other features excluded by the existing split admission guards. This is not a
claim of universal backend compatibility or completed performance optimization.

## Artifacts and run

`native_pdl_prototype_v3_20261013.patch` and
`native_pdl_source_overlay_v3_20261013.tar.gz` contain the kernel/serving changes.
Archive:168,485 bytes, SHA256:
`8b977ccac4bb8264eeb1caffeac23f5bb011b2ec77986630c9f9ed309645c7b6`.
The backend baseline was recovered from the live trial and matched SHA256
`6a5ae97cc6ee625f1cb91f942dc089e9293addbee83e47f62a46ca28d724e096`.
Local CUDA compilation and the three CPU glue tests pass.

`native_pdl_v3_validate_20261013.sh` started as PID482567 at18:36:34 +08:00. It
uses private `fi_native_pdl_v3`, `up_native_pdl_v3` and extension-cache paths.
Ten eager/graph kernel cases must actually contain split FA3 work and pass fp32
and unfused-control checks before the real EAGLE3 four-mode serving gate starts.
The serving gate requires native calls, real-input attention checks with a0.02
relative tolerance, unchanged output snapshots, and the existing output criterion.

The original serving trees, main task structure and big todo items remain intact.
No new serving gain was claimed from the local build or queued validation alone.

## Verified outcome

All ten split-KV cases passed eager numerical checks, CUDA-graph replay checks and
their unfused controls. The EAGLE3 four-mode serving gate then completed at18:42:04
with native calls actually exercised, zero fallbacks and zero tracebacks. It logged
127 layer0 attention checks, maximum relative error0.012052; any failed other-layer
fp32 adjudication would abort. The unchanged token-output criterion passes with
minimum long prefix11 (three long requests match all96 tokens). This is numerical
validation, not a claim of bitwise generation equivalence.

`native_pdl_v3_complete_20261013_audit.json` preserves the kernel audits, native
serving counters, verification summary and output comparison.

`native_pdl_serving_20261013.sh` started as PID487089 at18:49:07 +08:00. It first
reruns output gates without attention-verification instrumentation, then compares
the native PDL path against the faster stock-FA3/native-FA2 two-launch split across
32–64K,64–128K and128–256K contexts. Each range runs SPLIT/NATIVE/SPLIT/NATIVE at
chat24/s plus agent0.1/s. Dispatch rules and workload parameters remain identical.
Eight local execution-audit tests pass.

## Three-range serving result

The matrix completed at 20:31:46 +08:00 on 2026-10-13. All 12 arms completed
their expected request counts with actual split/native execution, CUDA-graph
decode, zero fallbacks and zero tracebacks. The ordinary output gates passed
their unchanged criterion: minimum long prefix 52 for SPLIT and 11 for NATIVE.
These are not bitwise-equivalent generation results.

Changes below compare the two-repeat mean for NATIVE against the two-repeat
mean for SPLIT. Negative latency changes are better; throughput is output tok/s.

| Context | Tenant | Throughput | Mean TTFT | Mean TPOT | P99 ITL |
|---|---|---:|---:|---:|---:|
| 32–64K | chat | 0.00% | +9.84% | +3.77% | +0.79% |
| 32–64K | agent | -2.64% | +9.36% | +1.72% | +0.27% |
| 64–128K | chat | +0.23% | +0.59% | -0.58% | -10.97% |
| 64–128K | agent | -0.64% | -0.64% | +8.74% | +1.89% |
| 128–256K | chat | -0.36% | +1.50% | +0.30% | -5.44% |
| 128–256K | agent | -0.50% | +1.21% | -0.72% | -1.73% |

The native PDL path does not improve mean serving performance over the best
two-launch control and is not promoted. Its lower chat P99 ITL at the wider
ranges does not offset the absent throughput/mean-latency gain. This comparison
does not meet the requested performance target against this control.

Acceptance also changes at 64–128K: mean agent tokens per verify falls from
3.253 to 3.102. Therefore the +8.74% TPOT change is not an isolated attention-
kernel latency measurement; aggregate acceptance and per-request TPOT averages
cannot be divided to infer a precise kernel-only contribution. At the other
ranges, acceptance is approximately unchanged. Absolute-SLO attainment is equal
except for worse chat attainment at 32–64K; unloaded relative-SLO references
remain separate outstanding work.

The completed audit, runner log, CSV, source manifest, environment and both
output-gate reports use the prefix `native_pdl_serving_complete_20261013`.
`native_pdl_acceptance_progress_20261013.json` additionally preserves the raw
acceptance totals for the first two ranges. The dedicated FA3 entry and compact
planning variants are separate experiments, not retroactive changes to this run.
