# A4 native-FA2 PDL prototype, 2026-10-13

The native nonpersistent FA2 `BatchPrefill` family remains faster than the custom
persistent FA2 executor, even after the capped-auto improvement. This prototype
reuses that native family as the FA3 grid's independent PDL consumer. It is not
promoted, and the main checklist and its todo structure remain unchanged.

## Implemented path

- `fused_primary_run` launches the existing SM-limited FA3 grid without its legacy
  FA2 executor or legacy reduction. The original `fused_run` interface is retained.
- `native_fa2_run` consumes the native wrapper's own 15-field plan and buffers,
  preserving its grid, KV pipeline and split reduction. The bridge currently
  supports bf16, head dimension128, NHD paged caches and causal attention.
- Native FA2 normally waits for its predecessor before loading attention inputs.
  The independent variant moves that wait to the global kernel's exit, including
  early-returning padded CTAs. Inputs must already be ready before FA3 starts;
  FA3 and FA2 write disjoint output rows. Downstream work waits for both roles.
- Native split reductions retain their ordinary dependency handling. Only the
  independent attention kernel changes the wait position. The existing persistent
  FA2 executor already waits at its tail and is not altered in that respect.
- Native CTA synchronization remains block-wide; named barriers remain enabled
  only for traits that explicitly request them. This resolves the old fused-only
  trait assumption when instantiating the native kernel family in the same unit.

The initial primary entry explicitly rejects split-KV FA3 plans. The test uses
`FUSED_NOSPLIT=1` for both the legacy and native comparisons; this restriction is
not silently treated as complete serving support. Deferred FA3 reduction or a
proper fallback is still required before general integration.

## Evidence and safeguards

Local CUDA13.1 / Torch2.9.1 compilation and operator registration pass, as does the
existing host-plan suite. The H200 trial rebuilds with its own Torch2.14.1+cu130;
no local binary is transferred. Nine local parser tests independently validate
timestamp order and reject contradictory overlap labels or incorrect outputs.

Optional `%globaltimer` instrumentation records the primary's first start and last
completion, and the first useful FA2 start/completion. The overlap gate requires a
FA2 block to finish actual work while FA3 remains active, not merely overlapping
kernel launch intervals. Serial and entry-wait controls must not show that event.
Both eager and CUDA-graph execution are checked; timing runs disable the markers.

`native_pdl_prototype_20261013.patch` and
`native_pdl_source_overlay_20261013.tar.gz` preserve the eight-file prototype.
Archive:124,897 bytes, SHA256:
`93f605a398e85b2293903db82f5bfb54c109f4bcaafb43e6cc34aa91688bb9dd`.
Working source: `/dev/shm/native_pdl_work_20261013`.
Local build log: `native_pdl_local_compile_20261013.log`.

## GPU test

`native_pdl_kernel_20261013.sh` started as PID463415 at17:32:25 +08:00 after the
wider capped-auto serving matrix completed. Baseline and candidate source hashes
were verified; it builds in private `fi_native_pdl`, `merged_native_pdl`, and
`torch_ext_native_pdl` paths while preserving the serving sources and caches.

The runner first checks rejection of an unsupported split-FA3 plan, then numerical
and timestamped overlap gates for serial, entry-wait PDL and independent PDL.
It also exercises graph-planned native FA2 buffers, including padded work slots.
Only after those gates pass does it run 48 timing cases: eight fixtures, two repeats,
legacy capped co-launch vs native serial vs native independent PDL. Every timed
case requires fp32 and graph-replay correctness plus a correct unfused control.
GPU results and actual four-mode serving integration remain pending.

## First GPU failure and isolated correction

The first build and split-FA3 rejection gate passed. The serial native probe then
failed: native FA2 rows matched, but all512 FA3 output rows remained nonfinite.
The same fixture also failed in the legacy control, without tracing or graph
capture. No timing result was accepted from that failed version.

The planner used the FA2 chunk length as task3's fallback whenever FA3 splitting
was disabled. With a4096 FA2 cap and65536 FA3 KV length, enumeration created one
FA3 item per tile while device metadata described sixteen chunks. The FA3 path
therefore took the partial-output branch despite an unsplit plan. This is the
specific `FUSED_NOSPLIT` combination exercised by the new prototype, not evidence
that the earlier normal serving runs used that metadata.

The v2 overlay computes independent role chunk lengths: unsplit FA3 gets the full
KV span, regardless of the FA2 chunk. Five host cases and the original host-plan
suite pass. `native_pdl_prototype_v2_20261013.patch` and the matching source archive
retain this correction separately from the failed v1 artifacts. Source archive:
125,350 bytes, SHA256:
`c67fd49a8d93a06a8bd61f7786aa09e718b4d5dce87ee837c9f21afa4fc834cc`.
The new runner uses fresh `fi_native_pdl_v2`, `merged_native_pdl_v2`,
`torch_ext_native_pdl_v2` and result paths; it does not overwrite the first run.
GPU revalidation remains required before any performance claim.

The v2 runner started as PID466439 at17:49:55 +08:00; baseline/candidate hashes
passed and the isolated rebuild began. The v1 failure and both confirming control
logs are retained in `native_pdl_v1_failure_20261013.json`. Neither the original
serving tree nor its source kernel overlay was replaced by the experiment.

## V2 gate result

The metadata correction removes the FA3 failure: serial, entry-wait and independent
PDL all pass the mixed fixture with zero bad rows out of544 and worst fp32 head
relative error0.003248. Both eager and CUDA-graph timestamp gates pass. Serial and
entry-wait controls start FA2 work after FA3 finishes; independent PDL completes a
real FA2 block while FA3 is still active. In the graph probe FA3 spans about1.569 ms,
while the first FA2 block completes about62 microseconds after FA3 starts.

The trace logs are preserved in `native_pdl_v2_trace_gates_20261013.json`. The timing
matrix is running under PID466439; initial FA2-only results match the stock native
control, but the complete matrix and serving integration remain unproven.

The v2 matrix subsequently completed all48 numerical/replay/control-gated cases
at18:02:04 +08:00. Mean CUDA-graph times over two repeats:

| Fixture | Capped legacy ms | Native serial ms | Native PDL ms | Stock two-launch ms |
|---|---:|---:|---:|---:|
| 46 q4 rows, KV49152 | 1.8337 | 1.4924 | 1.4949 | 1.5270 |
| 16 q4 rows, KV131072 | 1.7335 | 1.1647 | 1.1430 | 1.1816 |
| 16 q8 rows, KV32769 | 0.7940 | 0.2987 | 0.2986 | 0.2902 |
| fresh512 + 46 q4 rows, KV49152 | 1.8578 | 1.5379 | 1.5266 | 1.5548 |
| append512 + 46 q4 rows, KV49152 | 2.9310 | 2.6360 | 2.5703 | 2.2822 |
| prefill2048 + 8 verify rows | 5.7946 | 5.8777 | 5.8724 | 3.9500 |

This closes much of the FA2-family gap, not the remaining FA3-dominated gap.
Do not describe the long-prefill result as a win. All cases used the same forced
unsplit-FA3 setting for the custom controls; the stock wrapper retained its native
plan. Complete audits are in `native_pdl_v2_complete_20261013_audit.json`.
