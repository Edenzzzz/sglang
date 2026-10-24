# A: single-split FA3 PDL serving comparison

This continues the existing kernel item. No EDF, main-document restructuring or TODO changes.
The preceding failure investigation and controls are in `stock_fa3_dispatch_20261014.md`.

## Correctness result

FA3 role replacement localized the49152-token output mismatch to the primary. Native FA2 is
bit-identical to its public wrapper on24 changing-input kernel comparisons. The old primary
planner selects two splits on most sampled serving calls; a single-split generic primary is
bit-identical to public FA3 on the tested q512/KV49152 fixture.

The new opt-in `Split=false` primary specializes only host-proven single-split plans; other
plans keep the existing split kernel and deferred combine. CPU compilation, three GPU unit
tests and strict phase/projection/public-role replays pass. The separate one-split policy
override is tested with both the old generic kernel and this specialization, so the two
changes are not conflated.

All five132-SM serving gates finish00:06:29. The generic one-split arm agrees on96 long
tokens; specialized with same-input reference agrees on at least91, and specialized without
reference synchronization agrees on96. The plain repeat agrees on at least51. All12 requests
complete per arm with no server errors, correct backend counters and graphs. Short-prompt
generation is not fully bit-identical; this is the unchanged inherited prefix gate plus
separate same-input attention/fp32 checks, not a full model-equivalence claim.

An initial runner stop was a test-label bug (`UNSPLIT` accidentally matched the plain-split
counter rule). The corrected audit selects fused first and additionally requires positive
mixed execution and all12 requests. Four CPU tests pass; preserved completed outputs are
re-audited before continuing. Numerical and generation thresholds are unchanged.

## Matched timed pilot

PID963008, `stock_fa3_single_serving_20261015.sh`, first runs fresh generic/specialized112-SM
same-input serving gates. Only if both pass does it run eight timed arms:
MTP / unfused four-mode / split control / generic serial112 / generic PDL112 /
specialized PDL112 / specialized PDL132 / repeated split control.

All kernel arms use guard0, which actually routes long mixed-prefill rows to FA3. Frozen
workload:48 agent programs/96 rounds,24 active programs,32–64K contexts;2048 natural
ShareGPT requests at8 requests/s. EAGLE3 q4, graphs, YaRN and admission policy are fixed.
No reference/logit/profiler synchronization is enabled during timing. All arms use the same
private `up_stock_fa3_unsplit/python` tree; only the declared kernels, phase and SM budget vary.

Audits require exact request/output-length fingerprints and program identities, peak24,
positive long-context mixed routing, actual logged SM budget, intended stock phase and
single-split execution, no errors/fallbacks/EDF. Compiled kernel hashes and specialized-build
flags are recorded before running. Compare executor variants within four-mode arms; do not
credit the MTP-to-four-mode scheduling change to PDL. No result is promoted before the repeated
controls and both tenants' mean/p99 ITL, TTFT, throughput and acceptance are inspected.

This remains a finite gap-free length/session stress test, not faithful tool-delay replay,
original agent text, sustained capacity or the full context/concurrency/request-rate sweep.
No per-dataset SLO result is claimed without matching unloaded references.

Root/log: `/dev/shm/w/stock_fa3_single_serving_20261015[.log]`; no successor is queued.
Runner/auditor overlay4789 bytes, SHA256
`7d1f10e64ec0da11a12498d85bc51810494e867e17dbc20886a7bbb549b41e6d`.

Both fresh112-SM gates pass00:15:30 with minimum long prefix96; the MTP timed control
then starts. Four CPU auditor tests reject wrong actual SM budget, multiple splits and
missing policy evidence while accepting correctly declared execution.

Complete132-SM/units/phase evidence is local: `stock_fa3_unsplit_complete_20261014.tar.gz`,
185854 bytes,37 verified hashes, SHA256
`8fd77ef0438a0989f3638f5b40d08200505a13427c0bdc7d3a0112bbad9b341e`.
Five-arm independent summary: `stock_fa3_unsplit_complete_audit_20261014.json`.

## Complete: no promotion over the split control

All eight arms finish01:04:05. Each completes96 agent rounds and2048 chat requests, passes
the frozen-cohort, program-concurrency, backend, graph, actual-SM and single-split audits,
and has no errors/fallbacks/EDF. Relative to the two plain-split controls' mean:

| Candidate | Tenant | Output tok/s | Mean TTFT | Mean ITL | Mean of P99 ITL |
|---|---|---:|---:|---:|---:|
| Generic serial112 | Agent | -4.05% | +6.08% | -0.63% | -3.24% |
| Generic serial112 | Chat | -3.12% | +8.79% | +2.97% | +5.41% |
| Generic PDL112 | Agent | -1.57% | +3.82% | -1.47% | -3.83% |
| Generic PDL112 | Chat | -0.88% | +5.38% | +0.70% | +5.97% |
| Specialized PDL112 | Agent | -1.62% | +3.77% | -4.33% | -7.95% |
| Specialized PDL112 | Chat | -0.75% | +5.15% | +1.19% | +3.14% |
| Specialized PDL132 | Agent | -1.23% | -0.61% | +4.03% | +15.19% |
| Specialized PDL132 | Chat | -0.44% | +0.91% | +0.32% | -0.37% |

Candidate P99 changes compare one candidate run with the mean of two baseline percentiles;
these are not pooled percentiles. Agent strict acceptance differs: split controls0.829/0.877,
generic PDL1120.879, specialized PDL1120.881 and specialized PDL1320.829. The apparent agent
ITL changes therefore are not isolated proof of attention-kernel speedup. Original serial
versus PDL also changes wrapper entry; do not label its difference pure overlap benefit.

Specialized PDL132 versus the fresh MTP control improves agent throughput33.46%, TTFT32.49%
and mean ITL32.49%; chat improves31.70%,42.18% and25.60%, respectively. Chat mean ITL still
misses the30% target, and this finite-cohort point does not establish the full workload goal.
Keep plain split as the performance control. The single-split policy resolves the reproduced
correctness failure; neither PDL nor unsplit specialization gives a general gain over that
control here. Complete audits: `stock_fa3_single_serving_complete_audit_20261015.json`.

Compact raw evidence is local: `stock_fa3_single_serving_complete_20261015.tar.gz`,
1584501 bytes,71 verified hashes, SHA256
`34c884b78b2706b76fe64f3aa75c1843506b615800036b42fb84319e8b43167d`.
Independent local comparison confirms exact request keys and requested/actual output lengths
across all eight arms and both tenants. Full generated-text/token-arrival results remain on
the trial with hashes recorded in the compact cohort records.
