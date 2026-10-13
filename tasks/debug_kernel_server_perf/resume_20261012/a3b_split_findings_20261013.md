# A3b split follow-up evidence, 2026-10-13

This sidecar does not change the main checklist or its large todo items.
The comparison is against the existing plain split control, not stock MTP.
Both rules use native nonpersistent FA2, overlap off, graphs on, q4 EAGLE3,
and the race-fixed serving tree without a mixed-step synchronization barrier.

## Completed 32–64K repeats

The interleaved DEFAULT/A3B/DEFAULT/A3B runs completed at 12:40:10 +08:00.
Every arm completed 8,640 chat requests and 36 agent requests, with actual
split calls, mixed steps, graph-decode evidence, and no fallbacks or tracebacks.
Both rules previously passed the serving-output gate (minimum long prefix 52).

| Tenant | Metric | Mean DEFAULT | Mean A3B | Change |
|---|---|---:|---:|---:|
| chat | output tok/s | 1641.00 | 1641.80 | +0.05% |
| chat | mean TTFT ms | 8827.05 | 6472.20 | -26.68% |
| chat | mean TPOT ms | 20.785 | 19.605 | -5.68% |
| agent | output tok/s | 211.40 | 222.75 | +5.37% |
| agent | mean TTFT ms | 8842.60 | 6584.70 | -25.53% |
| agent | mean TPOT ms | 13.355 | 12.905 | -3.37% |

These are arithmetic means of two runs, not a pooled distribution or confidence
interval. Chat TTFT changed -28.35% and -24.96% in the two respective pairs;
agent TTFT changed -26.74% and -24.33%. Agent throughput changed +5.94% and
+4.80%. The repeat supports a real mix32 improvement, but not the full goal's
30% latency and 10% throughput targets. TPOT is not substituted for every ITL
statistic. A3b changes the full routing rule, including decode-population-guard
behavior, so these gains cannot be attributed solely to the work-floor change.

The original eight-arm run completed successfully at 13:09:42 +08:00; all eight
audits and both final markers were collected. PID `374895` has exited. The
64–128K follow-up also completed, at 13:50:32 +08:00, giving twelve audited runs
across all three ranges. Outer-range metrics are in `a3b_split_progress_20261013.json`;
the completed CSV and driver log use the `a3b_split_complete_20261013` prefix.

The complete second-pair raw results and server logs are recovered locally in
`a3b_split_mix32_pair2_20261013.tar.gz` (9,397,701 bytes), with remote SHA256:
`86bc762d22c9700d7e826bd4500bf8f1e1eb43ccafcf11924c55e54dab4ca064`.
Both arms' request counts and server-log hashes were independently checked.
The first-pair archive and output-gate archive are retained separately.

## First 128–256K pair: mixed result, not a selection

At 12:55 +08:00 six of eight arms were audited; `mix128 DEFAULT_2` was running.
The first long-context pair completed 3,840 chat and 16 agent requests per arm,
with actual split/graph execution and zero fallbacks or tracebacks.

| Tenant | Metric | DEFAULT_1 | A3B_1 |
|---|---|---:|---:|
| chat | output tok/s | 1150.5 | 1203.4 |
| chat | mean TTFT ms | 70605.0 | 49679.3 |
| chat | mean TPOT ms | 30.06 | 30.65 |
| chat | median tok/s/user | 54.91 | 39.93 |
| agent | output tok/s | 79.9 | 89.1 |
| agent | mean TTFT ms | 37343.3 | 35391.7 |
| agent | mean TPOT ms | 23.83 | 21.28 |
| agent | fixed-SLO attainment | 0.4375 | 0.1875 |
| agent | tokens per verify | 3.111 | 3.377 |

The lower mean latencies do not imply uniform improvements: chat mean TPOT and
median interactivity worsen, while agent fixed-SLO attainment decreases. Only
16 agent requests contribute to each arm. Their acceptance lengths also differ;
the aggregate throughput change is not an isolated attention-kernel speedup.
These are the existing absolute SLOs, not the still-pending tenant-alone relative
SLOs. The second pair, recorded next, does not remove these qualifications.

The raw first long-context pair is recovered in
`a3b_split_mix128_pair1_20261013.tar.gz` (4,200,921 bytes), SHA256:
`20fcc012dea951734e46c0f2a4bb7391de7bec4adb1390811aacd3ae0d110ddd`.
Per-request input and output length arrays match across the two rules for both
tenants (276,385 chat output tokens; 25,470 agent output tokens). This rules out
different request lengths as the throughput explanation, not different generated
token identities. Server hashes, full completion and zero client errors match.

## Completed 128–256K repeats

Both pairs reduce chat TTFT (-29.64%, -26.75%) and agent TTFT (-5.23%, -15.79%).
Agent throughput improves +11.51% and +10.30% against the split control. The
arithmetic means of the two runs give the following changes:

| Tenant | Throughput | Mean TTFT | Mean TPOT | Mean of per-run P99 ITL |
|---|---:|---:|---:|---:|
| chat | +6.12% | -28.22% | -0.24% | -33.33% |
| agent | +10.90% | -10.84% | -4.82% | -14.85% |

P99 values here are averaged run-level statistics, not pooled percentiles.
Agent TPOT and P99 ITL improve in pair 1 but slightly regress in pair 2; the
DEFAULT agent acceptance length changes from 3.111 to 3.366 across repeats,
while A3b is 3.377/3.374. A3b's fixed-SLO attainment is 0.1875 in both repeats,
versus DEFAULT's 0.4375/0.3125. Mean chat median interactivity falls 16.89%.
These results do not justify a universal dispatch switch or claim uniform
latency improvement. The control and acceptance thresholds are unchanged.

`a3b_split_midrange_20261013.sh` was queued as PID `390462` at 13:01:34 +08:00
and started after B3 completed at 13:12:04. It uses the existing mix64 settings
(64–128K, 24 agent requests, 8 groups, chat24/s), repeats DEFAULT/A3B twice,
and reruns both output gates. Four predecessor-gate tests and syntax checks pass.

The middle-range run completed both output gates (minimum long prefix52) and all
four timing arms, each with 5,760 chat and 24 agent requests and no runtime errors.
Mean-of-repeat changes against the plain split's default dispatch are:

| Tenant | Throughput | Mean TTFT | Mean TPOT | Mean of per-run P99 ITL |
|---|---:|---:|---:|---:|
| chat | +3.13% | -14.10% | -2.27% | -22.44% |
| agent | +11.15% | -15.76% | -6.87% | -15.54% |

Both tenants' mean median-per-user throughput decreases about4.7%. Agent absolute
SLO attainment is 0.25 vs0.1667 (six vs four of24 requests), not a robust universal
SLO result. Complete middle-range CSV/log/audits use
`a3b_split_midrange_complete_20261013`. This completes the planned routing
comparison, not the full performance goal or the unloaded-reference evaluation.

## Next kernel experiment

`b3_append_resume_20261013.sh` was queued as PID `388565` at 12:54:03 +08:00.
It waited on the same exclusive GPU lock and required successful A3b completion,
all eight execution audits, and unchanged serving-source hashes before starting.
Claude's superseded queue PID `270264` remains stopped; it was not resumed.

The runner preserves Claude's three pending FA3 append fixtures: 4x512 queries
at 64K KV, 8x1024 at 32K KV, and 2x256 at 128K KV. Separate `fi_skip` sources
and extension caches leave the active serving builds untouched. Correct normal
arms must pass the fp32 and graph-replay checks; zero-KV-load diagnostic arms
must fail numerically as expected while their unfused controls remain correct.
Timeouts, missing timings, nonfinite output and unexpected exits reject the run.
Nine local audit-parser checks and shell syntax validation passed.

This is only the KV-load-removal ceiling diagnostic on the custom fused kernel,
not a multicast implementation or a valid serving speedup. It completed all six
audited cases at 13:12:04 +08:00, with zero-KV-load latency reductions of 19–22%.
The output explicitly records `multicast_implemented: false`; B3 remains open.
Remote results are under `/dev/shm/w/b3_append_resume_20261013`, with the adjacent
`.log` file. Detailed findings are in `b3_append_findings_20261013.md`.
