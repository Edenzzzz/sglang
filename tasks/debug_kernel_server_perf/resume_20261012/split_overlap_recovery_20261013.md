# A4 overlap results recovered; A3b serving follow-up

Collected 2026-10-13 after the Claude session exhausted API capacity. The main
task document, its structure and todo items are unchanged. This note records
results and an experiment handoff, not closure of the project or its targets.

## Completed overlap comparison

Trial `095f01a3e1585eea` (`vibecuda`, one H200), runtime tree `/dev/shm/w/up_racefix`.
The existing `split_overlap_20261013.sh` finished at 09:22:32. All six serving
arms completed: 8,640 chat / 36 agent requests per mix32 arm and 3,840 chat / 16
agent requests per mix128 arm, with zero fallbacks and tracebacks. Decode graph
use is present in every log. Reported split-call counts are 35,501–104,001;
`overlapped` equals zero in controls and the split-call count in both overlap
variants. This confirms code-path engagement, not a hardware concurrency trace.

Changes relative to the same sweep's plain split, at chat 24 requests/s and
agent 0.1 sessions/s (negative latency change is better):

| Context / tenant | Margin | Throughput | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|
| 32–64K / chat | 16 | +0.12% | -3.69% | -2.44% |
| 32–64K / chat | 32 | -0.02% | -3.01% | -0.77% |
| 32–64K / agent | 16 | +2.04% | -2.11% | -1.04% |
| 32–64K / agent | 32 | +0.57% | -1.43% | -0.74% |
| 128–256K / chat | 16 | -0.09% | -0.03% | +0.48% |
| 128–256K / chat | 32 | +0.06% | -0.10% | +0.32% |
| 128–256K / agent | 16 | 0.00% | +0.10% | -0.29% |
| 128–256K / agent | 32 | 0.00% | -0.08% | 0.00% |

One run per variant/range is insufficient to promote a marginal mix32 gain
beyond the previously measured noise floor. Mix128 mean metrics are neutral.
Agent mix128 P99 ITL improves 17.4% at margin16, but is based on only 16 requests;
it is not evidence of a reproducible broad latency gain. TPOT and P99 ITL are
separate columns in the CSV; neither is silently substituted for the other.
There is no mix64 overlap run in this experiment.

The margin16 output gate passed with minimum long-request common prefix 52;
the original runner did not separately output-gate margin32. The decision is to
retain plain split as the control, not to claim every overlap variant fully
validated or all kernel work finished.

Artifacts collected locally:
- `split_overlap_complete_20261013.csv`: all 12 per-tenant rows.
- `split_overlap_complete_20261013.log`: original driver/completion log.
- `split_overlap_complete_20261013_audit.json`: per-arm request counts,
  validation records, graph/call counters and raw-result/server-log SHA256s.

Original raw request JSONs and server logs remain under
`/dev/shm/w/split_overlap_20261013`. Reproduce using the checked-in
`split_overlap_20261013.sh`; analysis uses `drivers/serving/eval_curves.py` with
the same chat `[2000,50]` and agent `[10000,80]` SLOs. Those fixed SLOs poorly
discriminate long-context chat and are not the pending unloaded relative SLOs.

## A3b follow-up on the selected serving kernel

`a3b_split_serving_20261013.sh` compares DEFAULT/A3B/DEFAULT/A3B at mix32 and
mix128, both at chat24/s + agent0.1/s, with the same counts and YaRN/window
settings as the overlap comparison. Both arms use `MTPMIX_SPLIT`, margin0,
overlap off, graphs on and the race-fixed tree without a mixed-step barrier.
Each rule must first pass the existing q4 serving-output gate, with split calls
greater than zero. Every timed arm additionally checks complete request counts,
actual split calls, mixed steps, graph decode evidence, and absence of errors.
No numerical or token-output acceptance criterion is changed.

This is a follow-up on the selected non-persistent split, rather than treating
the old co-launch fit's microbenchmark accuracy as proof of serving benefit.
The current default rule also uses a decode-population guard that the A3 path
does not use: with one 512-token prefill at 32K or 128K KV plus 46 q4 verify
rows, the default routes zero rows to FA3 and A3b routes one. The experiment
therefore compares the complete existing rules, not only a work-floor constant.

Remote runner launched as PID `374895`; collect authoritative process status
before doing anything else. Log: `/dev/shm/w/a3b_split_serving_20261013.log`;
results: `/dev/shm/w/a3b_split_serving_20261013`. It retains source hashes and
benchmark environment metadata. Completion requires `A3B_SPLIT_ALL_DONE` and
`A3B_SPLIT_RUN_EXIT status=0`, plus all eight execution audits.

At the 12:12 poll, the runner was live, both output gates had passed with minimum
long-request prefix 52 (short minimum 1 for DEFAULT, 2 for A3B), and mix32
`DEFAULT_1` was running as driver PID `377215`, server PID `377216`. This is not
a timing verdict. Gate logs/JSONs, exact runner and source manifest are preserved
in `a3b_split_gate_20261013.tar.gz`. The runner's embedded execution audit also
passed six local checks covering valid execution, absent split calls, unexpected
overlap, missing graph evidence, fallbacks and partial request completion.

Claude's older queue PID `270264` was observed in state `T` and remains stopped.
Do not resume it concurrently: it contains the superseded co-launch/full-replay
sequence. Its successful full-length stock arm at rate0.005 remains intact under
`/dev/shm/w/tracelab_full_v3_20261013/full_r0.005`.

12:19 observation: mix32 `DEFAULT_1` completed at 12:17:23 with 103,001 split
calls, 3,951 mixed steps, 304 graph-decode log lines and no fallbacks/tracebacks;
`A3B_1` is running. Completed-arm audits and metrics are collected separately in
`a3b_split_progress_20261013.json`; do not treat an unfinished pair as a verdict.

While the GPU run continues, the local relative-SLO analyzer's reference lookup
was corrected: the queued producer writes `mix32_unloaded_chat`, which previously
became the wrong workload key `mix32_unloaded`. Five CLI regression tests cover
all three context ranges and both tenants, underscored workload names, legacy
layouts, absent-reference rejection and unchanged absolute SLOs. Command:
`python test/srt/test_eval_curves.py -v`. This change is local only and does not
alter the running experiment, any acceptance threshold, or existing measurements.

12:27 collection: the first mix32 pair completed without fallbacks or tracebacks
(DEFAULT 103,001 split calls; A3B 118,001). Provisional mean metrics:

| Tenant | Metric | DEFAULT_1 | A3B_1 |
|---|---|---:|---:|
| chat | output tok/s | 1641.7 | 1641.8 |
| chat | TTFT ms | 8949.8 | 6412.9 |
| chat | TPOT ms | 20.95 | 19.53 |
| agent | output tok/s | 210.5 | 223.0 |
| agent | TTFT ms | 8857.8 | 6489.4 |
| agent | TPOT ms | 13.45 | 12.87 |

The throughput comparison is against the existing split, not stock MTP. The
repeat pair and mix128 must finish before selecting a rule. The initial archive
download exceeded the webshell observation timeout; no benchmark was restarted.
The full raw first-pair artifacts were then recovered through Merlin's SSH
execution backend as `a3b_split_mix32_pair1_20261013.tar.gz` (10,676,842 bytes).
SHA256 matches the archive on the trial:
`3d7dbbc947d9b72d41dc9904b2e495ca3e356fdb8942fba483f50fb990ad8a3e`.
