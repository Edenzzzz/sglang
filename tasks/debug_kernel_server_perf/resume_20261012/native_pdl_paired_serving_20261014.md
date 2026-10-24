# KV128 round-robin serving comparison

The controlled kernel A/B identified useful work-distribution gains without
TMA, while KV96 and the smaller producer warp did not improve the matching
controls. This run therefore keeps the dedicated native-PDL primary at query
tile128/KV tile128, two stages and 112 CTAs, and enables the validated
round-robin-pair assignment. Hardware clustering and TMA remain disabled.

`native_pdl_paired_serving_20261014.sh` starts as PID 637372 on the existing H200.
It uses the corrected KV128 build in `fi_fa3_kv_geometry_v2` with its private
Python tree and caches. Page size remains1. The main task document and default
serving trees are not modified.

A same-input real-attention/output gate runs first. Ordinary output gates then
precede 12 fresh serving arms across 32–64K, 64–128K and 128–256K. Each range
uses SPLIT_1/NATIVE_1/NATIVE_2/SPLIT_2 at chat24/s plus agent0.1/s, EAGLE3 q4 and
CUDA graphs. Workloads and dispatch admission are unchanged; the control is
the faster stock-FA3/native-FA2 two-launch split. Native and compact execution
counters, complete request counts, graph decode and zero errors/fallbacks are
required. Instrumentation is disabled for timing.

The output root is `/dev/shm/w/native_pdl_paired_serving_20261014`. The inherited
analyzer filename is `native_pdl_combined_serving.csv` within that new root;
it does not overwrite the prior combined experiment. Kernel improvements are
not treated as serving gains until this comparison completes.

## First completed context range

The 32–64K range completed all four interleaved arms. Every arm completed8640
chat and36 agent requests, with zero errors/fallbacks and positive execution
counters. Mean native-versus-split changes over the two repeats:

| Tenant | Output tok/s | Mean TTFT | Mean ITL | Mean TPOT |
|---|---:|---:|---:|---:|
| Chat | -0.06% | +2.32% | +1.19% | +1.22% |
| Agent | -1.06% | +3.91% | +0.45% | +0.69% |

This range does not show a serving improvement. Acceptance length is closely
matched here (chat2.1402/2.1404, agent2.1515/2.1507 split/native). The larger
ranges are still running; no cross-range conclusion is made.
`native_pdl_paired_serving_progress_20261014.json` preserves scalar metrics and
execution audits for the first five completed arms, including mix64 SPLIT_1.

Workload scope: the chat arm uses the benchmark's `random` dataset mode with
ShareGPT source text and controlled lengths, not unmodified ShareGPT length
and output distributions. The sampler truncates/repeats prompts to256–512
tokens and samples48–96 output tokens. It remains a controlled comparison;
it is not evidence that all natural chat distributions meet the goal.

## Completed three-range result

All12 arms completed at04:23:45 +08:00 on2026-10-14. Every arm has the expected
request count, graph decode, positive split/native execution counters, zero
fallbacks and zero tracebacks. Collection independently rechecked server-log
hashes and per-request completion/error counts. Two-repeat mean changes:

| Context | Tenant | Output tok/s | Mean TTFT | Mean ITL | Mean TPOT |
|---|---|---:|---:|---:|---:|
| 32–64K | Chat | -0.06% | +2.32% | +1.19% | +1.22% |
| 32–64K | Agent | -1.06% | +3.91% | +0.45% | +0.69% |
| 64–128K | Chat | -0.07% | +0.56% | +0.15% | +0.27% |
| 64–128K | Agent | -0.35% | +0.81% | -0.15% | +0.54% |
| 128–256K | Chat | -1.88% | -1.06% | +1.91% | +1.92% |
| 128–256K | Agent | +1.23% | +6.33% | -12.44% | -9.41% |

The longest-context agent arm improves ITL, but worsens TTFT while chat loses
throughput. This is not a uniform serving win and does not meet the goal's
30% latency/10% throughput threshold against the split. Aggregate reported
acceptance differences stay below0.4%; that alone does not prove equivalent
per-request scheduling. Keep the plain stock-FA3/native-FA2 split as the
serving control for subsequent model validation.

`native_pdl_paired_serving_complete_20261014_audit.json` preserves all scalar
metrics, execution audits and result-file hashes. The adjacent shared archive
`paired_unpacked_complete_20261014.tar.gz` contains the serving CSV, environment,
source hashes, gate comparison logs and run log, plus the unpacked kernel logs.
Its SHA256 is
`b42013039ba233e112489fc4058917c88dfaca81bf918ba28ed554bffbd451bf`.
Full per-request serving outputs remain on the trial at the recorded root.
