# B: kernel-serving comparison at controlled program concurrency

2026-10-14. This measures the implemented epilogues with server scheduling held fixed.
It is not an EDF experiment, a complete workload evaluation, or a change to the main A/B/S
TODOs. Previous kernel evidence: `b1_chain_epilogue_20261014.md`.

## Concurrency basis and limits

[ThunderAgent Figure 4](https://arxiv.org/html/2602.13692v1#S5.F4) sweeps parallel workflows,
not a universal session-arrival rate. Its ToolOrchestra/Qwen3-8B panel uses24/32/40/48
workflows; several larger-model panels use24/48/72/96. The SVG's `data-text` labels were
checked directly. These counts are not a claim that our different model, hardware and
context sizes have the same capacity. This kernel screen uses24 and48 active programs,
with the same frozen48-program,96-round cohort at both limits.

`drivers/serving/session_concurrency.py` is a client-side worker pool. Each slot owns a
whole session until its last selected round completes, serializes that session's rounds,
and then admits the next program. It can retain the slot through recorded tool waits.
It is not a cap on individual HTTP requests and does not modify server admission policy.
Five async tests cover the cap, causal order, output ordering, tool-wait ownership, failures,
cancellation, empty input and invalid settings.

The full-release `syfi_full_min.jsonl.gz` preserves lengths/session identities but has **no
timing events**. Therefore this run is explicitly a finite, gap-free length/session kernel
stress test. It is not a faithful tool/human-delay replay, sustained fixed-concurrency
capacity measurement, or literal agent-text replay. The public-trace sampler synthesizes
token contents. It admits both providers, keeps outputs down to one token, and excludes
only records outside the requested context band or served total-token window. Consecutive
selected records need not be consecutive original rounds after those filters.

The future realistic workload matrix still needs original timing data and a properly
defined program/workflow boundary. This screen does not replace that requirement.

## Matched matrix

- Context bands32–64K,64–128K and128–256K; corresponding served windows131072/131072/262144
  and YaRN factors4/4/8, preserving the model's original `rope_theta`.
- Agent concurrency24/48, fixed48 programs and96 rounds selected with seed20261014.
- Co-tenant:2048 natural ShareGPT requests at8 requests/s, rather than controlled-length
  random chat. This single co-tenant rate is a kernel-isolation point, not the requested
  complete request-rate sweep.
- Each point runs BASE_1 / CHAIN / BOTH / BASE_2. BASE disables both epilogues; CHAIN enables
  only the new tree kernel; BOTH also enables the older greedy-verification epilogue.
- All arms use the same plain FA3/native-FA2 split, EAGLE3 chain q4, CUDA graphs, chunk
  settings and48-request server limit. No EDF flags, deadline policy, simulated acceptance
  or altered SM/dispatch rule.
- Both controls and candidates prewarm chain specializations for batch1–48 on the server
  during untimed client warmup, before releasing the two-stream start barrier. This prevents
  candidate-only JIT compilation during timed requests.

This is24 arms, including two independent baseline runs at each point. Performance claims
need the baseline-repeat variation and measured acceptance/composition, not just a faster
single candidate run. Finite tenants can drain at different times; the audit reports their
overlapping spans rather than describing the entire run as steady mixed load.

## Execution and acceptance checks

PID766826 on the existing single H200 trial `095f01a3e1585eea`, private tree
`/dev/shm/w/up_b1_kernel_serving/python`. Root/log:
`/dev/shm/w/b1_kernel_concurrency_20261014[.log]`. The synchronizing shadow-validation tree
is not used. No successor is queued.

Overlay SHA256 `801afdb05391d02ca153292c3e15160555cc7b9d7c8833ac979a012ae67bc758`.
All baseline/candidate source hashes pass before execution. A metadata-only preflight
checks enough eligible sessions in every band and records selected round identities.

`drivers/serving/audit_kernel_concurrency.py` independently reconstructs the active-program
count from start/end events, requires the configured peak, checks complete request counts,
exact prompt/output fingerprints and round identities against BASE_1, positive split/mixed
execution and the intended epilogue flags, graphs on, and no errors/fallbacks/EDF runtime.
Results retain per-request start/end times and cumulative token-arrival events.

Per-tenant TTFT, ITL/TPOT tails and throughput remain the evaluation outputs. No paper-derived
SLO attainment is claimed without matching zero-load references; the earlier128-request
ShareGPT calibration is not a reference for this different2048-request cohort.

## First completed point:32–64K,24 programs

All four arms completed09:44:34. Every arm completed96 agent rounds and2048 chat requests,
reached exactly24 active programs, passed fingerprint/session-identity matching, exercised
mixed/split attention and the intended epilogues, and had no errors/fallbacks. Tenant spans
overlap98.8–99.0%. Relative to the mean of the two baseline runs:

| Variant | Tenant | Output tok/s | Mean TTFT | Mean ITL | P99 ITL |
|---|---|---:|---:|---:|---:|
| CHAIN | Agent | -0.67% | +4.74% | -12.81% | -5.16% |
| CHAIN | Chat | -0.73% | +4.55% | +1.56% | +3.44% |
| BOTH | Agent | -1.46% | +5.60% | -8.99% | -5.31% |
| BOTH | Chat | -1.34% | +5.37% | +2.79% | +2.48% |

**No promotion:** there is no throughput/TTFT win, and agent ITL improvement alone does not
meet the goal or compensate for the other tenant's regression. Baseline-repeat spreads are
1.55–1.75% throughput,3.38–4.84% TTFT and0.58–5.60% P99 ITL; these are observed spreads,
not confidence intervals. Candidate runs need repetitions before small differences are
interpreted causally.

Acceptance is another limitation: the request-counter-derived agent strict draft acceptance
rates are0.817/0.862/0.809/0.877 for BASE_1/CHAIN/BOTH/BASE_2. The reported `accept_length`
near2.12 is a server-wide snapshot, not a per-tenant acceptance length. Do not attribute the
agent ITL change purely to the epilogue's microsecond savings.

`b1_kernel_concurrency_progress_20261014.json` contains the four completed audits. Complete
raw first-point archive remains on the trial at `/dev/shm/w/b1_kernel_mix32_c24_20261014.tar.gz`,
21626255 bytes,47 manifest entries, SHA256
`63d484d72eaf9b946184428aba0bf8bdae9b99925739453bebc7370f1830e93d`.
The48-program point and larger context bands continue under PID766826; no EDF work starts.

The first-point archive is now local. Its hash and all47 file hashes pass; independent local
checks confirm identical fingerprints and both requested and actual output lengths for all
four arms, for each tenant.

## Complete matrix

All24 arms finished20:05:41. Controller766826 is terminal. The complete scalar/execution
audits are in `b1_kernel_concurrency_complete_audit_20261014.json`; all request, fingerprint,
program-concurrency and execution checks pass. Relative to each point's two-baseline mean,
the final128–256K results are:

| Programs | Variant | Tenant | Output tok/s | Mean TTFT | Mean ITL | P99 ITL |
|---|---|---|---:|---:|---:|---:|
| 24 | CHAIN | Agent | +0.34% | +0.56% | -4.21% | -4.93% |
| 24 | CHAIN | Chat | +0.81% | -2.38% | +0.42% | -1.32% |
| 24 | BOTH | Agent | -0.18% | +2.09% | -6.45% | -4.14% |
| 24 | BOTH | Chat | -5.17% | +1.68% | +6.69% | +3.03% |
| 48 | CHAIN | Agent | +1.09% | -0.89% | +2.42% | -0.79% |
| 48 | CHAIN | Chat | -0.92% | -1.26% | -0.66% | +3.27% |
| 48 | BOTH | Agent | +0.62% | +0.49% | +4.33% | -1.38% |
| 48 | BOTH | Chat | +0.14% | -1.29% | -5.46% | -0.56% |

Neither epilogue variant provides a general serving win across all six points. Keep the
implementations opt-in. Kernel-only savings do not establish the end-to-end target, and
acceptance/composition variation prevents attributing small latency changes to fusion alone.
The new A stock-primary pilot uses a separate private tree and does not enable these epilogues.

The complete compact evidence archive is local:
`b1_kernel_concurrency_compact_complete_20261014.tar.gz`,1825387 bytes,124 verified file
hashes, SHA256 `6128c221d0153f3d3bee4981b7ce46185d0bdfe8cd33cd70a7a474e3df81a7a0`.
It contains all24 audits, server/driver logs and program-event records. Full token-arrival
and generated-text client outputs remain on the trial, except the first point's separate
full raw archive; their content hashes are retained in the audits.

## Second completed point:32–64K,48 programs

The four48-program arms completed10:16:41; all execution, completion and identity checks
pass. Measured program peak is48, and tenant spans overlap96.6–99.1%. Relative to the mean
of the two48-program controls:

| Variant | Tenant | Output tok/s | Mean TTFT | Mean ITL | P99 ITL |
|---|---|---:|---:|---:|---:|
| CHAIN | Agent | -1.93% | +2.52% | -7.00% | -3.02% |
| CHAIN | Chat | -0.03% | +1.69% | -0.02% | -1.48% |
| BOTH | Agent | -1.69% | +1.96% | -8.18% | -8.03% |
| BOTH | Chat | +0.11% | +1.88% | +0.05% | -1.15% |

Again, no general throughput/TTFT improvement and no promotion. The small chat changes are
not established gains; one run per candidate is insufficient for that claim. Eight of24
arms are complete. The64–128K band began10:16:41, with the same PID and fixed server policy.

## 64–128K points complete

Sixteen serving arms now pass their completion, identity, concurrency and execution audits.
Relative to the two-control mean at each concurrency:

| Programs | Variant | Tenant | Output tok/s | Mean TTFT | Mean ITL | P99 ITL |
|---:|---|---|---:|---:|---:|---:|
| 24 | CHAIN | Agent | +0.21% | +0.78% | -2.35% | -10.59% |
| 24 | CHAIN | Chat | -0.88% | +0.31% | -0.92% | +0.12% |
| 24 | BOTH | Agent | +0.24% | -1.38% | +4.39% | +4.07% |
| 24 | BOTH | Chat | +0.19% | -1.79% | -2.04% | -1.41% |
| 48 | CHAIN | Agent | -1.31% | +3.68% | +10.09% | +10.07% |
| 48 | CHAIN | Chat | -1.53% | +0.51% | -5.37% | -4.60% |
| 48 | BOTH | Agent | +0.05% | +0.47% | +8.67% | +8.89% |
| 48 | BOTH | Chat | +0.01% | +0.07% | -3.54% | -2.36% |

These do not establish a general serving win. Agent/chat latency trade-offs remain, and
sub-percent throughput changes are not evidence of the required capacity gain. Both flags
remain opt-in. The128–256K points remain pending.

After the `mix64_c48/BASE_2` audit, A's static-GQA kernel gates run under the controller's
inherited lock before the next serving point. This gap, like the earlier no-split experiment,
does not enter either client's benchmark duration. Use result timestamps, not ARM_DONE wall
time, for comparisons. Updated scalar audits are in the progress JSON.
