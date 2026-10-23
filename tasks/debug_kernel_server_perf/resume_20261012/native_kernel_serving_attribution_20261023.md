# Native-kernel serving attribution on the FA3 backend

This isolates the native FA3 + FA2 PDL executor, not a scheduler or backend
class change. Exact same-input emitted-token parity remains open. These
runs are exploratory performance evidence, not a production-quality pass
or proof of the overall 30% latency / 10% throughput target.

## Protocol and audit

One H200, job `34c7bd503d461dc1`, trial `304487505` (`vibecuda`), Qwen2.5-7B.
Both arms use the same frozen native FA3 backend, 2,048-token chunks,
48-request cap, CUDA graphs and CPU/GPU overlap enabled. No EDF, deadline
chunking, routing or SLO-admission changes. The candidate alone enables
native attention with `geometry_v2` and FP32 partials / separate final BF16
conversion. Four-mode uses trained EAGLE3, three draft steps / four verify
tokens; two-mode disables speculation. Numerical shadow/reference passes
are not included in timed serving.

Each mode uses two opposite-order pairs: FA3/native, then native/FA3. Each
arm starts a new server, runs eight excluded 16K natural-text warmups,
flushes KV, then completes client warmups before a two-tenant start barrier.
The benchmark records request identities and requested output lengths,
cumulative token-arrival events, program concurrency and actual speculation
counters. The native arms must log execution after the workload barrier;
the controls must have no native calls. All timed sources match across arms.

`audit_native_serving_pair_20261023.py` recomputes metrics from raw outputs,
rejects failures, partial outputs, mismatched requests and missing runtime
controls. Mean per-request TPOT and token-weighted mean ITL are reported
separately. Simultaneously streamed speculative tokens have zero delivery
interval; a stream chunk is not counted as one output token. No SLO threshold
is chosen after seeing these results.

## Completed 32–64K paired matrix

Eight arms complete without request errors: each has 32 agent requests
(16 sessions, two rounds) at program concurrency eight, plus 960 natural
ShareGPT chats offered at 8 requests/s with chat templates. TraceLab supplies
session/round lengths with generated contents and omitted recorded delays;
these are not original TraceLab prompts. All request identities and requested
output lengths match, including between two-mode and four-mode arms.

Changes below use the mean of two runs per arm, native relative to FA3:

| Mode / tenant | Output tok/s | Mean TTFT | Mean request TPOT | Token-weighted mean ITL |
|---|---:|---:|---:|---:|
| Four / agent | **+3.08%** | -1.28% | **-13.59%** | **-11.93%** |
| Four / chat | -0.01% | -6.98% | -4.73% | -1.79% |
| Two / agent | **+6.05%** | -6.06% | -1.81% | -1.14% |
| Two / chat | +0.00% | -11.23% | -5.47% | -4.58% |

Four-mode agent output throughput is 133.36 → 137.47 tok/s; request-mean
TPOT is 15.90 → 13.74 ms. Agent actual tokens per verify are 3.53–3.54 and
chat about 2.01. Native execution is present in both repeats; all arms show
actual graph execution with overlap scheduling enabled. This is not the
previous slow-FlashInfer-backend comparison.

The target is **not met** by this kernel-only matrix. Chat throughput is
offered-load limited, and the finite agent cohort ends after 54–58 seconds
while chat runs about 120 seconds. Only 46–49% of chat requests start before
the last agent finishes, so this is a mixed window plus a chat-only tail,
not steady-state mixed capacity. Two-mode agent worst-ten-token-window TPOT
increases 3.76% in the two-repeat mean despite lower aggregate mean ITL;
no uniform tail-latency win is claimed. Individual-repeat noise remains
visible in the raw summary.

Full generated texts, token-event traces, client/server logs, request keys,
sources and both successful/failed-start directories are archived under
`../../pd_colocation/results_20261023/native_serving_pair_h200/`.
`audit.json` contains raw SHA-256 digests, runtime checks and recomputed results.
The archive SHA-256 is
`b7ec45a94483085af935c3788068e8220ac9dd6045a0d5f660c61b1a8a4e26cc`.
The initial startup failed on a renamed CLI option before serving any
requests; the successful run uses `--cuda-graph-max-bs-decode 48`.

## Completed 128–256K paired matrix

All eight long-context arms complete with the same checks. Each contains
16 agent requests (eight two-round sessions) at concurrency four and 960
chats at 4 requests/s. The model uses YaRN factor 16 / 524,288 maximum
context, as in the corresponding Qwen2.5 captured-input evaluation.

| Mode / tenant | Output tok/s | Mean TTFT | Mean request TPOT | Token-weighted mean ITL |
|---|---:|---:|---:|---:|
| Four / agent | **+10.58%** | -4.46% | **-24.94%** | -11.58% |
| Four / chat | +0.01% | -7.93% | **-24.93%** | **-22.63%** |
| Two / agent | **+11.29%** | -2.31% | **-23.92%** | **-21.46%** |
| Two / chat | +0.00% | **-20.23%** | -19.48% | -13.44% |

Four-mode agent throughput is 61.67 → 68.20 tok/s, request-mean TPOT
35.17 → 26.40 ms; chat TPOT is 21.00 → 15.76 ms. Two-mode agent throughput
is 54.43 → 60.57 tok/s, request-mean TPOT 59.41 → 45.20 ms. Agent throughput
improves in both repeats of both modes. Two-mode agent TTFT regresses in
the first pair and improves in the second; do not describe its small mean
improvement as a repeatable win. The mean-latency 30% target is still unmet.
Chat throughput remains offered-load limited; the 10–11% numbers concern
the finite **agent cohort**, not aggregate cluster or saturated chat capacity.

Four-mode acceptance is a material confound: agent tokens per verify are
3.283 / 2.622 in the two FA3 runs and 3.401 / 3.363 for native. No acceptance
is simulated or forced, and outputs have the same requested lengths, but
their generated contents need not match. Thus end-to-end changes include
numerical/output and batch-composition effects, not only faster attention.
The two-mode pair avoids speculative-acceptance effects and still shows an
11.29% agent throughput increase. Exact same-input parity remains open.

The chat stream overlaps agents longer than in the 32–64K cohort: 77–86%
of four-mode chats and 87–99% of two-mode chats start before the last agent
finishes. This is still a finite replay, not sustained mixed capacity.
Requests, all token events, texts, logs, source snapshots and recomputed
audit are in `../../pd_colocation/results_20261023/native_serving_long_h200/`.
Archive SHA-256:
`f90a66dd6a81e7ea04ae729b7e3950ce73946e35ab06af2e3daa4591e22b2c87`.
The main task document and its large TODO structure remain unchanged.

## Qwen2.5 v3 common-workload screen, audited (2026-10-23, Claude, picking up from the Codex session)

The run Codex left completed but unaudited: `/dev/shm/w/qwen25_v3_common_screen_h200_20261023` on the H200
(`34c7bd503d461dc1`), exit 0, eight arms (two- / four-mode x two repeats x FA3 / native v3). Audited with the
unchanged script: `cd <root> && python audit_native_serving_pair_20261023.py <root> <root>/audit_claude.json`.
Every check passes: all requests identity-matched across arms, metrics recomputed from raw token arrivals, CUDA
graphs and CPU/GPU overlap verified on, identical source hashes in all arms (native executor
`bd0d6e0a...`, backend `82885f08...`, scheduler `67970565...`). Qwen2.5-7B, 32-64K TraceLab agents at program
concurrency 8 + ShareGPT chat 8 req/s; native v3 relative to FA3, mean of both repeats:

| mode / tenant | output tok/s | token-weighted ITL | mean request TPOT | mean TTFT | worst-10% TPOT |
|---|---:|---:|---:|---:|---:|
| two / agent | **+5.8%** | **-9.1%** | -6.5% | -1.0% | -8.0% |
| two / chat | -0.0% | -6.0% | **-8.7%** | -5.6% | -9.6% |
| four / agent | +1.7% | -7.4% | -5.4% | -1.3% | +0.8% |
| four / chat | -0.0% | +0.5% | -3.4% | -5.2% | +3.3% |

Consistent with the earlier 32-64K matrix (+3-6% agent throughput): a small, real gain at this context length,
with the large gains at 128-256K. Short finite cohorts (agent tenant ~62 s, 32 requests per arm), so this is
kernel attribution, not a capacity or SLO claim. Chat throughput is pinned at the offered rate in every arm.

## Role-split FA3 in serving, 128-256K (2026-10-24, Claude)

`SGLANG_ROLE_SPLIT_FA3=1`: the mixed step as two FA3 launches (prefill rows on a side stream, decode/verify rows with
num_splits 32), no FA2. It reproduces the fixed role-split FA3 that the trace-proportional benchmark found 1-2% faster
per step than FA3 + FA2. Same paired protocol, frozen tree and scheduler as the native v3 runs above
(`rolesplit_serving_h200_20261023.sh`, H200 `34c7bd503d461dc1`, root
`/opt/tiger/w/rolesplit_serving_h200_20261023/long`); audit passes every check (all requests identity-matched,
metrics recomputed from raw arrivals, graphs and overlap on); the candidate logged `NATIVE_PDL_CALL ... role_split=1`
(432 calls in two-mode r1), the controls none. Relative to FA3, mean of two repeats, with the FA3 + FA2 native v2
result at the same context from the matrix above:

| mode / tenant | tok/s | TTFT | request TPOT | token ITL | FA3 + FA2 native: tok/s / TPOT / ITL |
|---|---:|---:|---:|---:|---|
| four / agent | **+9.9%** | -12.9% | -9.0% | -1.5% | +10.6% / -24.9% / -11.6% |
| four / chat | -0.0% | **-22.5%** | **-25.7%** | **-18.8%** | +0.0% / -24.9% / -22.6% |
| two / agent | **+11.5%** | -13.6% | +6.6% | -5.7% | +11.3% / -23.9% / -21.5% |
| two / chat | +0.0% | **-21.5%** | -19.5% | -15.1% | +0.0% / -19.5% / -13.4% |

**The role split matches FA3 + FA2 on throughput (+9.9-11.5%) and on chat latency, but is clearly worse on agent
decode latency** -- two-mode agent TPOT +6.6% against -23.9%, agent ITL -1.5..-5.7% against -11.6..-21.5%. The
per-step kernel benchmark had the role split 1-2% faster, so the FA2 rows carry an advantage in serving that
per-step attention time does not capture (one candidate: the row launch's SM footprint and how much it delays the
decode-only steps that follow; the benchmark times each step in isolation). Caveat: the FA3 + FA2 numbers come from
an earlier session with the v2 policy, not this session's v3 tree, so the comparison is across runs; a same-session
three-way pair (FA3 / FA3 + FA2 v3 / role split) is the clean test.

## Role-split FA3 in serving, 32-64K (2026-10-24, Claude)

Same protocol and audit as the 128-256K role-split pair above (root
`/opt/tiger/w/rolesplit_serving_h200_20261023/short`, exit 0). Relative to FA3, with the same-session-era FA3 + FA2
native v3 result at 32-64K (common-workload screen, audited earlier today) alongside:

| mode / tenant | tok/s | TTFT | request TPOT | token ITL | FA3 + FA2 native v3: tok/s / TPOT / ITL |
|---|---:|---:|---:|---:|---|
| four / agent | +5.6% | -5.6% | -7.1% | -2.6% | +1.7% / -5.4% / -7.4% |
| four / chat | -0.1% | -6.3% | -6.1% | -5.7% | -0.0% / -3.4% / +0.5% |
| two / agent | +5.0% | +1.2% | -9.0% | **-11.7%** | +5.8% / -6.5% / -9.1% |
| two / chat | +0.0% | -8.2% | -6.0% | -2.9% | -0.0% / -8.7% / -6.0% |

At 32-64K the two kernels are within a few points of each other on every metric, with no consistent winner (role
split ahead on four-mode agent throughput and two-mode agent ITL; FA3 + FA2 ahead on four-mode agent ITL). The
clear separation is only at 128-256K agent decode latency, where FA3 + FA2 led; the same-session FA3 + FA2 v3 pair at
128-256K (`native_v3_long_serving_h200_20261024.sh`, running) is the clean test of that.

## Same-session three-way result at 128-256K: FA3 + FA2 wins decode latency (2026-10-24, Claude)

`native_v3_long_serving_h200_20261024.sh`: FA3 + FA2 native v3 (FP32 partials, direct BF16 output) through the
IDENTICAL paired protocol and tree, in the same H200 session, right after the role-split pair; audited, exit 0.
Relative to FA3 (each pair has its own FA3 control):

| mode / tenant | FA3 + FA2 v3: tok/s / TTFT / TPOT / ITL | role split (FA3 + FA3): tok/s / TTFT / TPOT / ITL |
|---|---|---|
| four / agent | **+10.7% / -2.2% / -24.2% / -20.8%** | +9.9% / -12.9% / -9.0% / -1.5% |
| four / chat | +0.0% / -15.3% / **-23.2% / -23.3%** | -0.0% / -22.5% / -25.7% / -18.8% |
| two / agent | **+11.6% / -2.7% / -20.1% / -16.7%** | +11.5% / -13.6% / +6.6% / -5.7% |
| two / chat | +0.0% / -18.5% / -19.0% / -13.1% | +0.0% / -21.5% / -19.5% / -15.1% |

**Same-session, the earlier cross-session reading holds**: both kernels give the same throughput gain (+10-12%,
from splitting the mixed step by role), but FA3 + FA2 cuts agent decode latency 17-24% where the role split manages
-1.5..+6.6%. The FA2 rows are therefore worth ~15-25 points of agent ITL / TPOT in serving -- invisible to the
per-step kernel benchmark, which times each step in isolation and found the role split 1-2% faster. The role split
does better on TTFT (-13% vs -2%), consistent with its prefill launch finishing earlier. **The fused FA3 + FA2 kernel
is the one to ship**; the role split is the right ablation to report beside it.

## Full single-node stack vs stock sglang on FA3, 128-256K (2026-10-24, Claude)

- [x] 2026-10-24 128-256K: ours +12% agent tput, chat TTFT -69/-79%, agent ITL -31/-12%; chat TPOT +17/+30%. Audit passes.
- [ ] 32-64K stage (running on H200 34c7bd503d461dc1, `ONLY=short`).
- [ ] Attainment vs chat rate at 128-256K (both arms), for capacity at 90%.

Command (H200 34c7bd503d461dc1, sglang `6acf9a297` + the audit fix in this commit):
`bash ours_full_stack_h200_20261024.sh now`. It runs `run_native_serving_pair_20261023.py <root> --stage mix128
--modes four two --concurrency 4 --sessions 8 --chat-rate 4 --chat-requests 960 --split-policy geometry_v3
--direct-output --packed-query-limit 64 --repeats 2 --ours`, then `audit_native_serving_pair_20261023.py`.
Root `/opt/tiger/w/ours_full_stack_h200_20261024/long`. Per-tenant scoring: `bench_per_tenant.py mix128_*/requests`.

- **The fa3 arm under `--ours` is stock sglang.** In four-mode it runs stock EAGLE3 MTP (no `SGLANG_MIXED_SPEC`).
- **The native arm is ours:** the EDF env (uncached-token deadlines, chunk preemption, deadline chunk 1024:2048:6000)
  plus FA3 + FA2 v3 plus the 4-mode step.
- **The audit enforces the split in both directions.** A four-mode merge is required exactly when
  `SGLANG_MIXED_SPEC=1`; an arm without it must log zero `MIXEDSPEC_STEPS`.
- **The previous audit rejected the valid stock control.** It required a merge in every four-mode arm, so this rerun
  of the audit exists only to apply the fix; the arms were not re-measured.

Audit (relative change, ours vs stock, mean over two opposite-order repeats):

| mode / tenant | tok/s | TTFT | TPOT | ITL | worst-10% TPOT |
|---|---:|---:|---:|---:|---:|
| four / agent | +12.4% | +2.9% | -34.9% | -30.6% | -51.4% |
| four / chat | +0.0% | -68.6% | +17.2% | +12.5% | -5.5% |
| two / agent | +11.8% | -14.5% | -35.0% | -11.7% | -20.7% |
| two / chat | +0.0% | -79.4% | +30.5% | +21.4% | +35.6% |

Per-tenant, per repeat (r1 / r2; chat n=960, agent n=16):

| arm | agent attain. 10 s/80 ms | agent TPOT p90 | chat TTFT mean | chat attain. 0.5 s/20 ms | chat TPOT p50 / p90 |
|---|---:|---:|---:|---:|---:|
| four, stock MTP | 12.5 / 12.5% | 115 / 126 ms | 23.0 / 23.6 s | 17.9 / 17.5% | 17.7 / 16.1, 60 / 64 ms |
| four, ours | 31.2 / 31.2% | 81 / 64 ms | 6.9 / 7.7 s | 27.7 / 27.0% | 16.7 / 17.0, 70 / 74 ms |
| two, stock | 12.5 / 12.5% | 123 / 114 ms | 21.6 / 21.0 s | 16.8 / 16.8% | 18.9 / 18.9, 68 / 69 ms |
| two, ours | 43.8 / 50.0% | 61 / 59 ms | 3.1 / 5.7 s | 32.1 / 31.5% | 17.2 / 16.6, 96 / 91 ms |

Reading:
- **Ours clears the throughput bar.** Agent tok/s rises 12% in both modes.
- **Ours clears the 30% latency bar on the tenant each mechanism targets.** Chat TTFT falls 69-79% (deadline order);
  agent TPOT falls 35% and four-mode agent ITL falls 31% (kernel + 4-mode).
- **Chat decode pays the cost.** Chat p50 TPOT is unchanged, but the p90 tail rises from 60-69 ms to 70-96 ms.
  Deadline order lets agent prefill chunks ride in chat decode steps.
- **This load is past both systems' capacity.** Chat attainment is 17% for stock and 27-32% for ours, and agent
  attainment counts are small (2/16 vs 5-8/16). The attainment and goodput ratios are therefore not a paper headline;
  a chat-rate sweep is needed.
