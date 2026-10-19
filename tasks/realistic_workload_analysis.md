# Realistic Workload Analysis for the Persistent Overlap Kernel

## Scope

This document contains the current, correctness-valid evaluation of persistent
prefill/decode overlap. Historical B200 scans, graph-incompatible runs,
median-only latency tables, synthetic prioritization notes, and superseded
arrival-rate surveys were removed; they remain available in Git history.

The paper-facing comparison has three methods (plus one variant):

- **PP:** prefill-prioritized scheduling with unchanged BatchPrefill.
- **DMC:** default mixed-chunk scheduling with unchanged BatchPrefill
  (`SGLANG_FLASHINFER_USE_PERSISTENT=0` set explicitly — the flag defaults on).
- **LPT:** mixed scheduling with the gated persistent overlap kernel
  (`SGLANG_MIXED_ATTENTION_KERNEL=auto`, measured k-NN gate) at the matched
  8,192-token chunk cap; 6,656 only where marked as the tail-optimized point.
- **LPT_EDF:** LPT plus the SLO-aware chunk controller driven by the analytic
  overlap cost model (`SGLANG_EDF_LATENCY_MODEL=analytic`,
  `SGLANG_EDF_PLAN_OVERHEAD_MS=0`).

Ordinary decode CUDA graphs are enabled for every method. Piecewise EXTEND
graphs are not combined with decode graphs because that combination is
output-incorrect at N=400. LPT has no minimum prefill chunk. Every reported
point retokenizes within 2% of its requested output.

The story order below: pure long-context replay (parity — the zero-regression
check), the occupancy scan that locates where the kernel matters, why
independent-prompt datasets cannot show it, and the mixed-tenancy scenarios
where it wins end to end, including SLO control with the analytic cost model.

## Baselines

Which systems each result is judged against (user, 2026-10-18; delineated by SLO control 2026-10-19). The bar is
>= 30% on TTFT and ITL, or on their SLO attainment rates, and >= 10% throughput on realistic mixed workloads (chat +
TraceLab), same session, same workload. "SLO control" = the decode-first worker rule (every step admits only the
prefill that keeps running decodes within their ITL SLO, no exception; `apply_edf_itl_slo_chunk.py`) and, multi-node,
the router predicting each node's TTFT / ITL from its load and routing to the highest-hit node that stays within SLO.

| setting | our arm | baselines | what it shows | priority |
|---|---|---|---|---|
| single node, no SLO control | fused split (FA3 prefill tiles + FA2 decode tiles, FA2 overlapped), EDF order + deadline chunk; without MTP and with MTP (4-mode step) | sglang **prefill-prioritized** chunked prefill (default, no mixed chunk) and sglang **mixed chunk**, default kernel; each without spec and with MTP | kernel fusion + schedule efficiency: no re-launched chunks, no decode rows padded into prefill tiles, no head-of-line blocking | required |
| single node, SLO control | the same + decode-first rule | **pdmux** (stock `decode_bs_divisor` 36; its SM split is its own SLO mechanism) | ours meets decode SLOs with one CUDA-graph set and the CPU-GPU overlap scheduler, which pdmux lacks | nice to have |
| multi-node, SLO control | symmetric colocated nodes, decode-first workers, load-aware SLO-checked hit-first router, cluster KV pool; without and with MTP | **PD disaggregation** (sglang prefill / decode roles, mooncake) without and with MTP on decode nodes; **semi-PD**-style systems with an sglang implementation | higher capacity than disaggregation (request rate at a fixed SLO attainment target) while sticking to the SLOs | required (the main claim) |

- Model and data: Qwen2.5-7B-Instruct + EAGLE3 head; FP8 is out of scope. Agent load is program concurrency, chat by
  request rate.
- MTP is required, not optional: almost every model ships an MTP head, and disaggregation runs MTP on its decode nodes
  to raise their compute utilisation -- which would undercut the claim that colocation maximises compute utilisation
  and serving capacity. Every setting is run with and without MTP against the matching baseline.
- pdmux is a baseline only: it runs without the CPU-GPU overlap scheduler and captures one CUDA-graph set per SM group.
- [ ] Prefill-prioritized arm (sglang default, no `--enable-mixed-chunk`): not yet run on H200; the stock arms so far
  are mixed chunk (`coloc_sgl`, `coloc_sgl_mtps`).
- [ ] Single-node SLO-control arm vs pdmux; multi-node SLO-control arms vs disagg / semi-PD with the fitted 4-mode step
  model and node-reported load (`server_perf_model.md`).
- [x] **Single-node A/B, H200 `d6a80befdd4e4da5`, 2026-10-18 (same session; agent C=8 frozen 32-64K TraceLab x 2 rounds
  + ShareGPT 8 req/s; one worker; all six arms 0 tracebacks / 0 fallbacks; split arms >= 65.5K split calls,
  overlapped arm `overlapped=71501` = every split; chat throughput = offered load in every arm).** Agent tput tok/s /
  TTFT / ITL ; chat TTFT / ITL:
  | arm | agent | chat |
  |---|---|---|
  | stock chunked prefill + mixed chunk, no spec | 89.9 / 3.12 s / 60.5 ms | 1.04 s / 29.7 ms |
  | stock + MTP | 74.0 / 16.85 s / 24.0 ms | 8.95 s / 15.8 ms |
  | stock pdmux (divisor 36) | 101.4 / 8.11 s / 27.0 ms | 2.03 s / 15.5 ms |
  | ours, no MTP, split sequential | 102.7 / 3.90 s / 47.5 ms | 0.17 s / 22.7 ms |
  | **ours, no MTP, FA2 overlapped** | **113.0 / 3.61 s / 41.6 ms** | **0.15 s / 19.0 ms** |
  | ours + MTP (4-mode, sequential split) | 103.2 / 12.32 s / 17.1 ms | 2.81 s / 12.5 ms |
  Ours without MTP (overlapped) vs stock without MTP: agent +25.7% / **+15.6%** / -31.2%, chat TTFT -85.6% / ITL -36.0%
  -- clears everything except agent TTFT; the sequential split gives +14% / +24.8% / -21.5%, -84% / -23.6%, so FA2
  overlap is worth +10% agent throughput and -12 / -16% agent / chat ITL. Ours + MTP vs stock + MTP: agent +39.5% /
  -26.9% / -28.8%, chat -68.6% / -20.9%. Agent TTFT is the miss without MTP: stock prefills in 8192-token chunks, ours
  in the 1024 / 2048 deadline chunks tuned for the MTP arm. `bash tasks/pd_colocation/mn_h200_nospec_20261018.sh`;
  sglang `d33dcf44c`.
  - [x] **Chunk size on the no-MTP overlapped arm, and overlap on the MTP arm (same session, all clean, every split
    overlapped: `overlapped` = split calls, 56-72K):** vs stock without spec, agent tput / TTFT / ITL, chat TTFT / ITL:
    | ours, no MTP, overlapped, deadline chunk | agent | chat |
    |---|---|---|
    | 1024 / 2048 | +25.7% / +15.6% / -31.2% | -85.6% / -36.0% |
    | 2048 / 4096 | +24.9% / +0.3% / -26.3% | -86.7% / -30.6% |
    | 2048 / 8192 | +24.8% / **-10.4%** / -24.0% | -87.5% / -29.6% |
    Chunk size moves agent TTFT against agent ITL along one frontier; no setting clears both (stock's agent TTFT is
    3.12 s; -30% needs 2.19 s). Ours + MTP with overlap vs stock + MTP: agent +41.8% / -27.3% / -28.8%, chat -69.1% /
    -22.2% -- the same as the sequential split (verify rows are few, so FA2 overlap has little to hide).
    `ARMS=ours_mtp_ovl` / `ARMS=ours_ovl CHUNK={8192,4096} DEADLINE_CHUNK=2048:{8192,4096}:6000 bash
    tasks/pd_colocation/mn_h200_nospec_20261018.sh`.
  - [x] **Agent-TTFT floor and the deadline slope, no-MTP overlapped arm (2026-10-18, all arms clean).** Unloaded
    (1 program, chat 0.2 req/s, same 32-64K cold prompts): stock 1.30 s, ours (2048 / 8192) 1.20 s mean agent TTFT --
    so of the loaded 3.12 / 2.80 s, ~1.7 s is queueing, and -30% (2.18 s) is not ruled out by compute. EDF's deadline
    slope (default 1000 ms per 5K uncached tokens) puts every chat ahead of every agent prefill (chat TTFT 0.13-0.17 s
    vs stock 1.04 s), so flatter slopes trade chat headroom for agent TTFT. vs stock without spec, agent TTFT / ITL ;
    chat TTFT / ITL (agent throughput +22..+27% in every arm):
    | deadline chunk, slope | agent | chat |
    |---|---|---|
    | 2048 / 8192, 1000 | -10.4% / -24.0% | -87.5% / -29.6% |
    | 2048 / 8192, 250 | **-24.3%** / -19.8% | -62.8% / -29.3% |
    | 2048 / 8192, 0 | -20.4% / -20.8% | -39.8% / -31.6% |
    | 1024 / 4096, 250 | -13.3% / -23.5% | -60.6% / -31.6% |
    | 1024 / 4096, 500 | -4.3% / -25.1% | -72.3% / -31.3% |
    | 1024 / 2048, 250 | +2.2% / -28.6% | -51.9% / -36.4% |
    | 1024 / 2048, 1000 | +15.6% / **-31.2%** | -85.6% / -36.0% |
    Agent TTFT and agent ITL lie on one frontier (sum ~ -45..-50 points); no chunk / slope setting reaches -30% on
    both. `TTFT_SCALE_MS=<slope> ARMS=ours_ovl CHUNK=<large> DEADLINE_CHUNK=<small>:<large>:6000 bash
    tasks/pd_colocation/mn_h200_nospec_20261018.sh`; floor: `AGENT_CONC=1 SESSIONS=8 TURNS=2 CHAT_RATE=0.2 CHAT_N=40
    TAG=_floor ARMS="stock ours_ovl" CHUNK=8192 DEADLINE_CHUNK=2048:8192:6000 bash ...`. sglang `253e9968c`.
  - [x] **Output gate, both FA2-overlapped arms: PASS (2026-10-18, `tasks/pd_colocation/gate_h200_ovl_20261018.sh`,
    srv_correctness).** No MTP: stock LPT is deterministic run to run (LPT_REP agrees on all 96 long-prompt tokens,
    short >= 1); ours (`SPLIT_OVL_EDF`, deadline chunk 2048 / 8192, chunk 8192) diverges at token 11 on one long
    prompt -- and so does the null case, stock with sgl-kernel's FA3 backend at the same chunk (`LPT_FA3`,
    `--attention-backend fa3`: long >= 11, short >= 2), so the divergence is the attention kernel's bf16 rounding,
    not the split or the schedule. MTP: the overlapped 4-mode arm (`MTP_MIX_SPLIT_OVL_EDF`) vs stock MTP matches the
    chunk-size null case of the earlier 4-mode gate position for position (long >= 11, short 0; s1's first decode token
    flips on chunking alone). Split fired with every call overlapped (gate runs: 501 / 1501 split calls, `overlapped`
    equal); 0 tracebacks / 0 fallbacks. `bash /dev/shm/w/mn/gate_h200_ovl_20261018.sh` on pod `d6a80befdd4e4da5`.
- [x] **Single-node A/B on the long-context mix (2026-10-18, H200; TraceLab 32-250K, YaRN x8 / context 262144, agent
  C=6, 18 sessions x 2 rounds, 4 rounds > 200K, + ShareGPT 2 req/s; our arms with the 2x deadline above 200K; all arms
  0 tracebacks / 0 fallbacks; split calls 28.5K (no MTP) / 40.5K (MTP)).** Agent tput / TTFT / ITL ; chat tput / TTFT /
  ITL ; TTFT SLO attainment agent (10 s, 20 s >= 200K) / chat (2 s):
  | arm | agent | chat | attainment |
  |---|---|---|---|
  | stock, no spec | 46.3 / 9.92 s / 105.4 ms | 341.7 / 12.49 s / 178.4 ms | 66.7% / 32.0% |
  | stock + MTP | 46.1 / 27.51 s / 71.2 ms | 329.8 / 36.92 s / 100.4 ms | 19.4% / 0.2% |
  | ours, no MTP (overlapped, 2048 / 8192) | 67.8 / 8.33 s / 67.8 ms | 402.0 / 0.60 s / 98.2 ms | 63.9% / 97.3% |
  | ours + MTP (4-mode) | 71.0 / 18.22 s / 36.5 ms | 406.5 / 3.57 s / 58.1 ms | 58.3% / 70.0% |
  **Ours + MTP vs stock + MTP clears all six:** agent +54.0% / -33.8% / -48.7%, chat +23.3% / -90.3% / -42.1% -- the
  second repeat on this mix (the first, 2026-10-18 earlier: +53% / -37.5% / -44%, +23% / -86% / -45.5%). **Ours
  without MTP vs stock without spec:** agent +46.4% / -16.1% / -35.7%, chat +17.6% / -95.2% / -45.0% -- five of six;
  agent TTFT misses. Both stock arms serve below the offered chat load (~407 tok/s); ours serve all of it. Stock
  without speculation has far lower TTFT than stock + MTP here as on the standard mix, so the MTP arm's TTFT lead is
  over stock + MTP only (vs stock without spec: agent TTFT +84%, ITL -65%). `YARN_FACTOR=8.0 CTX_LEN=262144
  MIN_CTX=32768 MAX_CTX=250000 AGENT_CONC=6 SESSIONS=18 CHAT_RATE=2 CHAT_N=600 TAG=_long_base LONG_DEADLINE=200000:2`
  then `ARMS="stock stock_mtp"`, `ARMS=ours_ovl CHUNK=8192 DEADLINE_CHUNK=2048:8192:6000`, `ARMS=ours_mtp` with
  `bash tasks/pd_colocation/mn_h200_nospec_20261018.sh`; sglang `c739f4294`.
  - [x] Same mix, no-MTP arm with deadline slope 250 (`TTFT_SCALE_MS=250`; clean): agent 64.1 / 7.05 s / 75.3 ms, chat
    402.0 / 2.95 s / 110.3 ms -> vs stock without spec agent +38.4% / **-28.9% / -28.6%**, chat +17.6% / -76.4% /
    -38.2%. The slope moves agent TTFT against agent ITL as on the standard mix (default slope: -16.1% / -35.7%); both
    sit 1-1.5 points short of -30% here. Slopes 400 / 500 (clean): agent -27.0% / -30.6% and -23.7% / -33.5% (TTFT /
    ITL), chat -84.4% / -40.1% and -88.4% / -44.4%, agent throughput +41% / +45%. Four slopes trace one frontier
    through (-29, -29) -> (-27, -31) -> (-24, -34) -> (-16, -36); single repeats over 36 agent rounds, so the 250-400
    points are within noise of the bar on both. `TTFT_SCALE_MS={250,400,500}` + the long-mix command above.
  - [x] **Three matched repeats of slopes 250 / 400 (stock re-run each repeat; all clean).** Stock agent TTFT 9.92 /
    9.87 / 10.31 s, ITL 105.4 / 105.7 / 104.2 ms; chat 12.49 / 12.65 / 13.20 s, 178.4 / 178.1 / 174.5 ms. Ours slope
    250: agent 7.05 / 6.99 / 6.74 s, 75.3 ms x3; slope 400: 7.24 / 6.93 / 7.12 s, 73.1 / 73.8 / 73.3 ms. Means vs stock:
    | slope | agent tput | agent TTFT | agent ITL | chat TTFT | chat ITL |
    |---|---|---|---|---|---|
    | 250 | +39% | **-30.9%** | -28.4% | -78% | -38% |
    | 400 | +41% | -29.2% | **-30.2%** | -85% | -40% |
    The spread between repeats is small (stock TTFT +-2.5%, our ITL +-1%), so these are two points on the measured
    frontier: each clears one agent metric and misses the other by 0.8-1.6 points; neither clears both. `for REPN in 2
    3; do TAG=_long_base_rep$REPN; ARMS=stock ...; for TS in 250 400; do TTFT_SCALE_MS=$TS ARMS=ours_ovl CHUNK=8192
    DEADLINE_CHUNK=2048:8192:6000 ...` (`/tmp/h20/long_ts3.sh`, long-mix env above). Slope 325 (two repeats, clean):
    agent 7.01 / 6.87 s, 74.7 / 74.7 ms -> vs the three-repeat stock mean -30.8% / -28.9%, chat -82% / -38%: at agent
    TTFT -30% the frontier puts agent ITL at ~-29%. Slope 250 with deadline chunk 2048 / 4096 (three repeats, clean,
    `bash /tmp/h20/long_ts5.sh`): agent 8.07 / 8.25 / 8.29 s, 70.4 / 70.0 / 69.8 ms -> -18.2% / -33.3% (agent
    throughput +43%, chat -70% / -44%): the chunk moves the point along the same frontier, it does not lift it. Seven
    chunk / slope settings now trace one agent TTFT / ITL frontier on this mix with no point at -30% / -30%.
- [x] **Per-tenant rerun of the standard mix, two matched repeats (2026-10-19, H200 `dadf94f0e1da7238`, re-forked and
  provisioned from the H20 tree: upstream 3fef129 + the current H20 overlay, `scheduler.py` md5 aa706592; agent C=8
  frozen 32-64K TraceLab x 2 rounds + ShareGPT 8 req/s; all eight arms 0 tracebacks / 0 fallbacks; no-MTP split arms
  71.5K / 72.5K split calls, every one overlapped; MTP arms 34.5K / 33.5K split calls = ~1.2K 4-mode steps, plus ~1.4K
  short prefills run on their own by the context / queue gate (`MIX_MIN_CTX` 8192, queue < 4, the default since 10-16);
  the 10-18 ">= 66K" MTP count cannot be re-checked, its logs went with that pod).** Means of the two repeats, agent tput /
  TTFT / ITL ; chat TTFT / ITL:
  | arm | agent | chat |
  |---|---|---|
  | stock mixed chunk, no spec | 78.4 / 5.45 s / 62.9 ms | 2.45 s / 34.8 ms |
  | ours, no MTP, FA2 overlapped | 112.9 / 3.52 s / 42.6 ms | 0.145 s / 19.2 ms |
  | stock + MTP | 63.7 / 20.70 s / 24.3 ms | 13.31 s / 17.6 ms |
  | ours + MTP (4-mode, 1024 / 2048) | 102.2 / 12.64 s / 16.3 ms | 2.96 s / 12.65 ms |
  Ours without MTP vs stock without spec: agent **+44% / -35.3% / -32.4%**, chat **-94.1% / -44.8%** -- all clear in
  both repeats (rep 1 +44 / -34 / -32, -94 / -45; rep 2 +44 / -36 / -32, -94 / -45). Ours + MTP vs stock + MTP: agent
  **+60.5% / -38.9% / -32.9%**, chat **-77.8%** / ITL -28.1% (chat ITL misses by 1.9 points, rep 1 -27.7%, rep 2 -28.6%).
  **CORRECTION (2026-10-19, later the same day): stock is bimodal, and the "all clear" above rests on its slow mode.**
  A third stock repeat (no spec) gave agent 89.8 tok/s / 3.01 s / 61.1 ms, chat 1.04 s / 29.8 ms -- the 10-18 numbers
  (89.9 / 3.12 s / 60.5 ms, 1.04 s / 29.7 ms), not reps 1-2 (78.4 / 5.45 s). Same pod, tree, seeds; all three prefill
  the same 2.12M tokens, but the slow runs use 20% more, smaller prefill steps (2329-2339 vs 1946; mean 910 vs 1089
  new tokens) and queue 140+ requests behind the first agent prefills in the first minute (queue mean 17.5-18.0 vs
  2.3). Against rep 3, ours without MTP is agent +26% / TTFT **+17%** / ITL -30%, chat -86% / -35% -- agent TTFT is the
  miss again, as on 10-18. Three more matched repeats of all four arms are queued (`h200_more_reps_20261019.sh`) so
  each comparison is reported over the distribution, not one mode.
  Stock + MTP rep 3 also matches 10-18 (72.3 / 17.26 s / 23.8 ms, chat 9.47 s / 16.3 ms). Means vs stock's fast mode
  (rep 3), ours + MTP: agent +41% / -26.8% / -31.5%, chat -68.7% / -22.4%. **Per tenant vs rep 3** (attainment = TTFT
  and per-request TPOT both met; ours = its two reps):
  | pair | agent 10 s + 80 ms | agent goodput | chat 0.5 s + 20 ms | chat 2 s + 50 ms | throughput |
  |---|---|---|---|---|---|
  | no MTP: stock rep 3 / ours | 55.6% / 91.1-93.3% | 51.7 / 98.2-101.9 tok/s | 58.5% / 69.7-69.9% (+19%) | 70.5% / 80.0-80.3% (+14%) | +26% |
  | MTP: stock rep 3 / ours | 2.2% / 42.2-46.7% | 2.9 / 39.3-41.2 tok/s | 41.0% / 58.1-60.2% (+44-47%) | 42.8% / 61.9-65.7% (+45-53%) | +41% |
  With MTP the attainment bar clears against stock's fast mode on both tenants (> +30%) with +41% throughput. Without
  MTP chat attainment gains only +14-19%: our chat TPOT p90 (~61 ms) caps it. Single repeat of the fast mode; reps
  4-6 decide the distribution.
  Earlier caveat: stock is slower on this pod than on the 10-18 pod (no spec: TTFT 5.45 vs 3.12 s, chat 2.45 vs 1.04 s;
  MTP: 20.7 vs 16.85 s) while our arms match their 10-18 numbers (3.5 vs 3.6 s; 12.6 vs 12.1-12.3 s). Same-session
  comparison is the rule and both repeats agree, but why stock moved is not explained yet -- open.
  Per tenant (attainment = share of requests meeting both a TTFT and a per-request TPOT target, TPOT = sum(itls) /
  (output - 1); goodput = their output tok/s), rep 1 | rep 2:
  | arm | agent 10 s + 80 ms | agent 10 s + 30 ms | chat 0.5 s + 20 ms | chat 2 s + 50 ms | chat TPOT p90 |
  |---|---|---|---|---|---|
  | stock, no spec | 48.9 / 44.4% (42.0 / 36.5 tok/s) | 2.2 / 2.2% | 52.6 / 52.9% | 60.9 / 61.0% | 100.7 / 97.8 ms |
  | ours, no MTP | **93.3 / 91.1% (98.2 / 101.9)** | 2.2 / 2.2% | **69.7 / 69.9%** | **80.0 / 80.3%** | 62.7 / 61.1 ms |
  | stock + MTP | 2.2 / 2.2% (2.5 / 2.6) | 0 / 0% | 31.7 / 32.9% | 32.3 / 33.7% | 36.7 / 36.6 ms |
  | ours + MTP | **46.7 / 42.2% (41.2 / 39.3)** | **42.2 / 40.0%** | **58.1 / 60.2%** | **61.9 / 65.7%** | 30.6 / 30.8 ms |
  Per tenant, chat meets its SLO 1.3x (no MTP) and 1.8-1.9x (MTP) as often as stock, agent goodput is 2.3-2.8x and
  ~16x. Chat attainment is capped by a TPOT tail (p90 61-63 ms without MTP, 31 ms with): the chat requests decoding
  through deep-context mixed steps. Per-request records (TTFT, TPOT, lengths, errors):
  `tasks/pd_colocation/results_h200_per_tenant_20261019.json.gz`. `bash tasks/pd_colocation/h200_per_tenant_20261019.sh`;
  scoring `tasks/perf_model/bench_per_tenant.py`; sglang `f9673a12d`.
- **Out of scope:** FP8 (weights, activations or KV); FP8 runs in the task docs are exploratory records, not results.

## TraceLab causal replay: the zero-regression check

The headline setting is one H800 with Qwen2.5-7B-Instruct, 400 causal rounds
from ten real Claude coding sessions, 64K–100K contexts, mean 3,812-token
append, mean 481-token output, and recorded think/tool gaps compressed 100x.
All ten sessions are admitted together (`--request-rate inf`). Warmup uses
same-length but token-disjoint prompts, so it exercises kernels and memory
allocation without pre-populating the timed sessions' radix prefixes. The
server seed is fixed to one.

The replay preserves session identity, consecutive round ordering, prefix
growth, tool-result prefills, output lengths, causal dependencies, and the
relative timing of inter-round gaps. It is not a bag of independent prompts.

### Offered-load sweep (honest baselines, 2026-07-31)

For causal TraceLab replay, `--request-rate` is the Poisson arrival rate of
**new sessions**; later rounds remain causal. The sweep uses N=400 with fixed
server seed one, identical sampled rounds, ordinary decode CUDA graphs for
both methods, and token-disjoint warmup. The rates below are sessions/s;
`inf` admits all ten sessions together.

Earlier versions of this sweep (in git history) compared two servers that
both ran the persistent kernel (the DMC launcher never disabled
`SGLANG_FLASHINFER_USE_PERSISTENT`) and both paid the plan()-synchronize
stall. The current sweep ran with sglang
`f158e4b85` / flashinfer `c6d2a00`: true BatchPrefill DMC versus gated LPT
(`SGLANG_MIXED_ATTENTION_KERNEL=auto`, matched 8,192 cap), raw runs in
`h800_qwen25_tracelab_claude_pareto_gated_n400/`. The honest baseline is
itself 6.1–10.7% faster than the contaminated one.

| Rate | Method | Total tok/s | Mean / P99 TTFT | Mean / pooled P99 ITL |
|-----:|--------|------------:|----------------:|----------------------:|
| 0.02 | true DMC | 28,907 | **725 ms** / 7.87 s | 17.83 / 96.12 ms |
| 0.02 | gated LPT | 28,903 | 758 ms / **7.28 s** | **17.73 / 93.54 ms** |
| 0.05 | true DMC | 30,775 | 892 ms / 9.92 s | **18.25** / 98.99 ms |
| 0.05 | gated LPT | **30,794** | **816 ms / 9.55 s** | 18.38 / **98.59 ms** |
| `inf` | true DMC | 31,558 | **1.29 s / 31.0 s** | 18.33 / 100.41 ms |
| `inf` | gated LPT | **31,643 (+0.27%)** | 1.32 s / 33.0 s | **18.25 / 95.83 ms (−4.6%)** |

The gated build is parity or better on throughput and pooled P99 ITL at every
rate (P99 ITL −2.7%/−0.4%/−4.6%); saturation TTFT differences are within the
±6% run-to-run band measured in paired repeats. The gate selected the
persistent kernel for 63/500 (0.02) to 203/1500 (inf) mixed batches. The
figures below are regenerated from this run.

Current corrected TraceLab figures:

- [Capacity versus mean ITL](realistic_workload_analysis/tracelab_current_n400/figures/pareto_capacity_vs_mean_itl.png)
- [Pooled P99 ITL versus session rate](realistic_workload_analysis/tracelab_current_n400/figures/p99_itl_vs_rate.png)
- [TTFT SLO attainment](realistic_workload_analysis/tracelab_current_n400/figures/slo_ttft_attainment_vs_rate.png)
- [Per-request P99-ITL SLO attainment](realistic_workload_analysis/tracelab_current_n400/figures/slo_itl_attainment_vs_rate.png)

The compact source is
[`summary.json`](realistic_workload_analysis/tracelab_current_n400/summary.json);
multi-megabyte per-request JSON and server logs remain local benchmark
artifacts.

## Workload sweet-spot scan

The kernel needs simultaneous long decode and intermittent prefill, not merely
long contexts: long decode makes attention dominant, but persistence helps end
to end only while an intermittent append arrives against enough active decodes
to form repeated mixed forwards. The screen varies real causal session counts
(N=100, output ≥512 tokens, fixed seed, disjoint warmup, 64K–100K contexts).
Configured session count is not itself a controlled offered-load coordinate —
a twelve-session sample reached only 1.57 observed concurrency because its
real windows contained long tool/think gaps. A pre-fix version of this scan
(git history) ran against the contaminated baseline and misplaced the win
boundary at "concurrency ≈ 11".

### Gated-stack sweet-spot rescan (2026-07-31, honest baselines)

The scan was repeated at g12/g14/g18/g20 with the fixed stack (sync-free
plan, kernel gate, prewarm; sglang `f158e4b85`, flashinfer `c6d2a00`),
true-BatchPrefill DMC, gated LPT at the tail-optimized 6,656 cap, plus the
earlier validated g10/g16 screens. Results dirs:
`h800_tracelab_decodeheavy512_gated_g*_n100/`.

| Groups | Obs. conc | Mean ITL regime | LPT tput | pooled P99 ITL | reqmax P99 | worst ITL (LPT / DMC) |
|-------:|----------:|-----------------|---------:|---------------:|-----------:|----------------------:|
| 12 | 1.4 | 17.8 ms (light) | −0.07% | **+72%** (82 vs 47 ms) | −14% | 0.88 / 1.02 s |
| 10 | 3.2 | 17.5 ms (light) | −0.11% | +11.6% | −18% | 0.89 / 1.08 s |
| 14 | 2.7 | 44 ms (heavy) | −0.20% | **−15.7%** | −14% | 0.87 / 1.02 s |
| 18 | 8.2 | 52 ms (heavy) | **+0.64%** | **−16.0%** | **−44%** | **0.89 s / 53 s** |
| 16 | 10.9 | 43 ms (heavy) | **+0.77%** | **−14.6%** | −12% | 1.05 / 32 s |
| 20 | 14.5 | 47 ms (heavy) | −0.39% | **−15.7%** | **−67%** | **0.89 s / 82 s** |

The discriminator is the **mixed-step interference regime**, not raw
concurrency: whenever the sample keeps mixed chunks colliding with decodes
(mean ITL ≈ 43–52 ms; g14/g16/g18/g20), the gated stack delivers a uniform
−15 to −16% pooled P99 ITL at parity-to-+0.8% throughput, cuts request-max
tails by up to 67%, and replaces DMC's 32–82-second worst stalls with
sub-second worst gaps. In light-mixing samples (mean ITL ≈ 17.5 ms;
g10/g12), the 6,656 cap costs pooled P99 (absolute 35–82 ms, still under
the 150 ms SLO) while request-max stays 14–18% better; the matched 8,192
cap restores strict parity there (validated at g10: pooled P99 −6.2%).

The sweet spot is therefore g16–g18 (concurrency ~8–11 under heavy mixing):
simultaneous throughput, pooled-tail, and worst-stall wins. Notably the win
region is much broader than the pre-fix scan suggested — g14 (conc 2.7) and
g20 (conc 14.5) also win the tail decisively; the old "only wins near
concurrency eleven" boundary was mostly plan-sync overhead and the
contaminated baseline. Gate picks stay modest everywhere (1–4% of mixed
batches persistent at long context), so the tail gains come from the gate
keeping BatchPrefill where it is faster while the cap shapes interference.

## Transfer to independent coding datasets

Matched N=400 sweeps use Qwen3-8B and ordinary decode CUDA graphs for every
method (Code-Feedback 4–24 req/s plus saturation; SWE-Gym 0.5–4 req/s plus
saturation; raw runs in `tasks/edf_benchmark/results_valid_n400/`). DMC and
LPT land within fractions of a percent of each other on throughput and mean
ITL at every rate — parity, not a win, and expectedly so.

These datasets contain independent prompts averaging only about 242 and 690
tokens. They omit the multi-turn intermittent tool-result prefills that create
repeated mixed forwards in TraceLab, so the persistent kernel is active during
too small a fraction of server time to materially shift the full curve.

## Mixed-tenancy workloads (2026-08-05): the win scenario, end to end

`run_mixed_workloads.py --scenario agent_plus_shortchat`: one H800 serving a
coding-agent tenant (12 causal TraceLab sessions @32–48K, rounds issued
back-to-back, 60 requests) beside a chat tenant (2,400 fresh 512-token
prompts at 12 req/s, 96-token outputs). True-BatchPrefill DMC baseline vs the
gated persistent path (merged 1,316-point table with the KV-concentration
feature). Identical token volumes, 100% retokenization on every stream.

| Tenant | Metric | true DMC | gated LPT | delta |
|--------|--------|---------:|----------:|------:|
| codex  | total tok/s | 13,809 | **18,844** | **+36.5%** |
| codex  | wall time   | 188 s | **138 s** | −27% |
| codex  | mean ITL    | 43.4 ms | **40.5 ms** | −6.7% |
| chat   | mean ITL    | 27.3 ms | **15.8 ms** | **−42.2%** |
| chat   | P99 ITL     | 80.1 ms | **61.7 ms** | −23.0% |
| chat   | mean TTFT   | 38.2 s | **24.1 s** | −37% |

Mechanism, confirmed by full-run nsys captures of both methods: mixed
forwards are 14–30% of all forwards in this scenario, and the gated path
spends 112 s of mixed-step wall time where DMC spends 157 s (DMC's worst
mixed step stalls 13.9 s; the persistent kernel's per-instance max is
0.9 ms). The coding tenant converts that directly into throughput; the chat
tenant stops waiting behind slow mixed steps.

**In-band multi-turn + SLO control (2026-08-11).** `agent_plus_multiturn`
(2K follow-up turns on 8K shared prefixes over the coding pool) drives 49% of
forwards mixed — the balanced band. At deliberately saturated load (all arms
queue 18–77 s of TTFT), adding the closed-form-model EDF controller
(`LPT_EDF`: `SGLANG_EDF_LATENCY_MODEL=analytic`, ITL SLO 100 ms) pins P99 ITL
almost exactly at the budget it was told to spend — codex 497→113 ms (−77%),
turns 258→107 ms (−59%), turns per-request SLO attainment 1.3%→54.8% — at the
expected cost of TTFT/throughput for the deferred prefill (the correct
tradeoff beyond capacity; no controller can also absorb an infeasible queue).
The controller's guard is set to 0 for the analytic model, which already
carries fixed overheads in `c0`; the adaptive 5–10 ms guard predates the
plan-sync fix and would double-count. The feasible-load variant
(`agent_plus_multiturn_feasible`) replicates the controller property at a
second operating point — pooled P99 ITL pinned at 104–105 ms against the
100 ms budget (codex −77%, turns −51%; codex request-max P99 645→150 ms) at
identical node-wide throughput (23,319 vs 23,317 tok/s) — but tenant-level
TTFT/attainment comparisons there are confounded: the codex tenant runs at
rate=inf, and with an infinite-backlog prefill source present, FCFS
admission ordering rather than compute dominates the other tenant's latency
(LPT's turns TTFT moves 0.7→6.6 s at identical node throughput and identical
turns tput). The clean experiment (`multiturn_slo_clean`, both streams rate-limited,
2026-08-11) closes the loop: with identical arrivals and node throughput
matched to four digits across all three arms (9,368/9,369/9,368 tok/s),
the gated kernel alone improves codex pooled P99 ITL −28.3% (81→58 ms,
turns −8.1%) at parity elsewhere, and adding the analytic-model controller
pins turns pooled P99 at **100.24 ms against the 100 ms budget** while
cutting request-max P99 ITL **−60% on both tenants** (519/522→208/208 ms),
with unchanged throughput. Across the three operating points the achieved
pooled P99 converges onto the configured budget (113 → 105 → 100.2 ms) as
load moves from saturated to feasible — the controller delivers exactly the
tail it is asked for whenever the load permits, which is the operational
meaning of a 3%-accurate cost model. A recurring
plain-LPT turns anomaly (request-max 1.4 s, retok 98.9%) matches the
retraction-cascade signature — third sighting.

The `shortchat_only` control (3,000 fresh 512-token prompts at 16 req/s, no
long-context tenant) lands at parity — −0.02% throughput, −0.45% mean ITL —
with the gate selecting persistent for only 6 of 2,566 mixed batches. The
gate takes the win where the composition supports it and declines everywhere
else, which is the zero-regression contract.

One LPT trial (archived in `h800_mw_agent_plus_shortchat/run1_regressed/`)
showed a −19% episode whose damage was concentrated in the first two
arrival-deciles of the codex prefill storm (chat ITL 98.7 ms early vs
15–16 ms steady-state) with 98.5% retokenization, consistent with a
KV-retraction cascade compounding at fixed arrival rate. The identical
configuration reproduces the win in repeat trials and under nsys; the
baseline is stable across all trials. Multi-trial runs are the suite's
default defense against episode-vs-property confusion.

- [Capacity versus mean ITL](edf_benchmark/current_policy_n400/pareto_capacity_vs_mean_itl.png)
- [Pooled P99 ITL versus offered rate](edf_benchmark/current_policy_n400/p99_itl_vs_rate.png)
- [TTFT SLO attainment](edf_benchmark/current_policy_n400/slo_ttft_attainment_vs_rate.png)
- [Per-request P99-ITL SLO attainment](edf_benchmark/current_policy_n400/slo_itl_attainment_vs_rate.png)

TTFT and ITL attainment are kept separate. The continuous P99-ITL and
capacity/interactivity curves are the primary evidence; SLO attainment is a
threshold-sensitive diagnostic. Pooled P99 ITL is accompanied by
request-maximum ITL because pooled token percentiles hide rare multi-second
stalls.

## Which attention kernel wins, and does our workload answer that? (2026-09-27)

Full kernel-level detail (planner structure, isolated-composition
measurements, FA3 levers tried) is consolidated in
`tasks/debug_kernel_server_perf.md` → "Ours vs FA3: the complete picture"; this
section carries the serving-level verdict only.

Kernel-level replay of the realized mixed-batch distribution
(`sweep_from_capture.py`, time-weighted over 2,845 real batches) gives
**FA2 1.00× / FA3 0.67× / persistent 2.60×**. That reads as a decisive win for
the persistent kernel, but it is a statement about *this* workload, and the
workload has a bias that matters.

**The bias: our captured chunks do almost no prefill work** — though *not*
because the cache is missing. Those runs hit the radix cache **79.7%** in token
terms with peak KV occupancy 0.61 (no eviction). The per-row distribution is
bimodal: the coding tenant's continuations hit heavily but are few rows (3 of
43 sampled mixed batches carry `cached-token > 512`, and they hold 3.8M of the
cached tokens), while the chat tenant's fresh random prompts — which have no
reusable prefix by construction — dominate the row count. Hence attended prefix
2 tokens at p50, **94% of rows below 512**, and a median chunk doing ~550
token-pairs of attention, ~**30,000×** less than a 2,048-token chunk against an
8K prefix. It is a property of the scenario's tenant mix, not of caching and
not of serving in general. Splitting the same replay by attended prefix shows
the comparison inverting as prefill work appears:

| attended prefix | weight | FA3/FA2 | persistent/FA2 | persistent/FA3 |
|-----------------|-------:|--------:|---------------:|---------------:|
| < 512 (fresh) | 2,793 | 0.63× | **3.95×** | **6.26×** |
| ≥ 8K | 52 | 0.95× | 1.02× | **1.08×** |

In the long-prefix band all three kernels are within 8% of each other — but it
carries only 2% of the weight here, so the headline 2.60× is set almost
entirely by the fresh-prefix band. Measured directly on prefill-dominant
batches, **FA3 is 1.35–1.54× faster than FA2 while the persistent kernel is
0.89–1.00× of FA2** (`tasks/debug_kernel_server_perf.md`, "FA3 on
prefill-dominant batches"). So the ordering genuinely reverses by regime:

- **decode-heavy, fresh/short chunks** → persistent wins, by a lot (3.2–5.1×)
- **prefill-dominant, long attended prefix** → FA3 wins (1.35–1.54×), ours is
  at parity-or-worse than FA2
- **crossover** around a handful of decode rows against a real chunk

**Verdict, measured on a workload that does have prefix reuse.** The above is
settled by capturing `agent_plus_multiturn_feasible` (multi-turn on 8K shared
prefixes + agent sessions, at the SLO-feasible operating point) and replaying
it: 409 mixed batches with **attended prefix p50 8,592, 56% ≥ 8K, chunk p50
2,598, 18.6M token-pairs of prefill work per chunk** — 34,000× the fresh
capture — alongside a live decode population (rows p50 12, p90 47) and KV
concentration 28.0K. Time-weighted over 365 replayed batches:

| policy | total kernel ms | vs FA2 |
|--------|----------------:|-------:|
| FA2 | 958,499 | 1.00× |
| **FA3** | 1,410,339 | **0.68×** |
| **persistent** | 713,424 | **1.34×** |
| deployed gate | 813,294 | 1.18× |
| oracle per-batch best-of-3 | 664,749 | **1.44×** |

**The persistent kernel still wins overall — but by 1.34×, half the 2.60× it
showed on the fresh-prefix workload.** Adding real prefill work closes most of
the gap, exactly as expected. FA3 remains 0.68× in aggregate because these
batches still carry a decode population.

The decode-row split is where the answer actually lives:

| split | weight | FA3/FA2 | persistent/FA2 | persistent/FA3 |
|-------|-------:|--------:|---------------:|---------------:|
| 1–3 decode rows | 74 | **1.39×** | 0.98× | **0.70×** |
| 4–15 rows | 127 | 1.30× | **1.48×** | 1.14× |
| 16–47 rows | 164 | 0.55× | **1.34×** | 2.43× |
| prefix < 512 | 76 | 0.52× | 3.26× | 6.22× |
| prefix 512–8K | 83 | 0.53× | 1.43× | 2.68× |
| prefix ≥ 8K | 206 | 0.78× | 1.20× | 1.55× |

**FA3 wins outright on prefill-dominant batches** — at 1–3 decode rows it is
1.39× over FA2 and **1.43× over ours** (persistent/FA3 = 0.70×). The crossover
is around **four decode rows**; above it ours wins, decisively so past 16.
FA3's aggregate loss is therefore not a statement that FA3 is a weak kernel, it
is that this workload spends most of its time in batches with many decode rows.

Two consequences. First, **the honest claim is regime-scoped**: ours wins the
mixed-batch regime a busy colocated node actually occupies, FA3 wins the
prefill-dominant corner. Second, **a three-way gate is worth building**: the
oracle best-of-3 is 1.44× against 1.34× for always-persistent, so ~7% is left
on the table by not considering FA3 at all — and the deployed gate only
realizes 1.18×, so there is more headroom in selection than in any single
kernel.

**Does our workload meet the "realistic mixed workload" bar?** The co-tenancy
capture alone does not; the prefix-reuse capture does. Both are needed:

- *Realistic in composition* — co-tenancy of a coding agent with chat is a real
  deployment, the request stream is a real trace, and the decode-side mixture
  (46 rows, high KV concentration) is what a busy node looks like.
- *Not realistic in prefix reuse per row* — 94% of prefill rows attend <512
  tokens (the chat tenant has no reusable prefix by construction), which is
  exactly the regime where FA3 cannot earn its tile. Note this is a row-mix
  effect, not a cache failure: token-level hit rate is 79.7%. A production node with radix-cache hits or multi-turn traffic
  sits much further right on the prefix axis. This is the same grid-vs-realized
  sampling problem found earlier, now on the prefix axis rather than the chunk
  axis.
- *Fixed by the second capture* — `agent_plus_multiturn_feasible` supplies the
  missing half (prefix p50 8,592, 56% ≥ 8K, SLO-feasible arrival rates), and it
  is the one that should be quoted for kernel-selection claims. Reporting only
  the fresh-prefix number overstates the persistent kernel by ~2× (2.60×
  vs 1.34×).

**Under common SLOs the kernel choice matters less than these ratios suggest.**
The gate-v3 A/B is the cleanest evidence: a **1.90× → 3.50×** improvement in
kernel policy on mixed batches produced **+0.0% node throughput**, moving only
tails (chat P99 ITL −43.1%; codex P99 +7.2%, inside the ±6% run-to-run band).
Mixed batches are a minority of forwards and attention a fraction of step time,
so kernel wins show up in the ITL tail — where an ITL SLO is actually spent —
rather than in capacity. A kernel-selection decision aimed at an SLO should
therefore be made on tail latency in the regime the node actually occupies, not
on aggregate kernel speedups.

## Reproduction

```bash
# Headline replay
bash tasks/realistic_workload_analysis/run_tracelab_claude.sh

# Offered session-rate sweep
bash tasks/realistic_workload_analysis/run_tracelab_pareto.sh

# Decode-heavy peak-load screen (real causal TraceLab sessions)
TRACELAB_MIN_OUTPUT_LEN=512 TRACELAB_NUM_GROUPS=16 \
  NUM_PROMPTS=100 LPT_CHUNKED_PREFILL_SIZE=6656 \
  bash tasks/realistic_workload_analysis/run_tracelab_claude.sh

# Analyze and plot accepted results
python tasks/edf_benchmark/analyze_results.py RESULTS_DIR
python tasks/edf_benchmark/plot_serving_tradeoffs.py SUMMARY.json OUTDIR

# Queue versus first-token service decomposition
python tasks/realistic_workload_analysis/analyze_request_timing.py \
  RESULTS_DIR/server_DMC_CG.log RESULTS_DIR/server_LPT.log

# Mixed-tenancy scenarios (win scenario, controls, SLO A/Bs)
python tasks/realistic_workload_analysis/run_mixed_workloads.py \
  --scenario agent_plus_shortchat            # or shortchat_only,
                                             # agent_plus_multiturn{,_feasible},
                                             # multiturn_slo_clean
python tasks/realistic_workload_analysis/summarize_mixed_workloads.py OUTDIR \
  --baseline DMC_CG --method LPT             # or --method LPT_EDF
```

## Final conclusions

1. **Zero regression where the kernel cannot win.** Pure long-context
   TraceLab replay — every chunk attends a 32–100K prefix — is
   BatchPrefill's territory; the gated stack holds parity-or-better on
   throughput and pooled P99 ITL at every offered rate, and the
   `shortchat_only` control lands at parity with the gate declining
   persistent on 2,560/2,566 mixed batches. The gate, not scheduling
   restraint, is what makes the comparison safe by construction.
2. **The win scenario is co-tenancy**: fresh, short-prefix prefills meeting
   a long-context decode pool (chat + coding agent on one node). Codex
   tenant +36.5% throughput, chat tenant −42% mean ITL / −37% TTFT at
   identical token volumes; nsys attributes it to mixed-step wall time
   (157 s → 112 s; worst single mixed step 13.9 s → 0.9 ms).
3. **Predictability is the fused kernel's second product.** With the
   analytic cost model (`T_attn = max(T_p,T_d) + 0.53·min`, 3.1% MAPE) the
   EDF controller spends an ITL budget exactly: across three operating
   points the achieved pooled P99 converges onto the configured 100 ms
   budget (113 → 105 → 100.24 ms) as load moves from saturated to feasible,
   with request-max P99 −60% on both tenants at unchanged throughput.
4. **Occupancy, not context length, decides.** The kernel matters only
   while mixed forwards recur: heavy-mixing regimes in the sweet-spot
   rescan give a uniform −15 to −16% pooled P99 ITL and replace 32–82 s
   worst stalls with sub-second ones, while independent short-prompt
   datasets are structurally at parity (too few mixed forwards).

### Evaluation-hygiene lessons

- **Baseline contamination**: `SGLANG_FLASHINFER_USE_PERSISTENT` defaults
  on; every baseline launcher must disable it explicitly. The honest
  BatchPrefill baseline is itself 6.1–10.7% faster than the contaminated
  one, so contamination understated both sides.
- **Exact-prefix warmup** was the primary early evaluation bug: cache
  residency and eviction order dominated first-cohort admission and
  flipped apparent TTFT winners. Warmup is now length-matched but
  token-disjoint from every timed prefix.
- **A kernel speedup and a TTFT speedup are not interchangeable.**
  Request-timing decomposition (queue delay vs admission-to-first-token
  service) showed service time unchanged while queue time moved ~9–11%:
  near saturation, small service or completion-order changes drain the
  queue nonlinearly and the sign is phase-sensitive. Attribute TTFT
  changes to queue vs service before crediting the kernel.
- **Pooled token-level P99 hides rare multi-second stalls**; request-max
  P99 ITL is reported alongside it everywhere. A tenant at
  `--request-rate inf` makes FCFS admission ordering dominate the other
  tenant's latency, confounding attainment comparisons — clean SLO A/Bs
  keep every stream rate-limited.
- **Fair-baseline rules**: no minimum prefill chunk was introduced; DMC
  retains its stock kernel, 8,192-token cap, and CUDA graphs. Multi-trial
  runs are the default defense against episode-vs-property confusion
  (the −19% retraction-cascade episode reproduced nowhere in repeats).

## Context length for the kernel A/B — 2026-10-06

**Decision (user, 2026-10-06): the A/B must run at a context length where attention is
>50% of end-to-end time, by one of two routes.**

1. **Force-extrapolate RoPE** up to TraceLab's maximum session length. Normal agentic
   sessions (Claude/Codex-style) reach 300-400k tokens and stay under 1M, so extrapolating
   beyond the trained window is a fair simulation of the real regime even though
   perplexity is not meaningful there -- this measurement is about attention cost, not
   output quality.
2. **Or use a larger model with a natively longer context**, if extrapolation muddies the
   result.

**Why this matters, and why earlier runs could not show the bar.** At
`--random-input-len 2048` attention is a small fraction of the forward, so an attention-only
speedup is invisible end to end: a 1.21x kernel win cannot produce a >=30% TTFT/ITL gain
there no matter how correct it is. Those runs were measuring in a regime where the bar is
unreachable by construction. At 64k+ attention dominates, and at 300-400k it is >50% of
e2e, where the same kernel win translates directly.

**Also decided: use the TraceLab-derived mixed scenarios, not `--dataset-name random`.**
`run_mixed_workloads.py` has the real shapes (`agent_plus_shortchat`, `loadsweep_*`), whose
multi-turn tool-result prefills create the repeated mixed forwards this kernel exists for.
Independent-prompt random datasets omit exactly that structure -- this doc already records
that they average 242-690 tokens and cannot show the effect. A `SPLIT` method is now wired
into that driver.

