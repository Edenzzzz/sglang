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
- [x] **Six-repeat standard-mix result (2026-10-20, H200 `dadf94f0e1da7238`; every repeat 0 tracebacks / 0 fallbacks;
  ours no MTP 71.5-72.5K split calls, all overlapped; ours + MTP 33.5-34.5K split calls).** Mean +- std over repeats
  (per-repeat values in `tasks/perf_model/bench_reps_summary.py` output); attainment = TTFT and per-request TPOT both
  met, agent 10 s + 80 ms, chat 0.5 s + 20 ms. Stock mixed chunk ran its slow mode in 5 of 6 repeats (agent TTFT
  4.65 +- 0.89 s), so its mean includes both modes; prefill-prioritized stock is stable (2 repeats).
  | arm | agent tput | agent TTFT | agent TPOT | agent att. | chat TTFT | chat TPOT | chat att. |
  |---|---|---|---|---|---|---|---|
  | stock mixed chunk (6) | 80.5 +- 4.2 | 4.65 +- 0.89 s | 77.5 ms | 44.1 +- 6.7% | 2.07 s | 38.0 ms | 53.8 +- 2.1% |
  | stock prefill-prioritized (2) | 92.6 | 2.42 s | 76.6 ms | 58.9% | 0.86 s | 34.8 ms | 59.9% |
  | **ours, no MTP (5)** | **113.2 +- 0.4** | 3.40 +- 0.08 s | **52.1 ms** | **92.0 +- 1.8%** | **0.14 s** | **20.7 ms** | **69.8 +- 0.1%** |
  | stock + MTP (6) | 65.7 +- 3.1 | 20.23 +- 1.28 s | 29.2 ms | 3.0% | 12.54 s | 17.2 ms | 34.2 +- 3.2% |
  | stock prefill-prioritized + MTP (2) | 81.6 | 15.47 s | 29.8 ms | 8.9% | 7.67 s | 16.6 ms | 49.2% |
  | **ours + MTP (5)** | **102.1 +- 1.3** | **12.54 +- 0.16 s** | **18.6 ms** | **45.8 +- 1.8%** | **2.94 s** | **13.0 ms** | **58.9 +- 0.8%** |
  vs stock mixed chunk: no MTP agent +41% / TTFT -27% / TPOT -33% / attainment +109%, chat TTFT -93% / TPOT -46% /
  attainment **+29.7%**; MTP +55% / -38% / -36% / 15x, chat -76% / -24% / **+72%**. vs prefill-prioritized: no MTP
  +22% / TTFT **+40%** / -32% / +56%, chat -84% / -41% / **+16.5%**; MTP +25% / -19% / -38% / 5x, chat -61% / -22% /
  **+20%**. **Against mixed chunk the attainment bar clears with MTP and sits at the line without it; against
  prefill-prioritized chat attainment falls short in both pairs.** Our arms vary by <= 1.4 tok/s and <= 1.8 points
  attainment across repeats. `bash tasks/pd_colocation/h200_per_tenant_20261019.sh`, `h200_more_reps_20261019.sh`,
  `h200_pp_and_trace_20261019.sh`; sglang `09aadc008`.
- [x] **Single-node per-row slack decode-first (2026-10-20, H200; negative).** The decode-first rule budgeting per running
  row against its mean-TPOT SLO (chat 20 ms, agents >= 24K prompt 80 ms; measured decode base; `ITL_SLO=...:measured:
  slack:cls24000_80`, `h200_slack_20261020.sh`) starves prefill on one node too: ours without MTP agent 50.4 tok/s /
  TTFT 30.19 s / ITL 18.9 ms, chat TTFT 18.99 s / ITL 16.7 ms (vs 113 tok/s / 3.4 s without the rule). Most caps were
  0: with ~2400 chat requests some running row is almost always slightly behind its SLO, and the rule then admits no
  prefill until it catches up. Stopped after the first arm. Single-node chat attainment is pursued instead with smaller
  deadline chunks (`h200_chunk_trade_20261020.sh`), trading part of the agent margin (92% vs 59% attainment over
  prefill-prioritized) for chat TPOT.
- [x] **Chunk trade (2026-10-20, H200; negative).** Deadline chunk 512 / 1024 (chunk 1024), one repeat each; per tenant
  (agent 10 s + 80 ms, chat 0.5 s + 20 ms): no MTP -- chat 69.3% (vs 69.8% at 1024 / 2048), chat TPOT p90 50.1 ms (vs
  ~61), agent 80.0% (vs 92.0%), agent TTFT 4.72 s; MTP -- chat 57.1% (vs 58.9%), agent 28.9% (vs 45.8%), agent TTFT
  14.37 s. A 512-token chunk at ~50K depth still costs ~23 ms of attention beside the ~9 ms decode step, and agent
  prefill runs almost continuously, so ~30% of chats keep a mean TPOT above 20 ms; holding them under it would need
  chunks below ~200 tokens. The 256 arms were stopped. **On one node at this load the chat-attainment gap to
  prefill-prioritized stock (+16-20%) is structural**: chats decode beside continuous long-context agent prefill.
- [x] **Every MTP arm so far was concurrency-capped at 48 running requests (found 2026-10-20; the cap is the better
  operating point -- see the cap-128 result at the end of this item).** sglang derives
  `max_running_requests` from the speculative CUDA-graph batch size: ours + MTP, stock + MTP and prefill-prioritized
  + MTP all ran with 48 and hit it (ours: queue mean 29.6, max 105, one new request admitted per step while 30-90
  wait); the no-MTP arms run uncapped (4096) and peak at 115 running. Chat misses split accordingly (TTFT > 0.5 s /
  TPOT > 20 ms): ours no MTP 3.0-3.2% / 30%; ours + MTP 36-39% / 23-25%; stock + MTP 67-68% / 32%. So the MTP
  arms' chat attainment was set by the cap, for ours and the baselines alike. Rerun with the same raise for every
  MTP arm, `WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs 128"`, two repeats
  (`h200_mtp_cap128_20261020.sh`; the first launch never started -- this sglang splits `--cuda-graph-max-bs` into
  `-decode` / `-prefill`, rerun with `--cuda-graph-max-bs-decode 128`, `max_running_requests: 128` confirmed in the log).
  **Cap 128 is worse for every MTP arm** (rep 1, clean): stock + MTP 32.8 tok/s / TTFT 32.45 s / TPOT 77.6 ms, chat
  39.65 s / 92.1 ms, chat throughput below offered; prefill-prioritized + MTP 31.6 / 41.37 s / 84.7 ms, chat 49.16 s /
  101.3 ms; ours + MTP 57.0 / 9.58 s / 40.1 ms, chat 1.82 s / 55.1 ms (vs 102.1 / 12.54 s / 18.6 ms, chat 2.94 s / 13.0
  ms at 48). Up to 128 rows verifying 4 draft tokens each make every step slower, so decode slows for everyone and the
  queue grows anyway: the cap of 48 protects MTP serving. The cap-48 MTP comparison stands, and the chat-attainment
  gap to prefill-prioritized stock is not a configuration artifact. Rep 2 stopped.
  **Cap 64 (2026-10-20, `h200_mtp_capsweep_20261020.sh`, one repeat each, clean)** is already worse than 48 for every
  arm: stock + MTP 36.6 tok/s / TTFT 33.42 s / TPOT 48.5 ms, chat 41.54 s / 42.6 ms; prefill-prioritized + MTP 37.1 /
  32.88 s / 52.2 ms, chat 42.67 s / 44.2 ms; ours + MTP 60.4 / 18.97 s / 31.7 ms, chat 10.90 s / 27.9 ms. 48 is the best
  cap for each system, so the cap-48 MTP comparison is each system at its best; cap 80 not run.
- [x] **Full MTP configuration: rows + SLO admission + FA2 overlapped beside FA3 (2026-10-20, H200, chat 8 req/s;
  `h200_rows_ovl_20261020.sh`, `h200_rows_ovl_more_20261020.sh`; four repeats, clean; every split call overlapped --
  29.0K / 29.5K).** Agent 104.0 / 104.3 / 101.8 / 102.6 tok/s, attainment 46.7 / 55.6 / 44.4 / 44.4%; chat TTFT
  2.17 / 2.14 / 2.44 / 2.19 s, TPOT 13.6 / 15.1 / 15.1 / 14.2 ms, attainment 62.9 / 63.7 / 62.3 / 62.9% (mean 63.0%,
  the same as without overlap): the verify batch is too small beside a deep-context FA3 chunk for the overlap to hide
  anything. vs prefill-prioritized + MTP (49.3%) **+27.7%**. With the page-128, acceptance-3.0, overlap and util
  variants all measured, chat attainment at 8 req/s settles at 62-64% (+26-30%) for every variant of ours.
- [x] **Single-node table rerun with chat-templated chat prompts, real heads only (2026-10-21, H200, standard mix, chat 8
  req/s, two repeats each, all clean; `bash tasks/pd_colocation/h200_chattemplate_20261021.sh`, CHAT_TEMPLATE=1; the
  TraceLab agent stream replays recorded token streams and is unchanged).** Logged accept len 2.02-2.03 in every MTP arm
  -- the chat template does not move this head's acceptance. Means of the two repeats, per tenant (agent 10 s + 80 ms,
  chat 0.5 s + 20 ms):
  | arm | agent tput | agent TTFT | agent att. | chat TTFT | chat TPOT | chat att. |
  |---|---|---|---|---|---|---|
  | stock mixed chunk | 78.8 | 4.91 s | 35.6% | 2.29 s | 38.8 ms | 52.9% |
  | stock prefill-prioritized | 89.2 | 2.76 s | 55.6% | 1.05 s | 40.5 ms | 58.2% |
  | **ours, no MTP** | **113.8** | 3.34 s | **92.2%** | **0.14 s** | **20.6 ms** | **69.8%** |
  | stock + MTP | 64.8 | 20.63 s | 2.2% | 12.94 s | 17.2 ms | 33.2% |
  | stock prefill-prioritized + MTP | 75.9 | 17.06 s | 4.4% | 9.07 s | 16.8 ms | 44.0% |
  | **ours + MTP (rows + SLO admission)** | **101.3** | **11.40 s** | **42.2%** | **2.37 s** | **15.2 ms** | **62.7%** |
  With MTP vs prefill-prioritized + MTP: chat attainment **+42%**, agent ~10x, agent throughput +33%, chat TTFT -74%,
  agent TTFT -33%; vs stock + MTP chat +89%, agent 19x, throughput +56%. Without MTP vs prefill-prioritized: chat +20%,
  agent +66%, throughput +28%, chat TTFT -87%, TPOT -49%; vs mixed chunk chat +32%, throughput +44%. With templated chat
  prompts the prefill-prioritized + MTP baseline drops from 49.3% to 44.0% chat attainment (its chat TTFT misses rise),
  so the MTP point that sat at +28% now clears +30%.
- [x] **Qwen3-8B at the Qwen2.5 load -- overloaded for every system (2026-10-21, H200, trained EAGLE3 head AngelSlim/
  Qwen3-8B_eagle3, accept 2.00-2.09, chat template on, YaRN to 128K; two repeats, all clean; ours + MTP 78K split calls;
  `h200_next_20261021.sh`).** Agent C=8 + chat 8 req/s exceeds Qwen3-8B's single-H200 capacity (36 layers, 8 KV
  heads): no-MTP chat attainment ours 24.5% vs prefill-prioritized 10.8% / mixed chunk 0.7% (agent 69% vs 31% / 21%,
  throughput 55.6 vs 47.2 / 42.4); every MTP arm collapses (chat TTFT 100-170 s, 0% chat attainment; agents ours 13% vs
  7.8% / 10%). Not a valid comparison point; rerun at half load below.
- [x] **Qwen3-8B at half load (2026-10-21, H200, agents C=4 over 12 sessions + chat 4 req/s, chat template, trained
  EAGLE3 head; two repeats, 0 tracebacks; the stock no-MTP rep 1 arms wrote no records, so those rows are rep 2 only;
  `bash tasks/pd_colocation/h200_qwen3_half_20261021.sh`).** Per tenant (agent 10 s + 80 ms, chat 0.5 s + 20 ms):
  | arm | agent tput | agent att. | chat TTFT | chat TPOT | chat att. |
  |---|---|---|---|---|---|
  | stock mixed chunk | 52.8 | 81.8% | 0.86 s | 28.4 ms | 59.9% |
  | stock prefill-prioritized | 58.8 | 77.3% | 0.50 s | 26.0 ms | 63.5% |
  | **ours, no MTP** | **68.0** | **86.4%** | **0.07 s** | **19.1 ms** | **68.1%** |
  | stock + MTP (accept 1.79-1.85) | 17.4 | 4.5% | 59.2 s | 66.7 ms | 0% |
  | prefill-prioritized + MTP (1.83) | 15.3 | 18.2% | 88.9 s | 80.4 ms | 0% |
  | ours + MTP (1.82) | 29.4 | 18.2% | 18.0 s | 32.0 ms | 21.3% |
  Without MTP ours leads on throughput (+16% / +29%) and chat latency (TTFT -86%, TPOT -27% vs prefill-prioritized)
  but chat attainment only +7% / +14%. **With this head MTP is a loss for every system** (accept ~1.8 tokens per
  verify on this workload; the verify and draft passes cost more than they save), so for Qwen3-8B the no-spec
  comparison is the meaningful one.
- [x] **Bar status, one GPU, chat 8 req/s -- accepted as a stated limitation (user, 2026-10-21).** Chat SLO attainment
  (TTFT 0.5 s and per-request TPOT 20 ms both met) 63.0% for ours + MTP vs 49.3% for prefill-prioritized stock + MTP, a
  +28% relative gain (63.0 / 49.3) against the +30% bar (64.1% needed); four repeats, 61-65%. Agents clear (5.5x
  attainment), throughput +27%. Every other single-node load (chat 4 / 6 req/s) and every multi-node comparison clears.
- [x] **EDF base deadline at the chat SLO (TTFT_SLO_MS=500 instead of 2000) with rows + SLO admission (2026-10-21,
  H200, chat 8 req/s, two repeats, clean, 29.0K / 29.5K split calls; `bash
  tasks/pd_colocation/h200_rows_ttft500_20261021.sh`) -- no change.** Chat attainment 63.3 / 61.8% (TTFT misses 32.6 /
  32.8%), agent 37.8 / 44.4%. The chat TTFT misses are not ordering: they are the requests SLO admission holds back while
  long agent-chunk steps run. Across rows, admission, util, overlap, page size, acceptance, chunk caps and deadline
  base, chat attainment at 8 req/s stays at 61-65% (+24-32% over prefill-prioritized + MTP; four-repeat mean +28%).
- [x] **Rows + SLO admission + the per-row slack chunk cap (2026-10-21, H200, chat 8 req/s, one repeat each, clean;
  `bash tasks/pd_colocation/h200_rows_slack_20261021.sh`) -- negative.** The cap (largest chunk keeping every running row
  within its mean-TPOT SLO, `ITL_SLO=20:2.0:...:measured:slack:cls24000_80`) does what it is for -- chat TPOT misses
  30% -> 3.8% -- but binds on 86% of steps (6028 / 7001) and starves prefill: chat TTFT misses 57%, chat attainment
  42.9%, agent 13.3% (TTFT 24 s). Letting waiting short prompts through it (`apply_itl_slo_short_floor.py`, floor 1024
  tokens) changes nothing (chat 43.4%, agent 11.1%, cap on 88% of steps): late agent requests hold the earliest EDF
  deadlines, so the cap is sized on them and the adder serves them first. A per-row TPOT cap and deadline order pull
  against each other once agents run late; rows + admission without the cap (63.0%) stays the configuration.
- [x] **Rows + SLO admission without EDF (2026-10-21, H200, chat 8 req/s, FCFS 4-mode MTP at chunk 2048, two repeats,
  clean; `bash tasks/pd_colocation/h200_rows_noedf_20261021.sh`) -- negative.** Chat attainment 21.6 / 22.8% (TTFT 21.5
  / 20.5 s), agent 6.7 / 8.9% (TTFT 26.4 / 25.4 s), agent 68-70 tok/s; rows split on 1 step. Without deadline order the
  SLO admission check holds the queue head, and chats wait behind long agent prefills in arrival order; FCFS alone
  (ablation) was 59-62% chat. Admission needs EDF ordering.
- [x] **Acceptance sensitivity: MTP at 3.0 accepted tokens per verify (2026-10-20, H200, standard mix, chat 8 req/s;
  `SGLANG_SIMULATE_ACC_LEN=3.0` for every arm -- logged accept len 3.00, the head still drafts, the verify still runs,
  output lengths fixed by the drivers; `bash tasks/pd_colocation/h200_sim_acc3_20261020.sh`; two repeats, clean).**
  Our head measures ~2.1; 3.0 is nearer what trained EAGLE3 heads reach on chat. Per tenant, rep 1 | rep 2:
  | arm | agent tput | agent att. | chat TTFT | chat TPOT | chat att. |
  |---|---|---|---|---|---|
  | ours + MTP (rows + SLO admission) | 113.3 / 116.6 | 60.0 / 57.8% | 1.08 / 0.98 s | 10.1 / 9.8 ms | 69.0 / 69.8% |
  | prefill-prioritized + MTP | 89.3 / 89.2 | 24.4 / 22.2% | 5.49 / 5.51 s | 12.6 / 13.0 ms | 55.2 / 55.2% |
  | stock + MTP | 76.0 / 77.3 | 15.6 / 4.4% | 7.71 / 7.46 s | 14.1 / 14.5 ms | 48.3 / 48.9% |
  vs prefill-prioritized: chat attainment **+26%**, agent attainment 2.5x, agent throughput +29%, chat TTFT -81%, chat
  TPOT -22%. Higher acceptance lifts every system (prefill-prioritized chat 49 -> 55%); the chat-attainment gap to
  +30% at 8 req/s stays (ours ~24% TPOT misses at both acceptances).
- [x] **KV page 128 (2026-10-20, H200, standard mix, chat 8 req/s; `bash tasks/pd_colocation/h200_page128_20261020.sh`;
  two repeats, clean; ours logged page_size 128 and 33K split calls) -- negative.** Every arm before ran sglang's default
  page 1, where the FA3 role gathers K/V per token instead of paged TMA. At page 128: ours agent 93.2 / 94.7 tok/s,
  attainment 40.0 / 37.8%, chat 57.8 / 59.6% (vs 102.5 / 45.0% / 63.0% at page 1); prefill-prioritized + MTP chat 48.8 /
  51.3% (vs 49.3%). Page 128 costs ours ~9% throughput and 4 chat points and leaves stock unchanged; page 1 stays.
  - [x] **Why (2026-10-21, kernel + step traces).** FA3 itself is faster at page 128: sgl-kernel `flash_attn_with_kvcache`
    on the serving shapes (Qwen-7B heads, causal, scattered pages, every arm within 4.2e-3 of an fp32 causal reference;
    `tasks/debug_kernel_server_perf/fa3_page_bench_20261021.py`) runs 2048 @ 64K in 3.36 / 3.07 / 3.15 ms per layer at
    page 1 / 128 / 256 (583 / 637 / 620 TFLOP/s), 1024 @ 64K 1.81 / 1.54 / 1.57 ms, 2048 @ 32K 1.81 / 1.57 / 1.59 ms, 512
    @ 48K 0.68 / 0.59 / 0.60 ms -- 9-17% from paged TMA over the per-token gather. In serving (STEP_TRACE, ours + MTP rows
    + admission, page 1 vs 128, one repeat each; `h200_pg_steptrace_20261021.sh`, `tasks/perf_model/steptrace_compare.py`,
    `steptrace_gaps.py`) the deep mixed steps get that gain (1200-2100 new tokens at 16-49K prefix 110.4 -> 104.8 ms, at
    >= 49K 136.2 -> 125.4 ms) but decode steps -- 89% of steps -- get slower (GPU 9.10 -> 9.60 ms mean, p90 8.6 -> 11.5
    ms), so total GPU time is equal (297 vs 294 s) and page 128 finishes fewer steps (23.7K vs 22.5K; host gap 0.46 vs
    0.64 ms per step, negligible). Page 1 stays; a deep-context FA3 gain of ~8% on the mixed steps is the whole headroom
    the KV layout offers, small next to the 31-37 ms mean mixed step that sets the chat TPOT misses.
- [x] **MTP ablation and CUDA graphs off (2026-10-20, H200, standard mix, two repeats each, all 0 tracebacks;
  `bash tasks/pd_colocation/h200_mtp_ablation_20261020.sh`; sglang f7682cf73).** The full ours + MTP stack at cap 48
  (four-mode step + split + EDF + deadline chunk) minus one component, per tenant (agent 10 s + 80 ms, chat 0.5 s +
  20 ms), rep 1 | rep 2:
  | arm | agent tput | agent TTFT | agent att. | chat TTFT | chat TPOT | chat att. |
  |---|---|---|---|---|---|---|
  | full stack (refs 1-2) | 101.7 / 103.1 | 12.50 / 12.29 s | 46.7 / 46.7% | 3.01 / 2.89 s | 12.9 / 13.0 ms | 58.4 / 59.1% |
  | no split (4-mode step on the stock kernel; MIXEDSPEC_STEPS 1751, no split calls) | 66.8 / 68.9 | 19.97 / 20.08 s | 11.1 / 11.1% | 9.10 / 9.06 s | 17.8 / 17.6 ms | 35.6 / 38.2% |
  | no EDF (FCFS, chunk 2048, no preemption) | 107.6 / 103.5 | 12.17 / 12.75 s | 20.0 / 15.6% | 4.24 / 4.72 s | 11.2 / 11.8 ms | 62.3 / 59.1% |
  | no deadline chunk (EDF + preemption, fixed chunk 2048) | 102.8 / 102.4 | 12.43 / 12.38 s | 46.7 / 46.7% | 2.84 / 2.90 s | 12.8 / 12.9 ms | 59.1 / 58.9% |
  | ours, CUDA graphs off | 78.6 / 83.3 | 17.31 / 15.59 s | 13.3 / 20.0% | 5.86 / 5.20 s | 22.7 / 22.3 ms | 36.8 / 38.8% |
  | stock + MTP, CUDA graphs off | 64.1 / 61.7 | 20.96 / 21.19 s | 2.2 / 2.2% | 13.65 / 14.74 s | 23.6 / 24.6 ms | 26.0 / 20.8% |
  **The split kernel carries the MTP stack**: without it agent throughput falls 34% and both tenants' attainment
  collapses (agent 11%, chat 37%). EDF is what lets agents meet their SLO (47% vs 16-20% FCFS) at a cost of 0-3 chat
  points (FCFS admits chats in arrival order and lets agent prefill wait). The deadline chunk is neutral at this
  operating point (it mattered on the long mix, above). CUDA graphs are worth ~20% throughput to ours and ~4% to stock
  MTP; with graphs off ours still leads stock (agent +29%, chat attainment 37.8% vs 23.4%).
- [x] **Deadline-ordered decode rows (2026-10-20, H200).** Under the MTP cap the slot, not the GPU, is what
  chats wait for: our running chats decode at 13 ms/token against a 20 ms SLO while new ones queue for a slot.
  `apply_decode_rows_edf.py` lets up to `--max-running-requests` requests be resident but runs at most K rows per step
  (the 48-row speculative CUDA graph), the K whose next token is due soonest (a request that has emitted o tokens
  since it started decoding at t0 is due at t0 + SLO x (o + 1); chat 20 ms, prompts >= 24K 80 ms); the others sit
  out the step with their KV and draft state in place (the overlap relay is keyed by req_pool_idx). Rows merge back
  before admission, so the cap, KV reservation and chunk rules see all of them.
  - [x] **Output gate (2026-10-20, `gate_decode_rows_20261020.sh`, 160 ShareGPT prompts at once, greedy, 192 tokens,
    ours + MTP 4-mode; 0 tracebacks; K=16 split 501+ steps at 154 rows mean, 138 sitting out):** agreement with the
    cap-160 reference -- rerun of the reference (noise floor) 99/160 exact, mean common prefix 705 of 863 chars; K=16
    41/160, 422; **null case, no parking at cap 16 (the same 16-row steps) 43/160, 411.** Parking agrees as well as
    its batch-size null: the divergence is bf16 batch-shape numerics (texts diverge at near-tie words and stay
    coherent), not state corruption. PASS.
  - [x] **Without an admission check it trades TTFT misses for TPOT misses (K=48, cap 96, rep 1, clean; split at 73 rows
    mean, 25 sitting out):** chat TTFT miss 7.2% (vs 37.8% for ours at cap 48, rep 6), TPOT miss 42.0% (vs 23.4%), mean
    TPOT 57.6 ms -- attainment 58.0% vs 59.0%. Up to 96 resident rows each ran about half the steps. Agents: TTFT 4.25 s
    (vs 12.39 s) but TPOT 104.6 ms (vs 18.6), attainment 37.8% (vs 46.7%).
  - [x] **SLO-checked admission (`apply_decode_rows_admit.py`, applied on top):** admit while the resident rows' SLO
    load stays within util x K slots -- a row with SLO s takes dt / (s x a) of a slot per step, dt the measured step
    interval, a the resident rows' tokens per verify (output tokens / sglang's `spec_verify_ct`); the newcomer is
    charged as a chat row. The admission check binds when long mixed steps run (logged: dt 42.8 ms, a 2.03, load 45.2
    of 43.2; dt 104 ms, load 127.6).
  - [x] **Serving A/B (K=48, util 0.9, cap 128 as a bound; `bash tasks/pd_colocation/h200_decode_rows_20261020.sh`,
    `h200_decode_rows_more_20261020.sh`; four repeats, all clean; sglang 0e54fc671).** Per tenant (agent 10 s + 80 ms,
    chat 0.5 s + 20 ms), per-repeat values:
    | arm | agent tput | agent TTFT | agent att. | chat TTFT | chat TPOT | chat att. (TTFT miss / TPOT miss) |
    |---|---|---|---|---|---|---|
    | stock prefill-prioritized + MTP (4: rep 1-2, refs 1-2) | 81.0-82.2 | 15.46-15.53 s | 6.7-8.9% | 7.54-7.68 s | 16.0-18.1 ms | 49.1 / 49.3 / 49.3 / 49.5% (~50.5% / ~23.5%) |
    | ours + MTP, cap 48 (in-session refs 1-2) | 101.7 / 103.1 | 12.50 / 12.29 s | 46.7 / 46.7% | 3.01 / 2.89 s | 12.9 / 13.0 ms | 58.4 / 59.1% (38.4 / 37.6%, 24.2 / 23.8%) |
    | **ours + MTP, deadline-ordered rows + SLO admission (4)** | 106.2 / 101.8 / 101.8 / 100.2 | 10.60-11.37 s | 46.7 / 42.2 / 48.9 / 42.2% | 2.04-2.38 s | 14.4-15.1 ms | **65.0 / 62.3 / 63.6 / 61.2%** (31-33% / 29-31%) |
    | same, util 0.8 (1) | 100.3 | 13.07 s | 35.6% | 3.05 s | 13.6 ms | 59.9% (37.2% / 26.7%) |
    Means: chat attainment **63.0 +- 1.4% vs 49.3%, +28%** (the bar's 30% would need 64.1%); vs ours at cap 48, +4.1
    points. Agent attainment 45.0% vs 8.1% (5.5x), agent throughput +25%, chat TTFT -70%, agent TTFT -28%; chat TPOT
    14.8 vs 16.9 ms (-12%). The rows move chat misses from TTFT (38% -> 32%) to TPOT (24% -> 30%): the remaining TPOT
    misses are chats decoding through long mixed steps (logged step interval up to ~104 ms, where a chat row at 2.0
    tokens per verify cannot meet 20 ms in any schedule), and admission closes during those steps, which is where the
    remaining TTFT misses queue. **Short of +30% at chat 8 req/s by ~1 point.** util 1.0 (one repeat): 62.5%, agent
    44.4% -- the same range as 0.9.
  - [x] **Attainment vs chat rate (2026-10-20, `bash tasks/pd_colocation/h200_chat_rate_sweep_20261020.sh`, agent load
    fixed at C=8, chat 300 s at 2 / 4 / 6 req/s, one repeat each; 8 req/s = the four-repeat means above).** Per tenant,
    ours + MTP (rows + admission) / prefill-prioritized stock + MTP / stock + MTP (mixed chunk):
    | chat req/s | chat attainment | chat TTFT | chat TPOT | agent attainment | agent TTFT | agent TPOT | agent tput |
    |---|---|---|---|---|---|---|---|
    | 2 | **74.7** / 65.7 / 63.2% | 0.42 / 1.73 / 1.76 s | 13.6 / 28.0 / 28.5 ms | 66.7 / 66.7 / 62.2% | 5.86 / 4.22 / 4.81 s | 25.7 / 66.3 / 75.5 ms | 154 / 119 / 113 |
    | 4 | **70.3** / 60.5 / 58.2% | 1.01 / 4.07 / 4.69 s | 12.7 / 20.8 / 21.9 ms | 62.2 / 42.2 / 24.4% | 8.35 / 10.23 / 11.21 s | 19.3 / 39.5 / 42.6 ms | 132 / 102 / 95 |
    | 6 | **69.3** / 55.7 / 48.8% | 1.71 / 5.44 / 7.08 s | 12.6 / 18.0 / 18.8 ms | 51.1 / 20.0 / 4.4% | 10.47 / 12.86 / 14.57 s | 21.8 / 41.7 / 47.7 ms | 118 / 92 / 81 |
    | 8 | **63.0** / 49.3 / 34.2% | 2.29 / 7.59 / 12.54 s | 14.8 / 16.9 / 17.2 ms | 45.0 / 8.1 / 3.0% | 11.1 / 15.5 / 20.2 s | 39.0 / 30.1 / 29.2 ms | 103 / 81 / 66 |
    vs prefill-prioritized + MTP, chat attainment +14% / +16% / +24% / +28%; chat TTFT -76% / -75% / -69% / -70% and
    chat TPOT -51% / -39% / -30% / -12%; agent attainment equal / 1.5x / 2.6x / 5.5x; agent throughput +30% / +29% / +29%
    / +27%. At 4 and 6 req/s every bar item clears per tenant (chat TTFT and TPOT both >= -30%, agent attainment 1.5x /
    2.6x, throughput +29%); at 2 req/s chat clears on TTFT and TPOT but agents only tie on attainment (agent TTFT +39%: EDF serves the
    chats' 0.5 s deadlines first); at 8 req/s chat TPOT (-12%) and chat attainment (+28%) miss. **About 24% of chats
    miss TPOT at every rate, ours included**, so that share does not come from chat load: it is the chats that decode
    beside 32-64K agent prefill chunks. The first chat 6 run of ours crashed on the second-chunked-request assert (the
    re-forked H200 tree lacked `edf_one_chunked_req.patch`; now `apply_edf_one_chunked_req.py` on both pod sets); the
    row above is its clean rerun (`h200_chat6_rerun_20261020.sh`). All other arms 0 tracebacks; sglang f7682cf73.
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

