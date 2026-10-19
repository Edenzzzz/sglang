#!/bin/bash
# 5e load spike (2026-10-20): 8 H20s, text fan-out (C=32 agents, 16 x 64K prefixes, 180 s rotation) + ShareGPT chat
# 8 req/s; at t = 300 s a second chat stream adds 16 req/s for 240 s (3840 requests; CHAT_SPIKE in mn_bench.sh). Ours
# (SLO-checked hit-first router, pool, EDF; no decode-first) vs PD disagg 2P2D, two repeats. Scored in windows
# (pre / spike / post) by tasks/perf_model/spike_windows.py. Claude.
set -u
cd "$(dirname "$0")"
export FO_TEXT=1 CHAT_SPIKE=300:16:3840
for REPN in 1 2; do
  EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_text_spike_r$REPN ARMS=coloc_slo_edf_hc \
    bash mn_fanout_8gpu_20261018.sh
  REP=_text_spike_r$REPN ARMS=disagg_2p2d bash mn_fanout_8gpu_20261018.sh
done
echo MN_SPIKE_DONE
