#!/bin/bash
# Qwen2.5-14B at a load scaled to its capacity (2026-10-20): the 7B-sized load (C=32, chat 8 req/s) overloads 14B on
# H20 (chat TPOT p50 18-21 ms). Half load: agent programs C=16, chat 4 req/s (3600 requests); same 14B calibration.
# Ours vs PD disagg, two repeats. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export MODEL_DIR=qwen14b FO_TEXT=1 RA_LOG=/tmp/h20/mn_14b_calibrate.log AGENT_CONC=16 CHAT_RATE=4 CHAT_N=3600
for REPN in 1 2; do
  EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_text_14b_half_r$REPN ARMS=coloc_slo_edf_hc \
    bash mn_fanout_8gpu_20261018.sh
  REP=_text_14b_half_r$REPN ARMS=disagg_2p2d bash mn_fanout_8gpu_20261018.sh
done
echo MN_14B_HALF_DONE
