#!/bin/bash
# Qwen2.5-14B with MTP on 8 H20s (2026-10-20): no trained EAGLE3 head exists for 14B, so the draft is a random-weight
# head of 14B's shape (make_sim_eagle3_head.py -> qwen14b_eagle3_sim) and acceptance is simulated at 2.1 tokens per
# verify (SGLANG_SIMULATE_ACC_LEN, the 7B head's measured acceptance): the draft and verify work per step are real.
# Half load (agent C=16, chat 4 req/s, 3600 chats), as mn_14b_half_8gpu_20261020.sh. Ours (4-mode MTP, SLO-checked
# routing, pool, EDF) vs PD disagg 2P2D with MTP on every role; two repeats. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export MODEL_DIR=qwen14b DRAFT_DIR=qwen14b_eagle3_sim SIM_ACC=2.1 FO_TEXT=1 RA_LOG=/tmp/h20/mn_14b_calibrate.log
export AGENT_CONC=16 CHAT_RATE=4 CHAT_N=3600
for REPN in 1 2; do
  EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_text_14b_mtp_r$REPN ARMS=coloc_slo_edf_mtp_hc \
    bash mn_fanout_8gpu_20261018.sh
  REP=_text_14b_mtp_r$REPN ARMS=disagg_2p2d_mtps bash mn_fanout_8gpu_20261018.sh
done
echo MN_14B_MTP_DONE
