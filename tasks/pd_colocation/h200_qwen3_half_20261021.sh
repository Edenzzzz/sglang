#!/bin/bash
# Qwen3-8B single-node table at a load scaled to its capacity (H200, 2026-10-21): at the Qwen2.5-7B load (agents C=8,
# chat 8 req/s) every system is overloaded (chat attainment 0-25%, MTP arms' chat TTFT 100-170 s). Half load: agent
# programs C=4 over 12 sessions, chat 4 req/s, 1200 chats; trained EAGLE3 head (accept ~2.0), chat template on. Two
# repeats. Claude.
set -u
cd "$(dirname "$0")"
export CHAT_TEMPLATE=1 MODEL_DIR=qwen3_8b DRAFT_DIR=qwen3_8b_eagle3 AGENT_CONC=4 SESSIONS=12 CHAT_RATE=4 CHAT_N=1200
for REPN in 1 2; do
  TAG=_q3h_ct_rep$REPN ARMS="stock stock_pp ours_ovl stock_mtp stock_mtp_pp" CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 \
    bash mn_h200_nospec_20261018.sh
  DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 48" \
    TAG=_q3h_ct_rows48adm_rep$REPN ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
done
echo H200_Q3H_DONE
