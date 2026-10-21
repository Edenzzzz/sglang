#!/bin/bash
# Chat-rate sweep against stock on FA3 (H200, 2026-10-21), after h200_auto_margin_20261021.sh (pid $1). At chat 8 req/s
# stock on FA3 already attains ~80% chat, so the bar needs a load at which it saturates: chat 12 and 16 req/s (3600 /
# 4800 chats, 300 s), agents C=8 over 24 sessions, chat template, no MTP. Arms: stock mixed chunk and prefill-
# prioritized on FA3 (ATTN_BACKEND=fa3), ours on FA3 for every attention call (split rows + plain decode), without
# and with the per-step SM partition (SPLIT_MARGIN_AUTO). One repeat per rate. Claude.
set -u
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
export CHAT_TEMPLATE=1
for R in 12 16; do
  export CHAT_RATE=$R CHAT_N=$((R * 300))
  ATTN_BACKEND=fa3 TAG=_ct_fa3be_cr${R} ARMS="stock stock_pp" bash mn_h200_nospec_20261018.sh
  SPLIT_ROWS_FA3=1 DECODE_FA3=1 TAG=_ct_r3d3_cr${R} ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 \
    bash mn_h200_nospec_20261018.sh
  SPLIT_ROWS_FA3=1 DECODE_FA3=1 SPLIT_MARGIN_AUTO=1 TAG=_ct_r3d3am_cr${R} ARMS=ours_ovl CHUNK=2048 \
    DEADLINE_CHUNK=1024:2048:6000 bash mn_h200_nospec_20261018.sh
done
echo H200_FA3_RATE_SWEEP_DONE
