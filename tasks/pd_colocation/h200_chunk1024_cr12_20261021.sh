#!/bin/bash
# Ours' chat TPOT tail at chat 12 req/s (H200, 2026-10-21; p90 55 vs 43 ms for prefill-prioritized stock on FA3): a
# 2048-token deadline chunk at 32-64K prefix is ~2-3.4 ms per layer. Ours (every attention call on FA3, split policy)
# with the deadline chunk at 512 / 1024 (CHUNK=1024 DEADLINE_CHUNK=512:1024:6000) vs the 1024 / 2048 default.
# After h200_mtp_gap_20261021.sh (pid $1). One repeat. Claude.
set -u
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
export CHAT_TEMPLATE=1 CHAT_RATE=12 CHAT_N=3600
SPLIT_ROWS_FA3=1 DECODE_FA3=1 SPLIT_POLICY=1 TAG=_ct_pol_cr12 ARMS=ours_ovl CHUNK=1024 DEADLINE_CHUNK=512:1024:6000 \
  bash mn_h200_nospec_20261018.sh
echo H200_CHUNK1024_DONE
