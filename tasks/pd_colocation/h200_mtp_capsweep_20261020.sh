#!/bin/bash
# MTP running-request cap sweep (2026-10-20): at 48 (sglang's speculative default) 36-39% of our chats miss TTFT from
# the queue it builds; at 128 every arm's steps slow down (ours TPOT 18.6 -> 40 ms, stocks collapse). Intermediate caps,
# each system compared at its own best: ours + MTP at 64 and 80, stock + MTP and prefill-prioritized + MTP at 64.
# One repeat each. Claude.
set -u
cd "$(dirname "$0")"
for CAP in 64 80; do
  ARMS="ours_mtp"; [ $CAP = 64 ] && ARMS="stock_mtp stock_mtp_pp ours_mtp"
  WORKER_EXTRA="--max-running-requests $CAP --cuda-graph-max-bs-decode $CAP" TAG=_pt_cap${CAP}_rep1 ARMS="$ARMS" \
    bash mn_h200_nospec_20261018.sh
done
echo H200_MTP_CAPSWEEP_DONE
