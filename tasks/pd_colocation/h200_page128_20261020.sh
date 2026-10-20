#!/bin/bash
# KV page size in serving (H200, standard mix, chat 8 req/s, 2026-10-20): every serving arm so far ran sglang's default
# page_size 1, where the FA3 role cannot load K/V with paged TMA and gathers per token; the kernel harness measured it
# at larger pages. Both systems at --page-size 128 (FA3's paged TMA needs a multiple of its 128-token K/V tile; page 128 is the size
# the 2026-10-13 serving integration passed with): ours + MTP (rows + SLO admission) and prefill-prioritized stock + MTP,
# two repeats, measured head (no simulated acceptance). Starts after pid $1. Claude.
set -u
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
for REPN in 1 2; do
  DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="--page-size 128 --max-running-requests 128 --cuda-graph-max-bs-decode 48" \
    TAG=_pt_rows48adm_pg128_rep$REPN ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
  WORKER_EXTRA="--page-size 128" TAG=_pt_pg128_rep$REPN ARMS=stock_mtp_pp bash mn_h200_nospec_20261018.sh
done
echo H200_PAGE128_DONE
