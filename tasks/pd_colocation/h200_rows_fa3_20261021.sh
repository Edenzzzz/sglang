#!/bin/bash
# Split rows on FA3 (apply_split_rows_fa3.py, SPLIT_ROWS_FA3=1) in serving (H200, standard mix, chat template, real
# head; 2026-10-21). fusion_microbench_20261021.py: rows on FA3 cut the mixed step 49-89% vs the FA2-BatchPrefill rows.
#   1. gate: ours + MTP (rows + SLO admission) with SPLIT_VERIFY=1 -- every split layer against the full planned wrapper,
#      reduced load (SESSIONS=8, CHAT_N=600); pass = 0 FUSED_SPLIT_VERIFY FAIL lines and SPLIT_ROWS_FA3 calls > 0.
#   2. A/B, two repeats: ours + MTP rows + admission with rows on FA3, and ours without MTP (overlapped split) with rows
#      on FA3, against the same arms with FA2 rows run fresh in this session (one repeat) plus the _ct_ table's two.
# Claude.
set -u
cd "$(dirname "$0")"
export CHAT_TEMPLATE=1
WX="--max-running-requests 128 --cuda-graph-max-bs-decode 48"
SPLIT_ROWS_FA3=1 SPLIT_VERIFY=1 SESSIONS=8 CHAT_N=600 DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" \
  TAG=_ct_rows48adm_r3gate ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
for REPN in 1 2; do
  SPLIT_ROWS_FA3=1 DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" \
    TAG=_ct_rows48adm_r3_rep$REPN ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
  SPLIT_ROWS_FA3=1 TAG=_ct_r3_rep$REPN ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 bash mn_h200_nospec_20261018.sh
done
DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" TAG=_ct_rows48adm_rep3 ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
TAG=_ct_rep3 ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 bash mn_h200_nospec_20261018.sh
echo H200_ROWS_FA3_DONE
