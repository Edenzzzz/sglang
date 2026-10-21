#!/bin/bash
# Serving A/B after the rows-on-FA3 gate passed (H200, standard mix, chat template, real head; 2026-10-21). Stock now
# also on upstream's own Hopper default backend, FA3 (ATTN_BACKEND=fa3; get_default_attn_backend picks it for MHA with
# no spec or topk 1) -- every stock arm before forced flashinfer, whose BatchPrefill runs decode-shaped rows ~8x slower
# than FA3 (rows_layout_check_20261021.py). Two repeats: prefill-prioritized stock + MTP and stock mixed chunk without
# MTP on FA3, ours + MTP (rows + SLO admission) and ours without MTP (overlapped split) with rows on FA3.
# Claude.
set -u
cd "$(dirname "$0")"
export CHAT_TEMPLATE=1
WX="--max-running-requests 128 --cuda-graph-max-bs-decode 48"
for REPN in 1 2; do
  ATTN_BACKEND=fa3 TAG=_ct_fa3be_rep$REPN ARMS="stock_mtp_pp stock stock_pp" bash mn_h200_nospec_20261018.sh
  SPLIT_ROWS_FA3=1 DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" \
    TAG=_ct_rows48adm_r3_rep$REPN ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
  SPLIT_ROWS_FA3=1 TAG=_ct_r3_rep$REPN ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 bash mn_h200_nospec_20261018.sh
done
echo H200_ROWS_FA3_AB_DONE
