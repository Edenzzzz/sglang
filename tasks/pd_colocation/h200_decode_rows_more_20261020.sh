#!/bin/bash
# More repeats of deadline-ordered rows + SLO admission (util 0.9) and its util sensitivity (0.8, 1.0), after
# h200_decode_rows_20261020.sh (pid $1) ends; then the MTP ablation. Two repeats left chat attainment at +29% over
# prefill-prioritized stock + MTP with a 2.7-point spread between them. Claude, 2026-10-20.
set -u
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
for REPN in 3 4; do
  DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 48" \
    TAG=_pt_rows48adm_cap128_rep$REPN ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
done
for U in 0.8 1.0; do
  DECODE_ROWS=48:20:24000:80:$U WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 48" \
    TAG=_pt_rows48adm_u${U}_rep1 ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
done
TAG=_pt_rows_ref_rep3 ARMS=stock_mtp_pp bash mn_h200_nospec_20261018.sh
bash h200_mtp_ablation_20261020.sh
echo H200_DECODE_ROWS_MORE_DONE
