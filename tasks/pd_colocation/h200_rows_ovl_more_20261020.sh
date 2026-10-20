#!/bin/bash
# Repeats 3-4 of deadline-ordered rows + SLO admission + FA2 overlapped beside FA3 (the full MTP configuration;
# h200_rows_ovl_20261020.sh had reps 1-2 at chat attainment 62.9 / 63.7%). Claude, 2026-10-20.
set -u
cd "$(dirname "$0")"
for REPN in 3 4; do
  DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 48" \
    TAG=_pt_rows48adm_rep$REPN ARMS=ours_mtp_ovl bash mn_h200_nospec_20261018.sh
done
echo H200_ROWS_OVL_MORE_DONE
