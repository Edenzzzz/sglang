#!/bin/bash
# Deadline-ordered rows + SLO admission with FA2 overlapped beside FA3 (SPLIT_OVERLAP=1; H200, standard mix, chat 8
# req/s, 2026-10-20). At cap 48 the overlap did nothing for MTP (10-18: few verify rows to hide); with rows the verify
# batch stays near 48 rows, so it is measured again. Two repeats, after pid $1. Claude.
set -u
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
for REPN in 1 2; do
  DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 48" \
    TAG=_pt_rows48adm_rep$REPN ARMS=ours_mtp_ovl bash mn_h200_nospec_20261018.sh
done
echo H200_ROWS_OVL_DONE
