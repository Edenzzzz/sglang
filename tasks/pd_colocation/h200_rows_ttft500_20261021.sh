#!/bin/bash
# EDF base TTFT deadline set to the chat tenant's scored SLO (TTFT_SLO_MS=500, from the default 2000; length term
# unchanged, so a 64K agent's deadline goes 14 s -> 12.5 s) with deadline-ordered rows + SLO admission (H200, standard
# mix, chat 8 req/s, 2026-10-21). With a 2 s base EDF treats a chat as on time until 2 s and serves late agent prefills
# ahead of chats already past the 0.5 s SLO; 32% of chats miss TTFT. Two repeats. Claude.
set -u
cd "$(dirname "$0")"
for REPN in 1 2; do
  TTFT_SLO_MS=500 DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 48" \
    TAG=_pt_rows48adm_t500_rep$REPN ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
done
echo H200_ROWS_T500_DONE
