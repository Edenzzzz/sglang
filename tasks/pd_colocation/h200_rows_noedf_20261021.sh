#!/bin/bash
# Deadline-ordered rows + SLO admission on the FCFS 4-mode MTP stack (no EDF, no preemption, chunk 2048; H200, standard
# mix, chat 8 req/s, 2026-10-21). In the ablation, FCFS admission gave chat 59-62% (vs 58.9% with EDF) at agent
# attainment 16-20% (vs 47%); with rows + admission on top, chat may clear +30% over prefill-prioritized + MTP (49.3%)
# while agents still lead it (8.1%). Two repeats. Claude.
set -u
cd "$(dirname "$0")"
for REPN in 1 2; do
  DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 48" \
    TAG=_pt_rows48adm_rep$REPN ARMS=ours_mtp_noedf bash mn_h200_nospec_20261018.sh
done
echo H200_ROWS_NOEDF_DONE
