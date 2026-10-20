#!/bin/bash
# SLO attainment vs chat rate (H200 dadf94f0e1da7238, 2026-10-20): at chat 8 req/s + agents C=8 both MTP systems run
# overloaded (chat attainment 63% ours, 49% prefill-prioritized stock + MTP), so attainment is compared across load:
# chat 2 / 4 / 6 req/s with the same agent load (C=8, 32-64K TraceLab, 2 rounds), CHAT_N = 300 s of chat. Ours + MTP
# (4-mode, split, EDF, deadline-ordered rows + SLO admission util 0.9) vs prefill-prioritized stock + MTP and stock +
# MTP (mixed chunk). Starts after pid $1; then the prefill-prioritized reference repeat and the MTP ablation. Claude.
set -u
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
for R in 2 4 6; do
  N=$((R * 300))
  CHAT_RATE=$R CHAT_N=$N DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 48" \
    TAG=_pt_rows48adm_chat${R}_rep1 ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
  CHAT_RATE=$R CHAT_N=$N TAG=_pt_chat${R}_rep1 ARMS="stock_mtp_pp stock_mtp" bash mn_h200_nospec_20261018.sh
done
TAG=_pt_rows_ref_rep3 ARMS=stock_mtp_pp bash mn_h200_nospec_20261018.sh
bash h200_mtp_ablation_20261020.sh
echo H200_CHAT_RATE_SWEEP_DONE
