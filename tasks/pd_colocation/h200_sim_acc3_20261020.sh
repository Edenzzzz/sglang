#!/bin/bash
# Acceptance sensitivity (H200, standard mix, chat 8 req/s, 2026-10-20): our Qwen-7B EAGLE3 head accepts ~2.1 tokens
# per verify (3 draft steps, 4 verify tokens), below what trained EAGLE3 heads typically reach on chat. Same arms with
# acceptance fixed at 3.0 via SGLANG_SIMULATE_ACC_LEN (SIM_ACC; the head still drafts, the verify still runs; output
# lengths are fixed by the drivers, which ignore EOS): ours + MTP (rows + SLO admission), prefill-prioritized stock + MTP,
# stock + MTP. Two repeats. Claude.
set -u
cd "$(dirname "$0")"
export SIM_ACC=3.0
for REPN in 1 2; do
  DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 48" \
    TAG=_pt_rows48adm_acc3_rep$REPN ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
  TAG=_pt_acc3_rep$REPN ARMS="stock_mtp_pp stock_mtp" bash mn_h200_nospec_20261018.sh
done
echo H200_SIM_ACC3_DONE
