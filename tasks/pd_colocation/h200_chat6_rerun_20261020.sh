#!/bin/bash
# Rerun of ours + MTP at chat 6 req/s (h200_chat_rate_sweep_20261020.sh) after the one-chunked-request fix
# (apply_edf_one_chunked_req.py): the first run crashed on the scheduler's second-chunked-request assert. Starts after
# pid $1 (the sweep + ablation chain). Claude, 2026-10-20.
set -u
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
CHAT_RATE=6 CHAT_N=1800 DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 48" \
  TAG=_pt_rows48adm_chat6_rep1b ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
echo H200_CHAT6_RERUN_DONE
