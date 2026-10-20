#!/bin/bash
# MTP arms with the running-request cap raised (2026-10-20): every MTP arm so far ran with max_running_requests = 48
# (sglang derives it from the speculative CUDA-graph batch size; all three hit 48 running, queue mean 29.6 in ours),
# while no-MTP arms run uncapped and peak at 115. That cap, not the schedule, is why 37-68% of MTP chats miss on
# TTFT. Same raise for every MTP arm: --max-running-requests 128 --cuda-graph-max-bs-decode 128. Two repeats. Claude.
set -u
cd "$(dirname "$0")"
export WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 128"
for REPN in 1 2; do
  TAG=_pt_cap128_rep$REPN ARMS="stock_mtp stock_mtp_pp ours_mtp" bash mn_h200_nospec_20261018.sh
done
echo H200_MTP_CAP128_DONE
