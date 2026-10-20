#!/bin/bash
# Deadline-ordered decode rows on the standard mix (H200 dadf94f0e1da7238, 2026-10-20): ours + MTP with up to CAP
# running requests resident but at most 48 rows per step (the speculative CUDA-graph batch), the 48 whose next token is
# due soonest (chat 20 ms, prompts >= 24K tokens 80 ms; apply_decode_rows_edf.py). At cap 48 without it, 36-39% of our
# chat misses are TTFT misses queued behind the cap. Matched in-session repeats of ours + MTP at cap 48 and
# prefill-prioritized stock + MTP (the baseline the chat-attainment bar misses against). Rep 1 at cap 96 without an
# admission check let chat TPOT reach 57.6 ms (each resident row ran half the steps); from then on admission stops once
# the resident rows' SLO load reaches util 0.9 of the 48 slots (apply_decode_rows_admit.py), and the cap (128) is only
# a bound. Claude.
set -u
cd "$(dirname "$0")"
for REPN in 1 2; do
  for CAP in ${CAPS:-128}; do
    DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="--max-running-requests $CAP --cuda-graph-max-bs-decode 48" \
      TAG=_pt_rows48adm_cap${CAP}_rep$REPN ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
  done
  TAG=_pt_rows_ref_rep$REPN ARMS="ours_mtp stock_mtp_pp" bash mn_h200_nospec_20261018.sh
done
echo H200_DECODE_ROWS_DONE
