#!/bin/bash
# Single-node standard-mix table rerun with chat-templated chat prompts (CHAT_TEMPLATE=1) and real draft heads only
# (H200, 2026-10-21; the user: "if there's a real head do not simulate acceptance", "why not use chat template"). The
# TraceLab agent stream replays recorded token streams and is unchanged. Arms as in the six-repeat table: stock mixed
# chunk, prefill-prioritized, ours without MTP (overlapped split, EDF, deadline chunk), stock + MTP,
# prefill-prioritized + MTP, ours + MTP (4-mode, rows + SLO admission). Two repeats; logged accept len per MTP arm.
# Claude.
set -u
cd "$(dirname "$0")"
export CHAT_TEMPLATE=1
for REPN in 1 2; do
  TAG=_ct_rep$REPN ARMS="stock stock_pp ours_ovl stock_mtp stock_mtp_pp" CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 \
    bash mn_h200_nospec_20261018.sh
  DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 48" \
    TAG=_ct_rows48adm_rep$REPN ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
done
echo H200_CT_DONE
