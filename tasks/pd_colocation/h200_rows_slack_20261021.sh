#!/bin/bash
# Deadline-ordered rows + SLO admission + the per-row slack chunk cap (apply_edf_itl_slo_chunk.py ":slack", measured
# decode base, chat 20 ms / prompts >= 24K 80 ms) on ours + MTP (H200, standard mix, chat 8 req/s, 2026-10-21). Alone
# (10-20) the slack cap starved prefill: with ~2400 chats some row was always behind its SLO. With admission keeping the
# resident set within its TPOT SLO load, rows should rarely be behind, so the cap binds only when a long agent chunk
# would push a chat over 20 ms. Rep 1 without a floor: cap on 86% of steps, chat TTFT misses 57%; from here short
# prompts pass the cap (apply_itl_slo_short_floor.py). Two repeats. Claude.
set -u
cd "$(dirname "$0")"
for REPN in 1 2; do
  DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 48" \
    ITL_SLO=20:2.0:9:0.2:0.03:0.02:5e-7:measured:slack:cls24000_80 \
    TAG=_pt_rows48adm_slackfl_rep$REPN ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
done
echo H200_ROWS_SLACK_DONE
