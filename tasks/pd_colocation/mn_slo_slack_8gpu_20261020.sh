#!/bin/bash
# Multi-node SLO control, per-row slack (2026-10-20): the worker's decode-first budget is per running row against its
# mean-TPOT SLO (x <= s*(o + acc) - E), chat 20 ms / agents (prompt >= 24K) 80 ms, measured decode-step base; the router
# prices per class too (--itl-slo-ms 80 --itl-slo-short-ms 20, no adaptive SLO). The per-step rule lowered attainment
# (91% -> 65% chat, 551 -> 262 tok/s agent goodput) by enforcing the SLO on every step. Waits for pid $1. Claude.
set -u
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
export FO_TEXT=1
for REPN in 1 2; do
  LOAD_PUB=1 ITL_SLO=20:1:6.972:0.2228:0.1137:0.1091:4.652e-6:measured:slack:cls24000_80 \
    EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 80 --itl-slo-short-ms 20 --long-prompt-tokens 24000 --slo-chunk 20:0.1091:4.652e-6" \
    REP=_text_sloctls_r$REPN ARMS=coloc_slo_edf_hc bash mn_fanout_8gpu_20261018.sh
done
echo MN_SLO_SLACK_DONE
