#!/bin/bash
# After h200_itl_slo_chunk_20261019.sh (2026-10-19): (1) sglang's prefill-prioritized chunked prefill (no mixed chunk),
# without and with MTP, two repeats -- the single-node no-SLO-control baseline not yet run; (2) STEPTRACE runs for the
# 4-mode step-model search (tasks/perf_model/fit_4mode_step.py): five arms at the standard load and at a lighter one,
# and the two MTP arms at 2 draft steps / 3 verify tokens so the draft / verify terms are identifiable. Claude.
set -u
cd "$(dirname "$0")"
while pgrep -f h200_itl_slo_chunk_20261019.sh >/dev/null; do sleep 30; done
for REPN in 1 2; do TAG=_pt_rep$REPN ARMS="stock_pp stock_mtp_pp" bash mn_h200_nospec_20261018.sh; done
STEP_TRACE=1 TAG=_trace ARMS="stock ours_ovl pdmux stock_mtp ours_mtp" bash mn_h200_nospec_20261018.sh
STEP_TRACE=1 TAG=_trace_l4 AGENT_CONC=4 CHAT_RATE=4 CHAT_N=1200 SESSIONS=12 \
  ARMS="stock ours_ovl pdmux stock_mtp ours_mtp" bash mn_h200_nospec_20261018.sh
STEP_TRACE=1 TAG=_trace_s2t3 SPEC_STEPS=2 SPEC_TOKENS=3 ARMS="stock_mtp ours_mtp" bash mn_h200_nospec_20261018.sh
echo H200_PP_TRACE_DONE
