#!/bin/bash
# pdmux STEPTRACE run (apply_steptrace_pdmux.py, applied 2026-10-20 after the first pdmux trace emitted no lines), for
# the no-spec predictability comparison: ours vs pdmux vs stock, each with its best step model. Runs after the
# repeats script (pid 1944390). Claude.
set -u
cd "$(dirname "$0")"
while kill -0 1944390 2>/dev/null; do sleep 30; done
STEP_TRACE=1 TAG=_trace_v2 ARMS=pdmux bash mn_h200_nospec_20261018.sh
STEP_TRACE=1 TAG=_trace_l4_v2 AGENT_CONC=4 CHAT_RATE=4 CHAT_N=1200 SESSIONS=12 ARMS=pdmux bash mn_h200_nospec_20261018.sh
echo H200_PDMUX_TRACE_DONE
