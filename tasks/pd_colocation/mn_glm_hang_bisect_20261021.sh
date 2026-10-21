#!/bin/bash
# GLM-4.5-Air (MoE, TP4, one H20 pod) hung in ours-without-MTP at quarter load: "Scheduler watchdog timeout
# (self.watchdog_timeout=600.0)" on TP1, then SIGQUIT -- a forward step never returned (2026-10-21,
# _glm_L2_ct_pol_r1; the same arm completed at half load, and ours + MTP completed at both, so it is intermittent).
# That arm is the only one with DECODE_FA3=1 (plain decode on FA3). Bisect at the same load, 4 runs:
#   a1, a2  full config (SPLIT_ROWS_FA3 + DECODE_FA3 + SPLIT_POLICY) -- is it reproducible?
#   b       without DECODE_FA3
#   c       without SPLIT_POLICY
# Each prints whether the worker survived and what the watchdog said. Claude.
set -u
cd "$(dirname "$0")"
export CHAT_TEMPLATE=1 AGENT_CONC=2 SESSIONS=6 CHAT_RATE=2 CHAT_N=600
run() { # name, env
  echo "##### $1"
  env $2 TAG=_glm_hang_$1 ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 bash mn_h20_single_20261021.sh
}
run a1 "SPLIT_ROWS_FA3=1 DECODE_FA3=1 SPLIT_POLICY=1"
run a2 "SPLIT_ROWS_FA3=1 DECODE_FA3=1 SPLIT_POLICY=1"
run b  "SPLIT_ROWS_FA3=1 DECODE_FA3=0 SPLIT_POLICY=1"
run c  "SPLIT_ROWS_FA3=1 DECODE_FA3=1 SPLIT_POLICY=0"
echo MN_GLM_HANG_BISECT_DONE
