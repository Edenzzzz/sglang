#!/bin/bash
# PD point 1 on the final single-GPU stack (2026-10-17): does the router's EDF/SLO check (cache_slo: walk
# workers by hit length, take the first predicted to meet the request's TTFT deadline given the uncached
# work queued ahead) cut agent TTFT -- the metric no 8-GPU arm has cut by 30% -- versus pure cache affinity?
# 1) calibrate the router's perf model on one worker running the SAME stack (mtpsplit + EDF + preemption +
#    4-mode MTP + load/context gate); 2) at agent C=16 and C=32 (chat 8 req/s): ours with cache_slo vs ours
#    with cache, both in this session. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=bb854a10b8954223 NPODS=4 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export KERNEL=mtpsplit PREEMPT=1 CHUNK=2048
EDF=1 SPEC=1 python3 -u mn_run.py calibrate | tee /tmp/h20/mn_slo_calibrate.log
RA=$(grep -oE "ROUTER_ARGS=.*" /tmp/h20/mn_slo_calibrate.log | tail -1 | cut -d= -f2-)
[ -z "$RA" ] && { echo "calibration failed"; exit 1; }
echo "ROUTER_ARGS: $RA"
for C in 16 32; do
  export AGENT_CONC=$C SESSIONS=$((C * 3)) CHAT_RATE=8 CHAT_N=2400 RUN_TAG=_8g_SLO check_c${C}_r8
  ROUTER_ARGS="$RA" python3 -u mn_run.py coloc_slo_edf_mtp
  python3 -u mn_run.py coloc_cache_edf_mtp
done
echo SLO_CHECK_DONE
