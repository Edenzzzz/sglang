#!/bin/bash
# Point 1 on H20: calibrate router.py's perf model on OUR worker config (split + EDF + preemption, MTP),
# then the SLO-SLO check router (cache_slo: highest-hit node that the model says keeps this request and the
# tighter-deadline ones queued there within SLO) at C=64/chat16 and C=32/chat8. Claude, 2026-10-16.
set -u
cd "$(dirname "$0")"
export SID=1c90eac0a6848b48 NPODS=8 KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 EDF=1 SPEC=1
python3 -u mn_run.py calibrate | tee /tmp/h20/mn_calibrate.log
RA=$(grep -o "ROUTER_ARGS=.*" /tmp/h20/mn_calibrate.log | tail -1 | cut -d= -f2-)
echo "ROUTER_ARGS: $RA"
[ -n "$RA" ] || { echo "CALIBRATION_FAILED"; exit 1; }
export ROUTER_ARGS="$RA --ttft-slo-ms 2000 --itl-slo-ms 50"
export SESSIONS=192 TURNS=2 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536
AGENT_CONC=64 CHAT_RATE=16 CHAT_N=4800 RUN_TAG=_c64_r16 python3 -u mn_run.py coloc_slo_edf_mtp
AGENT_CONC=32 CHAT_RATE=8 CHAT_N=2400 RUN_TAG=_c32_r8 python3 -u mn_run.py coloc_slo_edf_mtp
echo SLO_DONE
