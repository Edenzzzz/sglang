#!/bin/bash
# Agent TTFT on 8 H20s is agent-vs-agent queueing on cold 32-64K prefills at the cache-affinity node. This routes with
# the SLO check (router.py cache_slo: predicted TTFT from each node's queue and the calibrated H20 prefill curve; a node
# predicted to miss the 10 s TTFT SLO is skipped for the next-best cache hit; no feasible node -> top-hit node) over the
# pdmux + EDF (uncached deadlines) + backlog-sharing (32K) workers. Calibration: the 2026-10-17 05:25 H20 fit
# (/tmp/h20/mn_slo_calibrate.log). Same cohort / rates as mn_pdmux_share_8gpu_20261017.sh. Claude, 2026-10-17.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=bb854a10b8954223 NPODS=4 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export AGENT_CONC=16 SESSIONS=48
RA=$(grep -oE "ROUTER_ARGS=.*" /tmp/h20/mn_slo_calibrate.log | tail -1 | cut -d= -f2-)
for R in ${RATES:-16}; do
  N=$((R * 120)); [ $N -lt 2400 ] && N=2400
  export CHAT_RATE=$R CHAT_N=$N RUN_TAG=_8g_pdmuxslo_c16_r${R}
  ROUTER_ARGS="$RA --ttft-slo-ms 10000 --itl-slo-ms 100 --check-residents 0 --no-feasible-fallback cache" \
    KERNEL=lpt PREEMPT=0 EDF_UNCACHED=1 PDMUX_DIV=30 MEM_FRAC=0.70 PDMUX_SHARE=32768 python3 -u mn_run.py coloc_slo_edf_pdmux
done
echo PDMUX_SLO_DONE
