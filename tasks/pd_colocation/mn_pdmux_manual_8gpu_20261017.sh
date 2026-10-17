#!/bin/bash
# Prefill-heavy pdmux SM groups on 8 H20s: divide_sm caps prefill at 56 of 78 SMs whenever decode runs, and on H20
# prefill (compute) is the bottleneck -- the 2026-10-17 run lost TTFT 1.9-3.2x to stock non-speculative colocation.
# manual_divisions 64/14 below 24 decoding requests, 56/22 from 24, 48/30 from 40. Same cohort / rates / router as
# mn_pdmux_8gpu_20261017.sh (baselines from that session). Claude, 2026-10-17.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=bb854a10b8954223 NPODS=4 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export AGENT_CONC=16 SESSIONS=48
for R in 16 32; do
  N=$((R * 120)); [ $N -lt 2400 ] && N=2400
  export CHAT_RATE=$R CHAT_N=$N RUN_TAG=_8g_pdmuxman_c16_r${R}
  KERNEL=lpt PREEMPT=0 EDF_UNCACHED=1 MEM_FRAC=0.70 PDMUX_MANUAL='[[64,14,0],[56,22,24],[48,30,40]]' \
    python3 -u mn_run.py coloc_cache_edf_pdmux
done
echo PDMUX_MANUAL_DONE
