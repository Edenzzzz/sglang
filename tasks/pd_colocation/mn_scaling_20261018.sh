#!/bin/bash
# PD point 4, node-count scaling inside the 16-GPU cap: 2 pods (4 H20s) vs 4 pods (8 H20s) at fixed per-GPU load
# (agent C = 2 per GPU, chat 2 req/s per GPU, sessions 6 per GPU). Arms: ours (pdmux + EDF uncached deadlines +
# per-step sharing with TTFT deadlines and a 24 ms ITL budget, cache-affinity routing), stock colocation + MTP, stock
# PD disaggregation (clean upstream; 1P1D at 2 pods, 2P2D at 4). The 4-pod ours / stock-coloc points are the
# 2026-10-17 runs (same cohort, same per-GPU load); disagg runs at both sizes here. Claude, 2026-10-18.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=bb854a10b8954223 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
run_size() {  # run_size <npods>
  local P=$1
  export NPODS=$P AGENT_CONC=$((4 * P)) SESSIONS=$((12 * P)) CHAT_RATE=$((4 * P))
  export CHAT_N=$((CHAT_RATE * 120)); [ $CHAT_N -lt 2400 ] && CHAT_N=2400
  export RUN_TAG=_scale_p${P}
  if [ $P != 4 ]; then
    KERNEL=lpt PREEMPT=0 EDF_UNCACHED=1 PDMUX_DIV=30 MEM_FRAC=0.70 PDMUX_SHARE_DEADLINE=1 PDMUX_ITL_BUDGET=24 \
      python3 -u mn_run.py coloc_cache_edf_pdmux
    KERNEL=lpt PREEMPT=0 python3 -u mn_run.py coloc_sgl_mtps
  fi
  SGL_PY=/dev/shm/w/up_clean/python KERNEL=lpt PREEMPT=0 python3 -u mn_run.py disagg_2p2d_mtps
}
run_size 2
run_size 4
echo SCALING_DONE
