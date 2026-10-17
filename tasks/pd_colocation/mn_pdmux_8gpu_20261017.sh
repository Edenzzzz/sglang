#!/bin/bash
# 8-GPU (4 H20 pods x 2) A/B of the single-GPU winner -- PD multiplexing + worker-side EDF with uncached-token
# deadlines, decode_bs_divisor 30 -- under cache-affinity routing, vs stock colocation with MTP (the multi-node
# baseline so far) and stock colocation without speculation, same session. Agent tenant: frozen TraceLab 32-64K at
# program concurrency C=16 (ThunderAgent), 2 turns; chat: ShareGPT at 16 / 32 req/s (2 / 4 per GPU). Ours runs first
# at each rate so a pdmux start-up failure on H20 (78 SMs) shows up before the baselines spend their time.
# Claude, 2026-10-17.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=bb854a10b8954223 NPODS=4 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export AGENT_CONC=16 SESSIONS=48
for R in 16 32; do
  N=$((R * 120)); [ $N -lt 2400 ] && N=2400
  export CHAT_RATE=$R CHAT_N=$N RUN_TAG=_8g_pdmux_c16_r${R}
  KERNEL=lpt PREEMPT=0 EDF_UNCACHED=1 PDMUX_DIV=30 python3 -u mn_run.py coloc_cache_edf_pdmux
  KERNEL=lpt PREEMPT=0 python3 -u mn_run.py coloc_sgl_mtps
  KERNEL=lpt PREEMPT=0 python3 -u mn_run.py coloc_sgl
done
echo PDMUX_8GPU_DONE
