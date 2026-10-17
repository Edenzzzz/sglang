#!/bin/bash
# TP=2 per pod (4 workers over 8 H20s): a cold 32-64K agent prefill spreads over two GPUs, halving the per-request
# prefill floor that bounds agent TTFT at TP=1 (~5.9 s on this cohort). Ours (pdmux + EDF with uncached deadlines +
# 32K backlog sharing, cache-affinity routing; see below) vs stock colocation without speculation, both at TP=2, chat 16 req/s,
# agent C=16. pdmux + TP=2 hung (2026-10-17 21:33: all GPUs at 100% with no progress for 30 min, killed), so ours here is
# EDF with uncached-token deadlines on chunked colocation + cache-affinity routing. Claude, 2026-10-17.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=bb854a10b8954223 NPODS=4 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export AGENT_CONC=16 SESSIONS=48 TP=2
for R in ${RATES:-16}; do
  N=$((R * 120)); [ $N -lt 2400 ] && N=2400
  export CHAT_RATE=$R CHAT_N=$N RUN_TAG=_8g_tp2_c16_r${R}
  KERNEL=lpt PREEMPT=0 EDF_UNCACHED=1 python3 -u mn_run.py coloc_cache_edf
  KERNEL=lpt PREEMPT=0 python3 -u mn_run.py coloc_sgl
done
echo TP2_DONE
