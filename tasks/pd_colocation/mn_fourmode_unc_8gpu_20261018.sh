#!/bin/bash
# The goal's required schedule on 8 H20s: EAGLE3 MTP in the 4-mode fused step (prefill chunk + verify rows in one
# target forward on the FA3 / FA2 split kernel, KERNEL=mtpsplit SPEC=1) with worker EDF + chunk preemption, now with
# TTFT deadlines on uncached tokens (EDF_UNCACHED=1), prefill chunk 2048 vs 1024, cache-affinity routing. Baseline:
# stock colocation + MTP (2026-10-17, same cohort, chat 16 req/s, agent C=16). Claude, 2026-10-18.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=bb854a10b8954223 NPODS=4 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export AGENT_CONC=16 SESSIONS=48 CHAT_RATE=${CHAT_RATE:-16}
export CHAT_N=$((CHAT_RATE * 120)); [ $CHAT_N -lt 2400 ] && CHAT_N=2400
for CH in ${CHUNKS:-2048 1024}; do
  RUN_TAG=_8g_4mode_unc_ch${CH}${DEADLINE_CHUNK:+_dl${DEADLINE_CHUNK//:/-}}_r${CHAT_RATE} KERNEL=mtpsplit PREEMPT=1 CHUNK=$CH EDF_UNCACHED=1 \
    python3 -u mn_run.py coloc_cache_edf_mtp
done
echo FOURMODE_DONE
