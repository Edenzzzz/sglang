#!/bin/bash
# The goal's required schedule on the re-forked single H200 (job d6a80befdd4e4da5): EAGLE3 MTP in the 4-mode fused step
# on the FA3 / FA2 split (KERNEL=mtpsplit, 112 FA3 SMs), worker EDF + chunk preemption, TTFT deadlines on uncached
# tokens, prefill chunk 2048 / 1024 / 512, vs stock MTP (coloc_sgl_mtps), same session. One worker (GPUS_PER_POD=1),
# agent C=8 + ShareGPT 8 req/s (the earlier H200 cohort load), frozen 32-64K TraceLab, 2 turns. Claude, 2026-10-18.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=d6a80befdd4e4da5 MN_CP=i18n-bd NPODS=1 GPUS_PER_POD=1 SM_FA3=112
export CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 TURNS=2 AGENT_CONC=8 SESSIONS=24 CHAT_RATE=8 CHAT_N=2400
KERNEL=lpt PREEMPT=0 RUN_TAG=_h200_c8 python3 -u mn_run.py coloc_sgl_mtps
for CH in ${CHUNKS:-2048 1024 512}; do
  RUN_TAG=_h200_4mode_unc_ch${CH}_c8 KERNEL=mtpsplit PREEMPT=1 CHUNK=$CH EDF_UNCACHED=1 python3 -u mn_run.py coloc_cache_edf_mtp
done
echo H200_FOURMODE_DONE
