#!/bin/bash
# The goal's required schedule on the re-forked single H200 (job dadf94f0e1da7238): EAGLE3 MTP in the 4-mode fused step
# on the FA3 / FA2 split (KERNEL=mtpsplit, 112 FA3 SMs), worker EDF + chunk preemption, TTFT deadlines on uncached
# tokens, prefill chunk 2048 / 1024 / 512, vs stock MTP (coloc_sgl_mtps), same session. One worker (GPUS_PER_POD=1),
# agent C=8 + ShareGPT 8 req/s (the earlier H200 cohort load), frozen 32-64K TraceLab, 2 turns. Claude, 2026-10-18.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=dadf94f0e1da7238 MN_CP=i18n-bd NPODS=1 GPUS_PER_POD=1 SM_FA3=112
export CHAT_SHAREGPT=1 MIN_CTX=${MIN_CTX:-32768} MAX_CTX=${MAX_CTX:-65536} TURNS=${TURNS:-2} AGENT_CONC=${AGENT_CONC:-8}
export SESSIONS=${SESSIONS:-24} CHAT_RATE=${CHAT_RATE:-8} CHAT_N=${CHAT_N:-2400}
[ "${SKIP_BASE:-0}" = 1 ] || KERNEL=lpt PREEMPT=0 RUN_TAG=_h200_c8${SPEC_TAG:-} python3 -u mn_run.py coloc_sgl_mtps
for CH in ${CHUNKS:-2048 1024 512}; do
  RUN_TAG=_h200_4mode_unc_ch${CH}${DEADLINE_CHUNK:+_dl${DEADLINE_CHUNK//:/-}}${STEP_BUDGET:+_sb${STEP_BUDGET%%:*}}${TTFT_SCALE_MS:+_ts${TTFT_SCALE_MS}}${CKPT_EVERY:+_ck${CKPT_EVERY}}${LONG_DEADLINE:+_ld${LONG_DEADLINE//:/-}}_c8${SPEC_TAG:-}${REP_SUFFIX:-} KERNEL=mtpsplit PREEMPT=1 CHUNK=$CH EDF_UNCACHED=1 python3 -u mn_run.py coloc_cache_edf_mtp
done
echo H200_FOURMODE_DONE
