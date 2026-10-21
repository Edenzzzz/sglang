#!/bin/bash
# Where ours + MTP loses to stock prefill-prioritized + MTP on FA3 (131 vs 161 tok/s, H200, chat 8 req/s; 2026-10-21),
# after h200_split_policy_20261021.sh (pid $1). Every arm one repeat, chat template, real head:
#   A  stock's schedule (prefill-prioritized, stock MTP, no mixed chunk, no EDF) on OUR backend: every extend step on
#      FA3 via the split policy, verify on FA3 -- isolates backend from scheduling (draft model stays on flashinfer)
#   B  ours + MTP (4-mode step, EDF, deadline chunk) WITHOUT decode-rows cap and SLO admission
#   C  ours + MTP 4-mode on FA3 but at a fixed chunk 2048 (no deadline chunk), with rows + admission
# Claude.
set -u
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
export CHAT_TEMPLATE=1 PATH=$HOME/.merlin-cli/bin:$PATH
FA="SPLIT_ROWS_FA3=1 VERIFY_FA3=1 SPLIT_POLICY=1"
(export SID=dadf94f0e1da7238 MN_CP=i18n-bd NPODS=1 GPUS_PER_POD=1 SM_FA3=112 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 \
   TURNS=2 AGENT_CONC=8 SESSIONS=24 CHAT_RATE=8 CHAT_N=2400
 env $FA MIXED_CHUNK=0 KERNEL=mtpsplit PREEMPT=0 RUN_TAG=_h200_pp_c8_ct_ourbe python3 -u mn_run.py coloc_sgl_mtps)
env $FA SPLIT_OVERLAP=1 TAG=_ct_norows_pol ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
env $FA SPLIT_OVERLAP=1 DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 48" \
  TAG=_ct_rows48adm_pol ARMS=ours_mtp_nodc bash mn_h200_nospec_20261018.sh
echo H200_MTP_GAP_DONE
