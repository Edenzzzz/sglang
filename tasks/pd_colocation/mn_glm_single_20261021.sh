#!/bin/bash
# GLM-4.5-Air single node: one 4-GPU H20 pod at TP4, its own MTP layer (NEXTN, accept ~2.5-2.75 in the output gate),
# standard mix (agents C=8 over 24 sessions, 32-64K TraceLab, + ShareGPT chat 8 req/s), chat template; 2026-10-21.
# Gate passed first (gate_glm_20261021.sh). Stock on FA3 (its Hopper default): prefill-prioritized + MTP, mixed chunk
# and prefill-prioritized without MTP. Ours on FA3 for every attention call with the per-step split policy: + MTP
# (4-mode, rows cap + SLO admission) and without MTP. One repeat to place the load. Claude.
set -u
cd "$(dirname "$0")"
export CHAT_TEMPLATE=1
ATTN_BACKEND=fa3 TAG=_glm_ct_fa3be_r1 ARMS="stock_mtp_pp stock stock_pp" bash mn_h20_single_20261021.sh
SPLIT_ROWS_FA3=1 VERIFY_FA3=1 SPLIT_POLICY=1 SPLIT_OVERLAP=1 DECODE_ROWS=48:20:24000:80:0.9 \
  WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 48" TAG=_glm_ct_rows48adm_pol_r1 ARMS=ours_mtp \
  bash mn_h20_single_20261021.sh
SPLIT_ROWS_FA3=1 DECODE_FA3=1 SPLIT_POLICY=1 TAG=_glm_ct_pol_r1 ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 \
  bash mn_h20_single_20261021.sh
echo MN_GLM_SINGLE_DONE
