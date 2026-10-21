#!/bin/bash
# GLM-4.5-Air single node: one 4-GPU H20 pod at TP4, its own MTP layer (NEXTN, accept ~2.5-2.75 in the output gate),
# standard mix (agents C=8 over 24 sessions, 32-64K TraceLab, + ShareGPT chat 8 req/s), chat template; 2026-10-21.
# Gate passed first (gate_glm_20261021.sh). At the standard load stock + MTP was overloaded (chat TTFT 136 s, agent
# 78 s, chat 772 of ~1640 tok/s offered; `_glm_ct_fa3be_r1`), so two lower loads: L=2 (agents C=2 over 6 sessions,
# chat 2 req/s) and L=4 (C=4 over 12 sessions, chat 4 req/s), 300 s of chat each. Stock on FA3 (its Hopper default): prefill-prioritized + MTP, mixed chunk
# and prefill-prioritized without MTP. Ours on FA3 for every attention call with the per-step split policy: + MTP
# (4-mode, rows cap + SLO admission) and without MTP. One repeat to place the load. Claude.
set -u
cd "$(dirname "$0")"
export CHAT_TEMPLATE=1
for L in 2 4; do
export AGENT_CONC=$L SESSIONS=$((3 * L)) CHAT_RATE=$L CHAT_N=$((300 * L))
ATTN_BACKEND=fa3 TAG=_glm_L${L}_ct_fa3be_r1 ARMS="stock_mtp_pp stock stock_pp" bash mn_h20_single_20261021.sh
SPLIT_ROWS_FA3=1 VERIFY_FA3=1 SPLIT_POLICY=1 SPLIT_OVERLAP=1 DECODE_ROWS=48:20:24000:80:0.9 \
  WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 48" TAG=_glm_L${L}_ct_rows48adm_pol_r1 ARMS=ours_mtp \
  bash mn_h20_single_20261021.sh
SPLIT_ROWS_FA3=1 DECODE_FA3=1 SPLIT_POLICY=1 TAG=_glm_L${L}_ct_pol_r1 ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 \
  bash mn_h20_single_20261021.sh
done
echo MN_GLM_SINGLE_DONE
