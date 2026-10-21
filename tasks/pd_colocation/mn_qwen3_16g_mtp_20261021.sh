#!/bin/bash
# Qwen3-8B + trained EAGLE3 head (AngelSlim, real acceptance ~1.8-2.0) on 16 H20s (4 pods x 4, job 2f608c1b1a67cf09),
# chat template on, 2026-10-21. Both systems on FA3: ours runs split rows and verify on FA3 (SPLIT_ROWS_FA3=1
# VERIFY_FA3=1; pods' flashinfer_backend.py md5 ed98c8e1fd51), disagg runs ATTN_BACKEND=fa3 (Hopper default). Load is
# half the Qwen2.5-7B 16-GPU MTP load (Qwen3-8B overloaded every system at the Qwen2.5 load on the H200): 16 tasks /
# 32 agent programs, chat 8 req/s, 900 s. The router's perf model is calibrated on ours' worker config first.
# Ours (4-mode MTP, SLO-checked hit-first routing, pool, EDF) vs disagg + MTP at 2 + 2 and 3 + 1. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=2f608c1b1a67cf09 MN_CP=cn-seed GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen3_8b DRAFT_DIR=qwen3_8b_eagle3
export CHAT_TEMPLATE=1 FO_TEXT=1 NPODS=4 FO_TASKS=16 AGENT_CONC=32 CHAT_RATE=8 CHAT_N=7200 POOL_KV_TOKENS=6696000
KERNEL=mtpsplit EDF=1 SPEC=1 PREEMPT=1 CHUNK=2048 SPLIT_ROWS_FA3=1 VERIFY_FA3=1 \
  python3 -u mn_run.py calibrate > /tmp/h20/mn_q3_calibrate.log 2>&1
grep -q "ROUTER_ARGS=" /tmp/h20/mn_q3_calibrate.log || { echo "calibration failed"; tail -20 /tmp/h20/mn_q3_calibrate.log; exit 1; }
export RA_LOG=/tmp/h20/mn_q3_calibrate.log
SPLIT_ROWS_FA3=1 VERIFY_FA3=1 EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_q3_16g_mtp_fa3_r1 \
  ARMS=coloc_slo_edf_mtp_hc bash mn_fanout_8gpu_20261018.sh
for arm in disagg_p2_mtps disagg_p3_mtps; do ATTN_BACKEND=fa3 REP=_q3_16g_mtp_fa3_r1 ARMS=$arm bash mn_fanout_8gpu_20261018.sh; done
echo MN_Q3_16G_MTP_DONE
