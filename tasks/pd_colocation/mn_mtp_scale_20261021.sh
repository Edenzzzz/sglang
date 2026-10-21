#!/bin/bash
# Qwen2.5-7B with MTP at real acceptance (trained EAGLE3 head, accept ~2.0), chat prompts through the chat template,
# 16 and 32 H20s (job 2f608c1b1a67cf09), 2026-10-21. Load scales with GPUs from the 8-GPU text fan-out: 32 GPUs = 64
# tasks, 128 programs, chat 32 req/s; 16 GPUs = 32 tasks, 64 programs, chat 16 req/s. Ours (4-mode MTP, SLO-checked
# hit-first routing, pool, EDF) vs PD disagg + MTP on every role at the ratios that bracket the no-spec best (5 / 8).
# Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=2f608c1b1a67cf09 MN_CP=cn-seed GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1 FO_TEXT=1
export RA_LOG=/tmp/h20/mn_32g_calibrate.log
for SCALE in ${SCALES:-32 16}; do
  if [ $SCALE = 32 ]; then
    export NPODS=8 FO_TASKS=64 AGENT_CONC=128 CHAT_RATE=32 CHAT_N=28800 POOL_KV_TOKENS=13392000; DIS="disagg_p5_mtps disagg_p6_mtps"
  else
    export NPODS=4 FO_TASKS=32 AGENT_CONC=64 CHAT_RATE=16 CHAT_N=14400 POOL_KV_TOKENS=6696000; DIS="disagg_p2_mtps disagg_p3_mtps"
  fi
  EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_text_${SCALE}g_mtp_r1 ARMS=coloc_slo_edf_mtp_hc \
    bash mn_fanout_8gpu_20261018.sh
  for arm in $DIS; do REP=_text_${SCALE}g_mtp_r1 ARMS=$arm bash mn_fanout_8gpu_20261018.sh; done
done
echo MN_MTP_SCALE_DONE
