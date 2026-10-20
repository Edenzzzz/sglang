#!/bin/bash
# Llama-3.3-70B-Instruct at TP4 on 4 pods x 4 H20 (job f3b4627e0864b06f, 2026-10-21): 4 replicas, MTP with the trained
# EAGLE3 head (yuhuili/EAGLE3-LLaMA3.3-Instruct-70B). On our raw-text prompts that head accepts only 1.29-1.39 tokens
# per verify (gate_70b_20261021.sh), so acceptance is fixed at 2.1 for every arm (SIM_ACC, the Qwen-7B head's measured
# value) while the head's draft work and the verify stay real. After the output gate (gate_70b_20261021.sh) the
# router's SLO model is calibrated on one 70B TP4 worker; then the text fan-out (16 x 64K shared prefixes) + ShareGPT
# chat, at a load scaled to 4 replicas (AGENT_CONC 8, chat 2 req/s, 900 s): ours (4-mode MTP, SLO-checked hit-first
# routing, pool, EDF) vs PD disagg 2P2D (2 prefill + 2 decode TP4 pods, MTP on every role), two repeats. Pool capacity
# for the router scales by KV per token (320 vs 57 KB for Qwen-7B). Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=f3b4627e0864b06f MN_CP=cn-seed NPODS=4 GPUS_PER_POD=4 TP=4 MODEL_DIR=llama70b DRAFT_DIR=llama70b_eagle3_sgl
export SM_FA3=${SM_FA3:-66} MEM_FRAC=${MEM_FRAC:-0.85} POOL_KV_TOKENS=${POOL_KV_TOKENS:-596000} SIM_ACC=${SIM_ACC:-2.1}
if ! grep -q "ROUTER_ARGS=" /tmp/h20/mn_70b_calibrate.log 2>/dev/null; then
  KERNEL=mtpsplit EDF=1 SPEC=1 PREEMPT=1 CHUNK=2048 python3 -u mn_run.py calibrate | tee /tmp/h20/mn_70b_calibrate.log
  grep -q "ROUTER_ARGS=" /tmp/h20/mn_70b_calibrate.log || { echo "calibration failed"; exit 1; }
fi
export FO_TEXT=1 RA_LOG=/tmp/h20/mn_70b_calibrate.log AGENT_CONC=${AGENT_CONC:-8} CHAT_RATE=${CHAT_RATE:-2} CHAT_N=${CHAT_N:-1800}
for REPN in ${REPS:-1 2}; do
  EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_text_70b${LOAD_TAG:-}_r$REPN ARMS=coloc_slo_edf_mtp_hc \
    bash mn_fanout_8gpu_20261018.sh
  REP=_text_70b${LOAD_TAG:-}_r$REPN ARMS=disagg_2p2d_mtps bash mn_fanout_8gpu_20261018.sh
done
echo MN_70B_DONE
