#!/bin/bash
# 32 H20s (8 pods x 4, job 2f608c1b1a67cf09; user cap raised 2026-10-21) -- ours vs PD disaggregation at several P:D
# ratios, so disagg is judged at its best split and at scale. Qwen2.5-7B TP1 per GPU (32 workers), no speculation (the
# 8-GPU headline pair). Load = the 8-GPU text fan-out x4: 64 tasks x 64K shared prefixes, 128 agent programs, 4 rounds,
# rotating Zipf popularity, + ShareGPT chat 32 req/s for 900 s. Arms: ours (coloc_slo_edf_hc: SLO-checked hit-first
# routing, mooncake pool, EDF), PD disagg with 2 / 4 / 5 / 6 of the 8 pods prefilling (8 / 16 / 20 / 24 prefill GPUs).
# Router calibrated on one worker first. REPS (default 1). Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=8 GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b
if ! grep -q "ROUTER_ARGS=" /tmp/h20/mn_32g_calibrate.log 2>/dev/null; then
  KERNEL=mtpsplit EDF=1 SPEC=0 PREEMPT=1 CHUNK=2048 python3 -u mn_run.py calibrate | tee /tmp/h20/mn_32g_calibrate.log
  grep -q "ROUTER_ARGS=" /tmp/h20/mn_32g_calibrate.log || { echo "calibration failed"; exit 1; }
fi
export FO_TEXT=1 RA_LOG=/tmp/h20/mn_32g_calibrate.log FO_TASKS=64 AGENT_CONC=128 CHAT_RATE=32 CHAT_N=28800
export POOL_KV_TOKENS=13392000
for REPN in ${REPS:-1}; do
  for arm in ${ARMS32:-coloc_slo_edf_hc disagg_p2 disagg_p4 disagg_p5 disagg_p6}; do
    case $arm in
      coloc_*) EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_text_32g_r$REPN ARMS=$arm bash mn_fanout_8gpu_20261018.sh ;;
      *) REP=_text_32g_r$REPN ARMS=$arm bash mn_fanout_8gpu_20261018.sh ;;
    esac
  done
done
echo MN_SCALE32_DONE
