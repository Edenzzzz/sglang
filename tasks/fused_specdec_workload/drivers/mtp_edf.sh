#!/bin/bash
# Goal point 3: EDF on the 4-mode MTP server, with and without acceptance length in the ITL
# budget. 1) traced run (step model + online estimator scoring), 2) fit the step model,
# 3) EDF arms at the same load: EDF0 a=1, EDFG global EWMA, EDFR per-request EWMA.
#   mtp_edf.sh [rate] [ttft_slo_ms] [itl_slo_ms]
set -u
W=/dev/shm/w
. $W/venv/bin/activate
export PYTHONPATH=$W/python:$W/fi HF_HOME=$W/hf FI_ROOT=$W/fi TORCH_CUDA_ARCH_LIST=9.0a
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export UP_STAGES=2 MMA_PV_RS=1 SCHED_LPT=0
RATE=${1:-24}; export TTFT_SLO_MS=${2:-2000} ITL_SLO_MS=${3:-50}
R=$W/mtpedf/rate$RATE${TAG:-}
cd $W/tasks/realistic_workload_analysis
summ() {
  L=$1/server_$2.log
  echo "$2: fallbacks=$(grep -ci fallback $L) tracebacks=$(grep -c Traceback $L) jsons=$(ls $1/*.json 2>/dev/null | wc -l) $(grep -oE 'EDF_ACCEPT .*' $L | tail -1)"
}
o=$R/trace_MTPMIX_SPLIT
# STEP_MODEL reuses a step model fitted at the same load (the SLO does not change it).
if [ -n "${STEP_MODEL:-}" ]; then mkdir -p $R; cp $STEP_MODEL $R/mtp_step.json; fi
if [ ! -s $R/mtp_step.json ]; then
  rm -rf $o
  SGLANG_STEP_TRACE=1 SGLANG_EDF_ACCEPT_TRACE=1 SGLANG_EDF_ACCEPT_LOG_EVERY=5000 CHAT_RATE=$RATE \
    python run_mixed_workloads.py --scenario agent_chat_rated --methods MTPMIX_SPLIT --outdir $o > $R.trace.log 2>&1
  summ $o MTPMIX_SPLIT
  python3 $W/tasks/perf_model/fit_mtp_step.py $o/server_MTPMIX_SPLIT.log --out $R/mtp_step.json | grep -E "mape|n_steps"
fi
export MTP_STEP_MODEL=$R/mtp_step.json
for m in ${ARMS:-MTPMIX_SPLIT MTPMIX_SPLIT_EDF0 MTPMIX_SPLIT_EDFG MTPMIX_SPLIT_EDFR}; do
  o=$R/r1_$m; rm -rf $o
  SGLANG_EDF_ACCEPT_LOG_EVERY=5000 CHAT_RATE=$RATE \
    python run_mixed_workloads.py --scenario agent_chat_rated --methods $m --outdir $o > $R.$m.log 2>&1
  summ $o $m
done
for f in "" --scaled; do python3 $W/slo_attain.py --root $R --ttft $TTFT_SLO_MS --itl $ITL_SLO_MS $f; done
echo MTPEDFDONE
