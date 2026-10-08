#!/bin/bash
# MTP load sweep: stock MTP vs the 4-mode step (FA2 / split) at chat rates 4, 8, 12, 16 /s with
# codex sessions rate-limited. Finds each arm's capacity and compares latency below saturation.
set -u
W=/dev/shm/w
. $W/venv/bin/activate
export PYTHONPATH=$W/python:$W/fi HF_HOME=$W/hf FI_ROOT=$W/fi TORCH_CUDA_ARCH_LIST=9.0a
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export UP_STAGES=2 MMA_PV_RS=1 SCHED_LPT=0
cd $W/tasks/realistic_workload_analysis
for rate in 4 8 12 16; do
  for m in MTP_LPT MTPMIX_LPT MTPMIX_SPLIT; do
    o=$W/mtpload/rate$rate/r1_$m
    rm -rf $o
    CHAT_RATE=$rate python run_mixed_workloads.py --scenario agent_chat_rated --methods $m --outdir $o > $W/mtpload_${rate}_$m.log 2>&1
    L=$o/server_$m.log
    echo "rate$rate $m: fallbacks=$(grep -ci fallback $L) tracebacks=$(grep -c Traceback $L) jsons=$(ls $o/*.json 2>/dev/null | wc -l) mixed=[$(grep -oE 'MIXED_SPEC steps=[0-9]+' $L | tail -1)]"
  done
  python3 $W/repsum.py --root $W/mtpload/rate$rate --base MTP_LPT | grep -E "^r1|mean over" -A0
done
echo MTPLOADDONE
