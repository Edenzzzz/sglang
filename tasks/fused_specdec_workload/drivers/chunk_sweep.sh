#!/bin/bash
# Chunked-prefill size sweep, LPT vs SPLIT, one interleaved pair per size (8192 is the
# 3-round A/B already recorded). Every arm reports fallbacks.
set -u
W=/dev/shm/w
. $W/venv/bin/activate
export PYTHONPATH=$W/python:$W/fi HF_HOME=$W/hf FI_ROOT=$W/fi TORCH_CUDA_ARCH_LIST=9.0a
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export UP_STAGES=2 MMA_PV_RS=1 SCHED_LPT=0
cd $W/tasks/realistic_workload_analysis
for c in 4096 16384; do
  for m in LPT SPLIT; do
    o=$W/chunk/c${c}/r1_$m
    rm -rf $o
    BENCH_CHUNK=$c python run_mixed_workloads.py --scenario agent_plus_shortchat --methods $m --outdir $o > $W/chunk_c${c}_$m.log 2>&1
    echo "c$c $m: fallbacks=$(grep -ci fallback $o/server_$m.log) tracebacks=$(grep -c Traceback $o/server_$m.log) jsons=$(ls $o/*.json 2>/dev/null | wc -l)"
  done
  python3 $W/repsum.py --root $W/chunk/c${c} --base LPT | sed -n '/mean over/,$p'
done
echo CHUNKDONE
