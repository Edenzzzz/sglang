#!/bin/bash
# Realistic serving A/B: coding-agent TraceLab replay + high-rate short chat
# (scenario agent_plus_shortchat), LPT (persistent FA2 on everything) vs SPLIT.
set -u
W=/dev/shm/w
cd $W
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [ -n "$busy" ]; then echo "REFUSING: GPU busy: $busy"; exit 1; fi
[ -e $W/.venv ] || ln -s $W/venv $W/.venv
. $W/venv/bin/activate
export PYTHONPATH=$W/python:$W/fi HF_HOME=$W/hf FI_ROOT=$W/fi TORCH_CUDA_ARCH_LIST=9.0a
export MODEL=$W/qwen7b
export TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export UP_STAGES=2 MMA_PV_RS=1 SCHED_LPT=0
for f in "$TRACE_PATH" "$CHAT_DATASET" "$MODEL/config.json" "$W/python/sglang/__init__.py"; do
  [ -e "$f" ] || { echo "MISSING $f"; exit 1; }
done
# stale flashinfer JIT modules silently ignore source edits -- start clean
rm -rf ~/.cache/flashinfer/90a/cached_ops
cd $W/tasks/realistic_workload_analysis
python run_mixed_workloads.py --scenario agent_plus_shortchat --methods LPT,SPLIT \
  --outdir $W/real_ab 2>&1 | tail -40
echo "--- fallbacks per arm (must be 0):"
for m in LPT SPLIT; do
  echo "$m: $(grep -ci 'fallback' $W/real_ab/server_$m.log 2>/dev/null)"
done
python summarize_mixed_workloads.py $W/real_ab --baseline LPT --method SPLIT 2>&1 | tail -30
echo REALABDONE
