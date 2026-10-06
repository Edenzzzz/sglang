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
# DIAGNOSTIC-SWITCH GUARD. Three measurement rounds were invalidated by a diagnostic left in
# its non-production state: MERGE_NO_STORE=1 (skipped work), and the device-side TRACE printf
# (a 3.8 GB server log and a baseline that measured printf). Refuse to run if any is set.
H=$W/fi/include/flashinfer/attention
bad=$(grep -lE '^#if 1  // (TRACE|TEMPORARILY)' $H/persistent.cuh 2>/dev/null
      grep -lE 'define FLASHINFER_MERGE_NO_STORE 1' $H/jit_merge.cuh 2>/dev/null
      grep -lE 'define FLASHINFER_FORCE_CLASSIC_REDUCE 1' $H/persistent.cuh $H/persistent_template.cuh 2>/dev/null
      grep -lE 'define FLASHINFER_PI_ZERO_DEFAULT 1' $H/scheduler.cuh 2>/dev/null
      grep -lE 'define FLASHINFER_PARTIAL_LSE_TILE_REL 0' $H/persistent.cuh 2>/dev/null)
if [ -n "$bad" ]; then echo "REFUSING: diagnostic switch left on in: $bad"; exit 1; fi
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
