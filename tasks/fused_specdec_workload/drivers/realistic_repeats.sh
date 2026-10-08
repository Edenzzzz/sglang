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
# Interleaved repeats: LPT, SPLIT, LPT, SPLIT, ... so drift over the run hits both arms equally.
# Each arm gets its own outdir; a crash in one does not stop the rest.
for r in 1 2 3; do
  for m in LPT SPLIT; do
    rm -rf ~/.cache/flashinfer/90a/cached_ops
    python run_mixed_workloads.py --scenario agent_plus_shortchat --methods $m \
      --outdir $W/rep/r${r}_$m > $W/rep_r${r}_$m.log 2>&1
    echo "r$r $m: fallbacks=$(grep -ci fallback $W/rep/r${r}_$m/server_$m.log 2>/dev/null) tracebacks=$(grep -c Traceback $W/rep/r${r}_$m/server_$m.log 2>/dev/null) jsons=$(ls $W/rep/r${r}_$m/${m}_*.json 2>/dev/null | wc -l)"
  done
done
echo REPSDONE
