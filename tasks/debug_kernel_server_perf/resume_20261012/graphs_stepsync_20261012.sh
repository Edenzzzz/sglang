#!/bin/bash
# Graphs-on 4-mode crash, bisect by step kind. Refuted so far: phase syncs inside a mixed step,
# no in-step graph, overlap scheduler off, synchronous flashinfer plans. Only launch-blocking is
# clean. Arms: device sync after every step / after mixed steps only / after non-mixed steps only,
# and the caching allocator off (rules allocator reuse in or out).
set -uo pipefail
W=/dev/shm/w
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python HF_HOME=$W/hf
unset DISABLE_CUDA_GRAPH SGLANG_FUSED_KERNEL_EXEC
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json CHAT_RATE=24 CODEX_RATE=0.1
cd $W/validated_upstream/tasks/realistic_workload_analysis
for arm in sync_all sync_mixed sync_nonmixed nocache; do
  OUT=$W/graphs_stepsync_20261012/$arm; mkdir -p $OUT
  case $arm in
    sync_all) envs="SGLANG_DEBUG_STEP_SYNC=all";; sync_mixed) envs="SGLANG_DEBUG_STEP_SYNC=mixed";;
    sync_nonmixed) envs="SGLANG_DEBUG_STEP_SYNC=nonmixed";; nocache) envs="PYTORCH_NO_CUDA_MEMORY_CACHING=1";;
  esac
  echo "START $arm $(date +%T)"
  env $envs timeout 900 $W/venv/bin/python -u run_mixed_workloads.py --scenario agent_chat_rated --methods MTPMIX_LPT --outdir $OUT > $OUT/run.log 2>&1
  rc=$?
  echo "END $arm rc=$rc tracebacks=$(grep -c Traceback $OUT/server_MTPMIX_LPT.log) mixed=$(grep -o 'MIXEDSPEC_STEPS [0-9]*' $OUT/server_MTPMIX_LPT.log | tail -1) $(date +%T)"
  pkill -9 -f 'sglang.launch_server'; pkill -9 -f 'sglang::'; pkill -9 -f 'sglang.benchmark.serving'; sleep 5
done
echo GRAPHS_STEPSYNC_DONE
