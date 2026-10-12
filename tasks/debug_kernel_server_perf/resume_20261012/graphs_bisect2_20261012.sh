#!/bin/bash
# Graphs-on 4-mode crash disappears under CUDA_LAUNCH_BLOCKING=1 + SGLANG_DEBUG_MIX_SYNC=1 (repro
# completed 1651 mixed steps clean), so it is an ordering race. One toggle per arm, unfused 4-mode,
# agent_chat_rated, graphs on; an arm "crashes" if its server log holds a Traceback.
#   sync_only    SGLANG_DEBUG_MIX_SYNC=1         (device sync at each 4-mode phase boundary)
#   block_only   CUDA_LAUNCH_BLOCKING=1
#   no_mixgraph  SGLANG_MIXED_SPEC_DISABLE_VERIFY_GRAPH=1 (diagnostic: no graph inside mixed steps)
#   control      nothing (expected to crash)
set -uo pipefail
W=/dev/shm/w
export PYTHONPATH=$W/validated_upstream/python HF_HOME=$W/hf
unset DISABLE_CUDA_GRAPH SGLANG_FUSED_KERNEL_EXEC
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json CHAT_RATE=24 CODEX_RATE=0.1
cd $W/validated_upstream/tasks/realistic_workload_analysis
until grep -q GRAPHS_BISECT_DONE $W/graphs_bisect_20261012.log 2>/dev/null; do sleep 20; done
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
for arm in no_overlap control2; do
  OUT=$W/graphs_bisect2_20261012/$arm; mkdir -p $OUT
  case $arm in
    control2) envs="";; no_overlap) envs="SERVER_EXTRA_ARGS=--disable-overlap-schedule";;
  esac
  echo "START $arm $(date +%T)"
  env $envs timeout 900 $W/venv/bin/python -u run_mixed_workloads.py --scenario agent_chat_rated --methods MTPMIX_LPT --outdir $OUT > $OUT/run.log 2>&1
  rc=$?
  tb=$(grep -c Traceback $OUT/server_MTPMIX_LPT.log)
  echo "END $arm rc=$rc tracebacks=$tb mixed=$(grep -o 'MIXEDSPEC_STEPS [0-9]*' $OUT/server_MTPMIX_LPT.log | tail -1) $(date +%T)"
  pkill -9 -f 'sglang.launch_server' ; pkill -9 -f 'sglang::'; pkill -9 -f 'sglang.benchmark.serving'; sleep 5
done
echo GRAPHS_BISECT2_DONE
