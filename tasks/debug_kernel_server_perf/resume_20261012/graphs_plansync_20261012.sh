#!/bin/bash
# Graphs-on 4-mode crash, hypothesis: pinned-staging reuse across two flashinfer plans issued in one
# step before the GPU executes the first plan's async copy. Bisect so far: control, phase syncs,
# no in-step graph and no overlap scheduler all crash at the first or fourth mixed step; only
# CUDA_LAUNCH_BLOCKING=1 runs clean (1651 mixed steps). SGLANG_FI_PLAN_SYNC=1 makes every flashinfer
# plan (class methods and the graph fast plans) copy synchronously and nothing else.
set -uo pipefail
W=/dev/shm/w
until grep -q B_BACKFILL_DONE $W/b_backfill_20261012.log 2>/dev/null; do
  pgrep -f '[b]_backfill_20261012' >/dev/null || break; sleep 30; done
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python HF_HOME=$W/hf
unset DISABLE_CUDA_GRAPH SGLANG_FUSED_KERNEL_EXEC
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json CHAT_RATE=24 CODEX_RATE=0.1
cd $W/validated_upstream/tasks/realistic_workload_analysis
for arm in plansync plansync_rep; do
  OUT=$W/graphs_plansync_20261012/$arm; mkdir -p $OUT
  echo "START $arm $(date +%T)"
  SGLANG_FI_PLAN_SYNC=1 timeout 900 $W/venv/bin/python -u run_mixed_workloads.py --scenario agent_chat_rated --methods MTPMIX_LPT --outdir $OUT > $OUT/run.log 2>&1
  rc=$?
  echo "END $arm rc=$rc tracebacks=$(grep -c Traceback $OUT/server_MTPMIX_LPT.log) mixed=$(grep -o 'MIXEDSPEC_STEPS [0-9]*' $OUT/server_MTPMIX_LPT.log | tail -1) graphs=$(grep -c 'cuda graph: True' $OUT/server_MTPMIX_LPT.log) $(date +%T)"
  pkill -9 -f 'sglang.launch_server'; pkill -9 -f 'sglang::'; pkill -9 -f 'sglang.benchmark.serving'; sleep 5
done
echo GRAPHS_PLANSYNC_DONE
