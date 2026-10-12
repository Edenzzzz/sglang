#!/bin/bash
# ITL tail: mixed steps carry a full 8192-token prefill chunk. Sweep --chunked-prefill-size for BOTH
# stock MTP and the fused 4-mode arm (stock can tune this knob too), graphs on, agent_chat_rated
# chat 24/s + codex 0.1/s. 4-mode arms keep the measured mixed-step barrier (root cause open).
set -uo pipefail
W=/dev/shm/w
until grep -q GRAPHS_BARRIER_DONE $W/graphs_barrier_ab_20261012.log 2>/dev/null; do
  pgrep -f '[g]raphs_barrier_ab' >/dev/null || break; sleep 30; done
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/port07 HF_HOME=$W/hf
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112
unset FLASHINFER_FUSED_PLAN_DEBUG SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE DISABLE_CUDA_GRAPH
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json CHAT_RATE=24 CODEX_RATE=0.1
cd $W/validated_upstream/tasks/realistic_workload_analysis
for chunk in 4096 2048; do
  for arm in MTP_LPT MTPMIXFUSED; do
    OUT=$W/chunk_v2_20261012/c${chunk}/$arm; mkdir -p $OUT
    sync_env=""; [ $arm = MTPMIXFUSED ] && sync_env="SGLANG_DEBUG_STEP_SYNC=mixed"
    echo "START c$chunk $arm $(date +%T)"
    env BENCH_CHUNK=$chunk $sync_env $W/venv/bin/python -u run_mixed_workloads.py --scenario agent_chat_rated --methods $arm --outdir $OUT 2>&1 | tail -4
    echo "END c$chunk $arm rc=$? $(date +%T) tracebacks=$(grep -c Traceback $OUT/server_$arm.log)"
  done
done
echo CHUNK_V2_DONE
