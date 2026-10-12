#!/bin/bash
# ITL tail: P99 ITL (~111 ms) sits on steps that carry a full 8192-token prefill chunk against
# 32-48K codex contexts. Sweep --chunked-prefill-size for BOTH stock MTP and the fused co-launch
# (stock can tune this knob too), CUDA graphs on, same realistic workload.
set -uo pipefail
W=/dev/shm/w
until grep -q GRAPHS_DONE $W/graphs_on_v2_20261012.log 2>/dev/null; do
  pgrep -f '[g]raphs_on_v2' >/dev/null || { echo "graphs chain ended without GRAPHS_DONE"; exit 1; }; sleep 60; done
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
  OUT=$W/chunk${chunk}_agent_chat_rated24_20261012; mkdir -p $OUT
  for arm in MTP_LPT MTPMIXFUSED; do
    echo "START c$chunk $arm $(date +%T)"
    BENCH_CHUNK=$chunk $W/venv/bin/python -u run_mixed_workloads.py --scenario agent_chat_rated --methods "$arm" --outdir "$OUT" 2>&1 | tail -8
    echo "END c$chunk $arm rc=$? $(date +%T)"
  done
done
echo CHUNK_SWEEP_DONE
