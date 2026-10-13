#!/bin/bash
# Production config: CUDA graphs ON. (1) q4 serving-output gate for stock MTP, unfused 4-mode and
# fused co-launch (64/32/16, auto FA2 chunk); stop on any failure. (2) realistic A/B with graphs on.
set -uo pipefail
W=/dev/shm/w
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/fi_colaunch HF_HOME=$W/hf
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112
unset FLASHINFER_FUSED_PLAN_DEBUG SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE DISABLE_CUDA_GRAPH

OUT=$W/q4_gate_graphs_20261012; mkdir -p $OUT
DRIVER=$W/validated_upstream/tasks/fused_specdec_workload/drivers/srv_correctness.py
( export CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET=$W/datasets/ShareGPT.original.json
  export BENCH_CHUNK=512 CORRECTNESS_DECODE_TOKENS=1200 SGLANG_FUSED_DECODE_GUARD=0
  export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
  for arm in MTP_LPT_Q4G MTPMIX_LPT_Q4G MTPMIXFUSED_CL_Q4G; do
    echo "GATE_START $arm $(date +%T)"
    $W/venv/bin/python -u "$DRIVER" "$arm" "$OUT" 2>&1 | tail -3
    grep -c "cuda graph: True" /tmp/comp_$arm.jsonl 2>/dev/null
  done
  $W/venv/bin/python -u "$DRIVER" --compare "$OUT" MTP_LPT_Q4G MTPMIX_LPT_Q4G 2>&1 || exit 1
  $W/venv/bin/python -u "$DRIVER" --compare "$OUT" MTP_LPT_Q4G MTPMIXFUSED_CL_Q4G 2>&1 || exit 1
  $W/venv/bin/python -u "$DRIVER" --compare "$OUT" MTPMIX_LPT_Q4G MTPMIXFUSED_CL_Q4G 2>&1 || exit 1 )
gate=$?
echo "G_GATE_DONE status=$gate"
[ $gate = 0 ] || { echo "OUTPUT GATE FAILED -- stopping before serving"; exit 1; }

export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/datasets/ShareGPT.original.json CHAT_RATE=24 CODEX_RATE=0.1
cd $W/validated_upstream/tasks/realistic_workload_analysis
OUT=$W/graphs_agent_chat_rated24_20261012; mkdir -p $OUT
for arm in MTP_LPT MTPMIXFUSED MTPMIX_LPT; do
  echo "START $arm $(date +%T)"
  $W/venv/bin/python -u run_mixed_workloads.py --scenario agent_chat_rated --methods "$arm" --outdir "$OUT" 2>&1 | tail -8
  echo "END $arm rc=$? $(date +%T) graphs=$(grep -c 'cuda graph: True' $OUT/server_$arm.log) nographs=$(grep -c 'cuda graph: False' $OUT/server_$arm.log)"
done
echo GRAPHS_DONE
