#!/bin/bash
# Graphs-on realistic A/B with the measured mixed-step barrier. Root cause of the graphs-on 4-mode
# crash is still open: a device sync after MIXED steps cures it (1651 mixed steps clean), after
# non-mixed steps does not, and launch-blocking cures it; phase syncs inside the step, no in-step
# graph, overlap off and synchronous flashinfer plans do not. The barrier (SGLANG_DEBUG_STEP_SYNC=
# mixed) can only cost the 4-mode arms CPU/GPU overlap; stock MTP runs without it (measured graphs
# on earlier today: chat 1283.6 tok/s). Gate first, then serving; also the A3 dispatch rule arm.
set -uo pipefail
W=/dev/shm/w
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/port07 HF_HOME=$W/hf
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112 SGLANG_DEBUG_STEP_SYNC=mixed
unset FLASHINFER_FUSED_PLAN_DEBUG SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE DISABLE_CUDA_GRAPH
# (1) output gate, graphs on, barrier on, incl. the A3 rule
OUT=$W/q4_gate_barrier_20261012; mkdir -p $OUT
cp $W/q4_gate_graphs_20261012/MTP_LPT_Q4G.json $OUT/
DRIVER=$W/validated_upstream/tasks/fused_specdec_workload/drivers/srv_correctness.py
( export CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
  export BENCH_CHUNK=512 CORRECTNESS_DECODE_TOKENS=1200 SGLANG_FUSED_DECODE_GUARD=0 MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
  SGLANG_FUSED_DISPATCH=a3 $W/venv/bin/python -u "$DRIVER" MTPMIXFUSED_A3_Q4G "$OUT" 2>&1 | tail -2
  $W/venv/bin/python -u "$DRIVER" --compare "$OUT" MTP_LPT_Q4G MTPMIXFUSED_A3_Q4G 2>&1 || exit 1 )
gate=$?; echo "BARRIER_GATE_DONE status=$gate"; [ $gate = 0 ] || exit 1
# (2) realistic A/B, graphs on
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json CHAT_RATE=24 CODEX_RATE=0.1
OUT=$W/graphs_barrier_agent_chat_rated24_20261012; mkdir -p $OUT
cd $W/validated_upstream/tasks/realistic_workload_analysis
for arm in MTPMIXFUSED MTPMIX_LPT MTPMIXFUSED_A3; do
  method=$arm; extra=""
  [ $arm = MTPMIXFUSED_A3 ] && { method=MTPMIXFUSED; extra="SGLANG_FUSED_DISPATCH=a3"; }
  mkdir -p $OUT/$arm
  echo "START $arm $(date +%T)"
  env $extra $W/venv/bin/python -u run_mixed_workloads.py --scenario agent_chat_rated --methods "$method" --outdir "$OUT/$arm" 2>&1 | tail -6
  echo "END $arm rc=$? $(date +%T) graphs=$(grep -c 'cuda graph: True' $OUT/$arm/server_$method.log) tracebacks=$(grep -c Traceback $OUT/$arm/server_$method.log)"
done
echo GRAPHS_BARRIER_DONE
