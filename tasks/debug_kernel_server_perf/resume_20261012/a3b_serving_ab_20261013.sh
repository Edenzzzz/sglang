#!/bin/bash
# A3b serving A/B. The A3 rule lost in serving (-3.5% throughput, +29% TTFT) because its FA3 work
# floor, fitted on band rows only (qo <= 32), also vetoed 512-token chat prefill chunks: 65% of
# mixed steps went one-sided onto the stock path vs 7% under the default rule. A3b applies the floor
# only inside the fitted regime (SGLANG_A3_WORK_MAX_QO=32). Gate first, then default vs A3b
# interleaved twice (the repeat is the noise floor), chat 24/s + TraceLab agent 0.1/s, graphs on,
# mixed-step barrier on both arms. Reports one-sided / split plan counts per arm.
set -uo pipefail
W=/dev/shm/w
until grep -q UNLOADED_REF_DONE $W/unloaded_ref_20261013.log 2>/dev/null; do
  pgrep -f '[u]nloaded_ref_2026|[b]3_append_ceiling|[t]racelab_full_v2' >/dev/null || break; sleep 60; done
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/port07 HF_HOME=$W/hf
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112 SGLANG_DEBUG_STEP_SYNC=mixed
unset FLASHINFER_FUSED_PLAN_DEBUG SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE DISABLE_CUDA_GRAPH
source $W/tree_env.sh
# (1) output gate, graphs on, barrier on, A3b rule
OUT=$W/q4_gate_a3b_20261013; mkdir -p $OUT
cp $W/q4_gate_graphs_20261012/MTP_LPT_Q4G.json $OUT/
DRIVER=$W/$TREE/tasks/fused_specdec_workload/drivers/srv_correctness.py
( export CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
  export BENCH_CHUNK=512 CORRECTNESS_DECODE_TOKENS=1200 SGLANG_FUSED_DECODE_GUARD=0 MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
  SGLANG_FUSED_DISPATCH=a3 $W/venv/bin/python -u "$DRIVER" MTPMIXFUSED_A3B_Q4G "$OUT" 2>&1 | tail -2
  $W/venv/bin/python -u "$DRIVER" --compare "$OUT" MTP_LPT_Q4G MTPMIXFUSED_A3B_Q4G 2>&1 || exit 1 )
gate=$?; echo "A3B_GATE_DONE status=$gate"; [ $gate = 0 ] || exit 1
# (2) realistic A/B, graphs on
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json CHAT_RATE=24 CODEX_RATE=0.1
OUT=$W/a3b_serving_ab_20261013; mkdir -p $OUT
cd $W/$TREE/tasks/realistic_workload_analysis
for arm in DEFAULT_1 A3B_1 DEFAULT_2 A3B_2; do
  extra=""; [ ${arm%_*} = A3B ] && extra="SGLANG_FUSED_DISPATCH=a3"
  mkdir -p $OUT/$arm
  echo "START $arm $(date +%T)"
  env $extra $W/venv/bin/python -u run_mixed_workloads.py --scenario agent_chat_rated --methods MTPMIXFUSED --outdir "$OUT/$arm" 2>&1 | tail -6
  L=$OUT/$arm/server_MTPMIXFUSED.log
  echo "END $arm rc=$? $(date +%T) graphs=$(grep -c 'cuda graph: True' $L) tracebacks=$(grep -c Traceback $L) last_plan=[$(grep -o 'FUSED_SPLIT_PLAN ok [0-9]*: fa3=[0-9]* fa2=[0-9]* (onesided so far [0-9]*)' $L | tail -1)]"
done
echo A3B_AB_DONE
