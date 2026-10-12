#!/bin/bash
# Kernel decision (2026-10-13): at serving verify shapes the co-launched persistent FA2 executor
# reaches 1.8-2.5 TB/s (38-52% of HBM) while the stock non-persistent BatchPrefill FA2 reaches
# 3.0-4.0 TB/s (63-83%); the two-launch split (FA3 rows + stock FA2 BatchPrefill on the rest) beats
# the fused co-launch on every fixture incl. chunk+verify (1.53 vs 2.57 ms). Serving's MTPMIX_SPLIT
# arm is that split (flash_attn_with_kvcache FA3 + BatchPrefillWithPagedKVCacheWrapper fa2).
# (1) output gate for MTPMIX_SPLIT, (2) MTPMIX_SPLIT vs MTPMIXFUSED, mix32 / mix128 x chat 16 / 24/s,
# race-fixed tree, no barrier. Fallback / split counters reported per arm.
set -uo pipefail
W=/dev/shm/w
[ "${INHERIT_LOCK:-0}" = 1 ] || { exec 9>"$W/a2_kernel.lock"; flock 9; }
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/port07 HF_HOME=$W/hf
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112
# Measured mixed-step barrier for the graphs-on 4-mode crash (root cause open; graphs_stepsync):
# a device sync after mixed steps only. No effect on stock MTP, which has no mixed steps.
export SGLANG_DEBUG_STEP_SYNC=mixed
unset FLASHINFER_FUSED_PLAN_DEBUG SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE DISABLE_CUDA_GRAPH
unset SGLANG_DEBUG_STEP_SYNC
source $W/tree_env.sh
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace_all.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
ROOT=$W/split_vs_fused_20261013; mkdir -p $ROOT
GATE=$ROOT/q4_gate; mkdir -p $GATE
cp $W/q4_gate_graphs_20261012/MTP_LPT_Q4G.json $GATE/
( cd $W/$TREE && export CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET=$CHAT_DATASET
  export BENCH_CHUNK=512 CORRECTNESS_DECODE_TOKENS=1200 SGLANG_FUSED_DECODE_GUARD=0 MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
  D=$W/$TREE/tasks/fused_specdec_workload/drivers/srv_correctness.py
  $W/venv/bin/python -u $D MTPMIX_SPLIT_Q4G $GATE 2>&1 | tail -2
  $W/venv/bin/python -u $D --compare $GATE MTP_LPT_Q4G MTPMIX_SPLIT_Q4G 2>&1 | tail -3; exit ${PIPESTATUS[0]} )
gate=$?; echo "SPLIT_GATE_DONE status=$gate"; [ $gate = 0 ] || exit 1
cd $W/$TREE/tasks/realistic_workload_analysis
for mix in "mix32 32768 65536 36 8 4.0 131072" "mix128 131072 262144 16 4 8.0 262144"; do
  set -- $mix; name=$1
  export CODEX_MIN_CTX=$2 CODEX_MAX_CTX=$3 CODEX_N=$4 CODEX_GROUPS=$5 YARN_FACTOR=$6 CONTEXT_LEN=$7 CHAT_SECONDS=$(( $4 * 10 ))
  for rate in 16 24; do
    export CHAT_RATE=$rate CODEX_RATE=0.1
    OUT=$ROOT/${name}_r$rate; mkdir -p $OUT
    for arm in MTPMIX_SPLIT MTPMIXFUSED; do
      [ -f $OUT/${arm}_validation.json ] && continue
      echo "START $name r$rate $arm $(date +%T)"
      timeout 2400 $W/venv/bin/python -u run_mixed_workloads.py --scenario mix_env --methods $arm --outdir $OUT 2>&1 | tail -3
      rc=${PIPESTATUS[0]}; L=$OUT/server_$arm.log
      echo "END $name r$rate $arm rc=$rc $(date +%T) tracebacks=$(grep -c Traceback $L) split=[$(grep -o 'FUSED_SPLIT_CALLS [0-9]*' $L | tail -1)] fused=[$(grep -o 'FUSED_KERNEL_CALLS [0-9]*' $L | tail -1)] onesided=[$(grep -o 'onesided so far [0-9]*' $L | tail -1)] fallback=$(grep -ci 'fallback' $L)"
    done
  done
done
$W/venv/bin/python $W/drivers/serving/eval_curves.py --root $ROOT --slo '{"chat": [2000, 50], "codex": [10000, 80]}' --csv $ROOT/split_vs_fused.csv > /dev/null 2>&1
awk -F, 'NR>1 {printf "ROW %s r%s %s %s out_tok_s=%s ttft=%s tpot=%s\n", $1,$2,$3,$4,$7,$9,$11}' $ROOT/split_vs_fused.csv
echo SPLIT_AB_DONE
