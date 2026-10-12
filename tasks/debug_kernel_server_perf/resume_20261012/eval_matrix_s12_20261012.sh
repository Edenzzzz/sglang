#!/bin/bash
# S1 x S2: request-rate sweep over three context-range mixtures (short chat + TraceLab agent),
# three arms, CUDA graphs on. Output layout <root>/<mixture>_r<chat rate>/<ARM>_<stream>.json is
# what drivers/serving/eval_curves.py turns into the throughput-vs-tok/s/user Pareto curve and
# SLO-attainment-vs-rate curves. The codex session rate scales with the chat rate (0.1/s at 24).
# Waits for the A3 v2 map and refuses to run unless the graphs-on output gate passed.
set -uo pipefail
W=/dev/shm/w
grep -q "G_GATE_DONE status=0" $W/graphs_on_v2_20261012.log || { echo "graphs-on gate did not pass -- not running"; exit 1; }
exec 9>"$W/a2_kernel.lock"; flock 9
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
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace_all.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json CHAT_SECONDS=150
[ -s $TRACE_PATH ] || { echo "missing $TRACE_PATH"; exit 1; }
ROOT=$W/eval_s12_20261012; mkdir -p $ROOT
cd $W/validated_upstream/tasks/realistic_workload_analysis
# mixture: min_ctx max_ctx codex_n codex_groups yarn context_len
MIXES=("mix32 32768 65536 36 8 4.0 131072" "mix64 65536 131072 24 8 4.0 131072" "mix128 131072 262144 16 4 8.0 262144")
for mix in "${MIXES[@]}"; do
  set -- $mix; name=$1
  export CODEX_MIN_CTX=$2 CODEX_MAX_CTX=$3 CODEX_N=$4 CODEX_GROUPS=$5 YARN_FACTOR=$6 CONTEXT_LEN=$7
  for rate in 4 8 12 16 20 24; do
    export CHAT_RATE=$rate CODEX_RATE=$(python3 -c "print(round($rate*0.1/24, 4))")
    OUT=$ROOT/${name}_r$rate; mkdir -p $OUT
    for arm in MTP_LPT MTPMIX_LPT MTPMIXFUSED; do
      [ -f $OUT/${arm}_validation.json ] && continue
      echo "START $name r$rate $arm codex_rate=$CODEX_RATE $(date +%T)"
      $W/venv/bin/python -u run_mixed_workloads_matrix.py --scenario mix_env --methods "$arm" --outdir "$OUT" 2>&1 | tail -6
      echo "END $name r$rate $arm rc=$? $(date +%T)"
    done
  done
done
echo EVAL_S12_DONE
