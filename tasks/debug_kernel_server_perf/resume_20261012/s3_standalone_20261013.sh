#!/bin/bash
# S3: standard standalone workloads (the papers' benchmarks), each rate-swept on three arms, CUDA
# graphs on, 4-mode arms with the mixed-step barrier. Layout <root>/<workload>_r<point>/<ARM>_main.json
# for drivers/serving/eval_curves.py. Points: request rate (req/s) for ShareGPT-format and LongBench;
# for Mooncake (recorded timestamps) the point is the speed-up 1/slowdown. Served at YaRN x4, 131072.
set -uo pipefail
W=/dev/shm/w
until grep -q A3B_AB_DONE $W/a3b_serving_ab_20261013.log 2>/dev/null; do
  pgrep -f '[a]3b_serving_ab|[u]nloaded_ref_2026|[b]3_append_ceiling|[t]racelab_full_v2' >/dev/null || break; sleep 60; done
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
[ -L $W/validated_upstream ] || { echo "validated_upstream is not the symlink to up"; exit 1; }
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/port07 HF_HOME=$W/hf
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112
# Measured mixed-step barrier for the graphs-on 4-mode crash (root cause open; graphs_stepsync):
# a device sync after mixed steps only. No effect on stock MTP, which has no mixed steps.
export SGLANG_DEBUG_STEP_SYNC=mixed
unset FLASHINFER_FUSED_PLAN_DEBUG SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE DISABLE_CUDA_GRAPH
source $W/tree_env.sh
export MODEL=$W/qwen7b CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export YARN_FACTOR=4.0 CONTEXT_LEN=131072 HF_DATASETS_OFFLINE=1
S=$W/workloads/standalone
ROOT=$W/s3_standalone_20261013; mkdir -p $ROOT
cd $W/$TREE/tasks/realistic_workload_analysis
# name|STANDALONE|points|seconds-per-point (N = point * seconds, at least 100)
WORKLOADS=("sharegpt|sharegpt|4 8 16 32|120"
           "gsm8k|sharegpt:$S/gsm8k.json|8 16 32|90"
           "math500|sharegpt:$S/math500.json|8 16 32|60"
           "humaneval|sharegpt:$S/humaneval.json|8 16 32|20"
           "codefeedback|sharegpt:$S/codefeedback.json|8 16 32|90"
           "mooncake_conversation|mooncake_conversation|0.5 1 2|0"
           "mooncake_toolagent|mooncake_toolagent|0.5 1 2|0"
           "longbench_v2|longbench_v2|0.25 0.5 1|400")
for wl in "${WORKLOADS[@]}"; do
  IFS='|' read -r name kind points secs <<< "$wl"
  for pt in $points; do
    export STANDALONE=$kind STANDALONE_RATE=$pt
    if [ "$secs" = 0 ]; then
      export STANDALONE_N=600 MOONCAKE_SLOWDOWN=$(python3 -c "print(1/$pt)")
    else
      export STANDALONE_N=$(python3 -c "print(max(100, int($pt * $secs)))")
    fi
    OUT=$ROOT/${name}_r$pt; mkdir -p $OUT
    for arm in MTP_LPT MTPMIX_LPT MTPMIXFUSED; do
      [ -f $OUT/${arm}_validation.json ] && continue
      echo "START $name r$pt $arm n=$STANDALONE_N $(date +%T)"
      timeout 3600 $W/venv/bin/python -u run_mixed_workloads.py --scenario standalone_env --methods "$arm" --outdir "$OUT" 2>&1 | tail -4
      rc=${PIPESTATUS[0]}; L=$OUT/server_$arm.log
      echo "END $name r$pt $arm rc=$rc $(date +%T) tracebacks=$(grep -c Traceback $L 2>/dev/null)"
    done
  done
done
echo S3_DONE
