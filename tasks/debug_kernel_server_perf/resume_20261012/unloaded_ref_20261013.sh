#!/bin/bash
# Unloaded reference for relative SLOs (eval_curves.py --slo-relative --ref-root): each S1 x S2
# tenant ALONE on the stock MTP server with the same mixture config (context range, YaRN, window),
# at a rate that leaves the server idle between requests -- chat 0.5/s, agent 0.02 sessions/s.
# The swept runs' lowest rate cannot serve as reference: mix128 chat already waits 38 s there.
set -uo pipefail
W=/dev/shm/w
until grep -q B3_APPEND_DONE $W/b3_append_ceiling_20261013.log 2>/dev/null; do
  pgrep -f '[b]3_append_ceiling|[t]racelab_full_v2' >/dev/null || break; sleep 60; done
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
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
[ -s $TRACE_PATH ] || { echo "missing $TRACE_PATH"; exit 1; }
ROOT=$W/eval_s12_unloaded_20261013; mkdir -p $ROOT
cd $W/validated_upstream/tasks/realistic_workload_analysis
# mixture: min_ctx max_ctx codex_n codex_groups yarn context_len
MIXES=("mix32 32768 65536 36 8 4.0 131072" "mix64 65536 131072 24 8 4.0 131072" "mix128 131072 262144 16 4 8.0 262144")
for mix in "${MIXES[@]}"; do
  set -- $mix; name=$1
  export CODEX_MIN_CTX=$2 CODEX_MAX_CTX=$3 CODEX_N=$4 CODEX_GROUPS=$5 YARN_FACTOR=$6 CONTEXT_LEN=$7
  for stream in chat codex; do
    export STREAMS=$stream CHAT_RATE=0.5 CHAT_SECONDS=120 CODEX_RATE=0.02
    OUT=$ROOT/${name}_unloaded_$stream; mkdir -p $OUT
    [ -f $OUT/MTP_LPT_validation.json ] && continue
    echo "START $name unloaded $stream $(date +%T)"
    timeout 3600 $W/venv/bin/python -u run_mixed_workloads.py --scenario mix_env --methods MTP_LPT --outdir "$OUT" 2>&1 | tail -4
    echo "END $name unloaded $stream rc=$? $(date +%T)"
  done
done
echo UNLOADED_REF_DONE
