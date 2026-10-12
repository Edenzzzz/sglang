#!/bin/bash
# Baselines on the S1 x S2 layout (mixtures mix32 / mix128, chat rate sweep, TraceLab agent 0.1/s):
#  LPT    stock sglang WITHOUT speculation (mixed chunk on): the non-speculative control
#  PDMUX  sglang PD-multiplexing (--enable-pdmux: GreenContext SM split between prefill and decode,
#         layer-split prefill; upstream's implementation of MuxWise, arXiv 2504.14489). Requires
#         chunked prefill off and overlap off, has no speculative path, and warns of degradation
#         with GreenContext + CUDA graphs on torch > 2.6 (pod: 2.14) -- recorded as a caveat.
# Output <root>/<mixture>_r<rate>/{LPT,PDMUX}_<stream>.json for eval_curves.py beside the S1 arms.
set -uo pipefail
W=/dev/shm/w
until grep -q QUEUE_CHAIN_DONE $W/queue_chain2_20261013.log 2>/dev/null; do
  pgrep -f '[q]ueue_chain2_2026' >/dev/null || break; sleep 60; done
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
unset SGLANG_DEBUG_STEP_SYNC
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace_all.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
ROOT=$W/eval_baselines_20261013; mkdir -p $ROOT
cd $W/validated_upstream/tasks/realistic_workload_analysis
MIXES=("mix32 32768 65536 36 8 4.0 131072" "mix128 131072 262144 16 4 8.0 262144")
for mix in "${MIXES[@]}"; do
  set -- $mix; name=$1
  export CODEX_MIN_CTX=$2 CODEX_MAX_CTX=$3 CODEX_N=$4 CODEX_GROUPS=$5 YARN_FACTOR=$6 CONTEXT_LEN=$7
  for rate in 8 16 24; do
    export CHAT_RATE=$rate CODEX_RATE=0.1 CHAT_SECONDS=$((CODEX_N * 10))
    OUT=$ROOT/${name}_r$rate; mkdir -p $OUT
    for arm in LPT PDMUX; do
      [ -f $OUT/${arm}_validation.json ] && continue
      extra=""; [ $arm = PDMUX ] && extra="--enable-pdmux --chunked-prefill-size -1 --disable-overlap-schedule"
      TMP=$OUT/tmp_$arm; rm -rf $TMP; mkdir -p $TMP
      echo "START $name r$rate $arm $(date +%T)"
      SERVER_EXTRA_ARGS="$extra" timeout 2400 $W/venv/bin/python -u run_mixed_workloads.py --scenario mix_env --methods LPT --outdir $TMP 2>&1 | tail -3
      rc=${PIPESTATUS[0]}
      for f in $TMP/LPT_*; do [ -e "$f" ] && mv "$f" "$OUT/${arm}_${f##*/LPT_}"; done
      for f in $TMP/*_LPT*.log; do [ -e "$f" ] && mv "$f" "$OUT/$(basename $f | sed s/LPT/$arm/)"; done
      echo "END $name r$rate $arm rc=$rc $(date +%T) tracebacks=$(grep -c Traceback $OUT/server_$arm.log 2>/dev/null) pdmux=$(grep -c -i 'pdmux\|green' $OUT/server_$arm.log 2>/dev/null)"
    done
  done
done
echo BASELINES_DONE
