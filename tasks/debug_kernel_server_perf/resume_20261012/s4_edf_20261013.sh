#!/bin/bash
# S4: EDF step model on the fused kernel vs the unfused 4-mode schedule (goal point 3), on the
# race-fixed tree without the mixed-step barrier (a per-step sync would distort the very step
# times being modelled). Runs only if graphs_racefix_20261013 passed.
#  (A) STEPTRACE on MTPMIX_LPT and MTPMIXFUSED: mix32 at chat 8/16/24/s and mix128 at 12/s
#  (B) fit one linear step model per arm (tasks/perf_model/fit_mtp_step.py; fit on the first half,
#      MAPE on the second): is the fused step easier or harder to predict?
#  (C) EDF (per-request acceptance EWMA) with each arm's own model vs the same arm without EDF,
#      mix32 chat 16/s and 24/s; SLO = the S1 chat SLO (TTFT 2000 ms, ITL 50 ms).
set -uo pipefail
W=/dev/shm/w
until grep -q QUEUE_CHAIN_DONE $W/queue_chain_20261013.log 2>/dev/null; do
  pgrep -f '[q]ueue_chain_2026' >/dev/null || break; sleep 60; done
R=$W/graphs_racefix_20261013
ok=1
for d in fixed_MTPMIX_LPT_1 fixed_MTPMIX_LPT_2 fixed_MTPMIXFUSED_1 fixed_MTPMIXFUSED_2; do
  m=${d#fixed_}; m=${m%_*}
  [ -f $R/$d/${m}_validation.json ] && [ "$(grep -c Traceback $R/$d/server_$m.log)" = 0 ] || ok=0
done
grep -q "RACEFIX_GATE_DONE status=0" $W/graphs_racefix_20261013.log || ok=0
[ $ok = 1 ] || { echo "racefix did not pass -- S4 not run"; exit 1; }
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
T=up_racefix
export PYTHONPATH=$W/$T/python SGLANG_FUSED_FI_ROOT=$W/port07 HF_HOME=$W/hf
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112
unset SGLANG_DEBUG_STEP_SYNC FLASHINFER_FUSED_PLAN_DEBUG SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE DISABLE_CUDA_GRAPH
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace_all.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json CODEX_RATE=0.1
ROOT=$W/s4_edf_20261013; mkdir -p $ROOT
cd $W/$T/tasks/realistic_workload_analysis
mix() {  # name min max n groups yarn ctx
  export CODEX_MIN_CTX=$2 CODEX_MAX_CTX=$3 CODEX_N=$4 CODEX_GROUPS=$5 YARN_FACTOR=$6 CONTEXT_LEN=$7
  export CHAT_SECONDS=$(( $4 * 10 ))
}
run() {  # label method [env...]
  local label=$1 method=$2; shift 2
  local OUT=$ROOT/$label; mkdir -p $OUT
  [ -f $OUT/${method}_validation.json ] && { echo "SKIP $label"; return; }
  echo "START $label $(date +%T)"
  env "$@" timeout 2400 $W/venv/bin/python -u run_mixed_workloads.py --scenario mix_env --methods $method --outdir $OUT 2>&1 | tail -3
  local rc=${PIPESTATUS[0]} L=$OUT/server_$method.log
  echo "END $label rc=$rc $(date +%T) tracebacks=$(grep -c Traceback $L) steptraces=$(grep -c 'STEPTRACE.*gpu=' $L)"
}
# (A) traces
for arm in MTPMIX_LPT MTPMIXFUSED; do
  mix mix32 32768 65536 36 8 4.0 131072
  for r in 8 16 24; do run trace_${arm}_mix32_r$r $arm CHAT_RATE=$r SGLANG_STEP_TRACE=1; done
  mix mix128 131072 262144 16 4 8.0 262144
  run trace_${arm}_mix128_r12 $arm CHAT_RATE=12 SGLANG_STEP_TRACE=1
done
# (B) fits
for arm in MTPMIX_LPT MTPMIXFUSED; do
  cat $ROOT/trace_${arm}_mix32_r8/server_$arm.log $ROOT/trace_${arm}_mix32_r16/server_$arm.log \
      $ROOT/trace_${arm}_mix128_r12/server_$arm.log $ROOT/trace_${arm}_mix32_r24/server_$arm.log > $ROOT/steptrace_$arm.log
  $W/venv/bin/python $W/$T/tasks/perf_model/fit_mtp_step.py $ROOT/steptrace_$arm.log --out $ROOT/step_model_$arm.json 2>&1 | tail -4
  echo "FIT $arm $(python3 -c "import json;d=json.load(open('$ROOT/step_model_$arm.json'));print('n',d['n_steps'],'heldout_mape',round(d['heldout_mape'],4),'const',round(d['heldout_mape_constant'],4))")"
done
# (C) EDF vs none, own model
mix mix32 32768 65536 36 8 4.0 131072
for r in 16 24; do
  for arm in MTPMIX_LPT MTPMIXFUSED; do
    run edf_${arm}_r$r ${arm}_EDFR CHAT_RATE=$r MTP_STEP_MODEL=$ROOT/step_model_$arm.json TTFT_SLO_MS=2000 ITL_SLO_MS=50
    run none_${arm}_r$r $arm CHAT_RATE=$r
  done
done
echo S4_DONE
