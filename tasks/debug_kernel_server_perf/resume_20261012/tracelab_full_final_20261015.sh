#!/bin/bash
# S2b full-length TraceLab with the final serving configuration (2026-10-15): the 8-session x 30-round
# seeded replay that fits the KV pool (seed 1014; context p10/50/90/99 38K/123K/223K/390K), YaRN x16,
# agent tenant only, session rates 0.005 and 0.02/s. Arms MTP | split static 3072 | split + EDF 50 ms
# + prefill floor 2048, all in tree up_edf_floor with guard 0 (the earlier v2/v3 stock arm ran in a
# different tree; trees differ ~30% in absolute speed, so every arm reruns here).
# Earlier note -- EDF across capacity, sweep 2 (2026-10-15): repeats of mix32 at 12/16 req/s (sweep 1 was one run
# per arm) and the same arms at mix128 (128-256K, YaRN x8) -- the step model was fitted on
# 32-64K traces only, so mix128 also tests its extrapolation. Original note -- EDF across capacity (2026-10-15). The frozen cohort is past saturation at every program
# concurrency (its ~4.7M agent prefill tokens alone take ~190 s), so concurrency cannot reach the
# regime where slack exists. This uses the S1 x S2 mixture instead -- mix32 (32-64K agent sessions
# at 0.1 sessions/s) + short chat at 8/12/16 req/s, which crossed capacity in S1 -- via
# run_mixed_workloads (mix_env). Arms: MTP | split static 3072 | EDF 50 ms + floor 2048, tree
# up_edf_floor, guard 0. Earlier note -- EDF near capacity (2026-10-15). The overloaded frozen cohort (24 programs + ShareGPT 8/s; TTFT
# 40-200 s in every arm) leaves no slack: an ITL-slack chunk budget starves prefill there
# (edf_dynchunk_20261015: 30 ms SLO -> ~300 prefill tokens/step, TTFT worse than MTP). Here: the
# same split model, EDF with a prefill floor (SGLANG_EDF_PREFILL_FLOOR, tree up_edf_floor =
# up_edf_budgetfix + edf_prefill_floor.patch), at two lighter load points where slack exists:
# 8 programs + chat 4/s and 16 programs + chat 6/s. Arms: MTP | split static 3072 |
# EDFR itl50 floor 2048, all in up_edf_floor, guard 0. Original note -- EDF dynamic chunk (user, 2026-10-15): instead of a static prefill chunk, each step serves the
# decode/verify rows first and admits only the prefill the step-time model says fits inside the
# tightest row's slack (ITL_SLO_MS x acceptance - elapsed), deadline-ordered. Built on the Codex
# kernel worker's EDF runtime (up_edf_budgetfix: mode-aware piecewise model, corrected budget units),
# with the decode-population guard OFF so long prefills reach FA3, on the frozen 32-64K cohort
# (48 agent programs/96 rounds, 24 active; natural ShareGPT 8/s) used by chunk_itl_sweep*.
#  T  step traces, guard 0: MTPMIX_SPLIT and MTPMIX_LPT at chunk 8192 and 2048
#  F  fit_step_matrix_piecewise (piecewise-relative) on those four traces only
#  E  MTP | SPLIT chunk 3072 (static control) | SPLIT_EDFR itl50 | SPLIT_EDFR itl30 | repeats
# Claude, 2026-10-15.
set -uo pipefail
W=/dev/shm/w
root=$W/tracelab_full_final_20261015
model_root=$W/edf_dynchunk_20261015
until grep -q 'EDF_RATE_EXIT' $W/edf_tier_20261015.log 2>/dev/null; do sleep 30; done
exec 9>"$W/a2_kernel.lock"; flock 9
trap 'printf "EDF_RATE_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
if pgrep -f '[s]glang.launch_server|[k]ernel_concurrency_driver|[f]used_reference|[r]un_mixed' >/dev/null; then echo "GPU predecessor"; exit 1; fi
test ! -e "$root" && mkdir -p "$root"
cd "$W"
export PYTHONPATH=$W/up_edf_floor/python HF_HOME=$W/hf MODEL=$W/qwen7b
export SGLANG_FUSED_FI_ROOT=$W/port07
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112 SGLANG_SPLIT_OVERLAP=0
export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3 SGLANG_FUSED_DECODE_GUARD=0
export SGLANG_EDF_MAX_PROFILED_PREFILL_PREFIX=262144 SGLANG_EDF_LONG_PREFIX_PROJECTION_THRESHOLD=262145
export TTFT_SLO_MS=2000 SGLANG_EDF_ACCEPT_LOG_EVERY=20000
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export TRACE_PATH=$W/workloads/tracelab/syfi_full_min.jsonl.gz
export AGENT_N=96 AGENT_GROUPS=48 AGENT_SEED=20261014 OTHER_N=2048 OTHER_RATE=8
export AGENT_MIN_CTX=32768 AGENT_MAX_CTX=65536 CONTEXT_LEN=131072 YARN_FACTOR=4.0
export BENCH_TRACE_MAX_TOTAL_TOKENS=131072 BENCH_SESSION_CONCURRENCY=24
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH BENCH_CHUNK SGLANG_FUSED_NATIVE_PDL SGLANG_FUSED_KERNEL_EXEC
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE SGLANG_FUSED_VERIFY_SNAPSHOT SGLANG_FUSED_DISPATCH
unset FLASHINFER_FUSED_PLAN_DEBUG SGLANG_STEP_TRACE SGLANG_EDF_SCHEDULING SGLANG_EDF_LATENCY_MODEL SGLANG_EDF_ACCEPT_MODE
unset FUSED_NOSPLIT FUSED_FA3_UNPACKED FUSED_FA3_CLUSTER_PAIR FUSED_PAGED_TMA_BUILD SERVER_EXTRA_ARGS STREAMS
sha256sum "$PYTHONPATH/sglang/srt/managers/scheduler.py" "$PYTHONPATH/sglang/srt/managers/edf_scheduler.py" \
  drivers/kernel_concurrency_driver.py drivers/serving/fit_step_matrix_piecewise_20261014.py > "$root/sources.sha256"
run() {  # label method chunk outdir [env...]
  local label=$1 method=$2 chunk=$3 out=$4; shift 4
  [ -s $W/codex_needs_gpu ] && { echo "EDF_RATE_YIELD $label"; exit 0; }
  mkdir -p "$out"
  echo "EDF_RATE_START $label method=$method chunk=${chunk:-8192} $(date -Is)"
  if [ -n "$chunk" ]; then export BENCH_CHUNK=$chunk; else unset BENCH_CHUNK; fi
  env "$@" timeout --kill-after=30 5400 "$W/venv/bin/python" -u drivers/kernel_concurrency_driver.py \
    --scenario kernel_concurrency_env --methods "$method" --outdir "$out" > "$out/driver.log" 2>&1
  local rc=$?; local L=$out/server_$method.log
  echo "EDF_RATE_END $label rc=$rc $(date -Is) tracebacks=$(grep -c Traceback $L) fallback=$(grep -ci fallback $L) split=[$(grep -o 'FUSED_SPLIT_CALLS [0-9]*' $L | tail -1)] edf_chunk=[$(grep -o 'EDF_RUNTIME_CHUNK calls=[0-9]* requested=[0-9]* admitted=[0-9]* prefill=[0-9]*' $L | tail -1)]"
  return $rc
}
export MTP_STEP_MODEL=$model_root/models/MTPMIX_SPLIT_step_model.json
test -s "$MTP_STEP_MODEL"
export TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace_all.jsonl.gz
export CODEX_RATE=0.1
runner=$W/up_edf_floor/tasks/realistic_workload_analysis/run_mixed_workloads.py
test -f "$runner" || runner=$W/up_racefix/tasks/realistic_workload_analysis/run_mixed_workloads.py
runmix() {  # label method chunk outdir [env...]
  local label=$1 method=$2 chunk=$3 out=$4; shift 4
  [ -s $W/codex_needs_gpu ] && { echo "EDF_RATE_YIELD $label"; exit 0; }
  mkdir -p "$out"
  if [ -n "$chunk" ]; then export BENCH_CHUNK=$chunk; else unset BENCH_CHUNK; fi
  echo "EDF_RATE_START $label method=$method chunk=${chunk:-8192} $(date -Is)"
  ( cd "$(dirname "$runner")" && env "$@" timeout --kill-after=30 3600 "$W/venv/bin/python" -u "$runner" \
      --scenario mix_env --methods "$method" --outdir "$out" > "$out/driver.log" 2>&1 )
  local rc=$? L=$out/server_$method.log
  echo "EDF_RATE_END $label rc=$rc $(date -Is) tracebacks=$(grep -c Traceback $L) fallback=$(grep -ci fallback $L) split=[$(grep -o 'FUSED_SPLIT_CALLS [0-9]*' $L | tail -1)] edf_chunk=[$(grep -o 'EDF_RUNTIME_CHUNK calls=[0-9]* requested=[0-9]* admitted=[0-9]* prefill=[0-9]*' $L | tail -1)]"
}
export TRACE_PATH=$W/workloads/tracelab/syfi_full_min.jsonl.gz
export YARN_FACTOR=16.0 CONTEXT_LEN=524288 AGENT_MAX_CTX=1040000 AGENT_N=240 AGENT_GROUPS=8 AGENT_SEED=1014 CHAT_RATE=0
for rate in 0.005 0.02; do
  export AGENT_RATE=$rate
  for v in "MTP|MTP_LPT||0" "SPLIT_C3072|MTPMIX_SPLIT|3072|0" "EDFR50_F2048|MTPMIX_SPLIT_EDFR||2048"; do
    IFS='|' read -r label method chunk floor <<< "$v"
    out=$root/full_r$rate/tmp_$label
    [ -s $W/codex_needs_gpu ] && { echo "EDF_RATE_YIELD $label"; exit 0; }
    mkdir -p "$out"
    if [ -n "$chunk" ]; then export BENCH_CHUNK=$chunk; else unset BENCH_CHUNK; fi
    echo "EDF_RATE_START full_r${rate}_${label} method=$method $(date -Is)"
    ( cd "$(dirname "$runner")" && env ITL_SLO_MS=50 SGLANG_EDF_PREFILL_FLOOR=$floor timeout --kill-after=30 7200 \
        "$W/venv/bin/python" -u "$runner" --scenario tracelab_full_env --methods "$method" --outdir "$out" > "$out/driver.log" 2>&1 )
    rc=$?; L=$out/server_$method.log
    echo "EDF_RATE_END full_r${rate}_${label} rc=$rc $(date -Is) tracebacks=$(grep -c Traceback $L) fallback=$(grep -ci fallback $L) split=[$(grep -o 'FUSED_SPLIT_CALLS [0-9]*' $L | tail -1)] edf_chunk=[$(grep -o 'EDF_RUNTIME_CHUNK calls=[0-9]* requested=[0-9]* admitted=[0-9]* prefill=[0-9]*' $L | tail -1)] replayed=[$(grep -ho 'replayed context[^;]*' $out/bench_${method}_agent.log | head -1)]"
    for f in "$out"/${method}_*.json; do case $f in *validation*) ;; *) mv "$f" "$root/full_r$rate/${label}_${f##*/${method}_}";; esac; done
  done
done
echo "EDF_RATE_DONE $(date -Is)"
