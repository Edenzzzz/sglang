#!/bin/bash
# A3 fitted dispatch, clean A/B (2026-10-15). The 2026-10-13 A3b serving test compared A3B against the
# DEFAULT rule with the decode-population guard ON, and A3B replaces the whole rule including that
# guard, so its gains (mix32 TTFT -25..-28%) could not be attributed to the fitted thresholds. Since
# then guard 0 is the serving default. Here both rules run with guard 0 on the split (static chunk
# 3072, no EDF, so only routing differs): q4 output gate per rule vs the MTP_LPT_Q4G noise floor,
# then interleaved DEFAULT/A3/DEFAULT/A3 on the frozen 32-64K cohort at C=16 + ShareGPT 8 req/s and
# one DEFAULT/A3 pair on full-length TraceLab at C=8. Program concurrency only. Queued behind
# edf_reserve_uncached_20261015.sh on a2_kernel.lock. Claude, 2026-10-15.
W=/dev/shm/w
root=$W/a3_guard0_20261015
until grep -q 'RES_EXIT' $W/edf_reserve_uncached_20261015.log 2>/dev/null; do sleep 60; done
exec 9>"$W/a2_kernel.lock"; flock 9
trap 'printf "A3G_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
if pgrep -f '[s]glang.launch_server|[k]ernel_concurrency_driver|[f]used_reference|[r]un_mixed' >/dev/null; then echo "GPU predecessor"; exit 1; fi
mkdir -p "$root"
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
export MTP_STEP_MODEL=$W/edf_dynchunk_20261015/models/MTPMIX_SPLIT_step_model.json
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH BENCH_CHUNK SGLANG_FUSED_NATIVE_PDL SGLANG_FUSED_KERNEL_EXEC
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE SGLANG_FUSED_VERIFY_SNAPSHOT SGLANG_FUSED_DISPATCH
unset FLASHINFER_FUSED_PLAN_DEBUG SGLANG_STEP_TRACE SGLANG_EDF_SCHEDULING SGLANG_EDF_LATENCY_MODEL SGLANG_EDF_ACCEPT_MODE
unset FUSED_NOSPLIT FUSED_FA3_UNPACKED FUSED_FA3_CLUSTER_PAIR FUSED_PAGED_TMA_BUILD SERVER_EXTRA_ARGS STREAMS
unset AGENT_RATE CODEX_RATE BENCH_MAX_CONCURRENCY
test -s "$MTP_STEP_MODEL"
grep -q BENCH_SESSION_CONCURRENCY $PYTHONPATH/sglang/benchmark/session_replay.py
test -f $PYTHONPATH/sglang/benchmark/session_concurrency.py
runner=$W/up_racefix/tasks/realistic_workload_analysis/run_mixed_workloads.py
sha256sum "$PYTHONPATH/sglang/srt/managers/scheduler.py" "$PYTHONPATH/sglang/srt/managers/edf_scheduler.py" \
  "$PYTHONPATH/sglang/benchmark/session_replay.py" "$PYTHONPATH/sglang/benchmark/session_concurrency.py" \
  "$runner" drivers/kernel_concurrency_driver.py "$MTP_STEP_MODEL" > "$root/sources.sha256"
ARMS=("MTP|MTP_LPT||0" "SPLIT_C3072|MTPMIX_SPLIT|3072|0" "EDFR50_F2048|MTPMIX_SPLIT_EDFR||2048")
end_line() {  # tag out method
  local L=$2/server_$3.log
  echo "A3G_END $1 $(date -Is) tracebacks=$(grep -c Traceback $L) fallback=$(grep -ci fallback $L) split=[$(grep -o 'FUSED_SPLIT_CALLS [0-9]*' $L | tail -1)] edf_chunk=[$(grep -o 'EDF_RUNTIME_CHUNK calls=[0-9]* requested=[0-9]* admitted=[0-9]* prefill=[0-9]*' $L | tail -1)] programs=[$(grep -ho '"concurrency_limit": [0-9]*, "sessions": [0-9]*, "rounds": [0-9]*, "peak_active_programs": [0-9]*, "mean_active_programs": [0-9.]*' $2/bench_$3_agent.log | head -1)] inflight=[$(grep -h '^Concurrency:' $2/bench_$3_agent.log | awk '{print $2}')]"
}

GATE=$root/gate; mkdir -p $GATE
cp $W/q4_gate_graphs_20261012/MTP_LPT_Q4G.json $GATE/
DRIVER=$W/up_racefix/tasks/fused_specdec_workload/drivers/srv_correctness.py
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
for rule in DEFAULT A3B; do
  routing=""; [ $rule = A3B ] && routing=a3
  arm=MTPMIX_SPLIT_${rule}_Q4G
  [ -s $W/codex_needs_gpu ] && { echo "A3G_YIELD gate"; exit 0; }
  echo "A3G_GATE_START $arm $(date -Is)"
  ( cd $W/up_racefix/tasks/fused_specdec_workload && CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET="$CHAT_DATASET" BENCH_CHUNK=512 \
    CORRECTNESS_DECODE_TOKENS=1200 SGLANG_FUSED_DISPATCH=$routing \
    timeout --kill-after=30 2400 "$W/venv/bin/python" -u "$DRIVER" "$arm" "$GATE" > "$GATE/$arm.run.log" 2>&1 )
  "$W/venv/bin/python" -u "$DRIVER" --compare "$GATE" MTP_LPT_Q4G "$arm" > "$GATE/$arm.compare.log" 2>&1
  echo "A3G_GATE $arm rc=$? $(tail -1 $GATE/$arm.compare.log)"
done
run_split() {  # tag routing scenario C
  local tag=$1 routing=$2 out=$root/$1
  [ -s $W/codex_needs_gpu ] && { echo "A3G_YIELD $tag"; exit 0; }
  [ -s "$out/MTPMIX_SPLIT_agent.json" ] && return
  mkdir -p "$out"; export BENCH_CHUNK=3072
  echo "A3G_START $tag routing=${routing:-default} C=$4 $(date -Is)"
  if [ "$3" = tracelab ]; then
    ( cd "$(dirname "$runner")" && env SGLANG_FUSED_DISPATCH=$routing BENCH_SESSION_CONCURRENCY=$4 \
        BENCH_SESSION_AUDIT_FILE=$out/MTPMIX_SPLIT_agent_programs.json \
        timeout --kill-after=30 7200 "$W/venv/bin/python" -u "$runner" --scenario tracelab_full_env \
        --methods MTPMIX_SPLIT --outdir "$out" > "$out/driver.log" 2>&1 )
  else
    env SGLANG_FUSED_DISPATCH=$routing BENCH_SESSION_CONCURRENCY=$4 \
      timeout --kill-after=30 5400 "$W/venv/bin/python" -u drivers/kernel_concurrency_driver.py \
      --scenario kernel_concurrency_env --methods MTPMIX_SPLIT --outdir "$out" > "$out/driver.log" 2>&1
  fi
  end_line "$tag rc=$?" "$out" MTPMIX_SPLIT
}
export TRACE_PATH=$W/workloads/tracelab/syfi_full_min.jsonl.gz
export AGENT_N=96 AGENT_GROUPS=48 AGENT_SEED=20261014 OTHER_N=2048 OTHER_RATE=8
export AGENT_MIN_CTX=32768 AGENT_MAX_CTX=65536 CONTEXT_LEN=131072 YARN_FACTOR=4.0
export BENCH_TRACE_MAX_TOTAL_TOKENS=131072
run_split cohort_c16_DEFAULT_1 "" cohort 16
run_split cohort_c16_A3_1 a3 cohort 16
run_split cohort_c16_DEFAULT_2 "" cohort 16
run_split cohort_c16_A3_2 a3 cohort 16
export YARN_FACTOR=16.0 CONTEXT_LEN=524288 AGENT_MAX_CTX=1040000 AGENT_N=240 AGENT_GROUPS=8 AGENT_SEED=1014
export CHAT_RATE=0 AGENT_RATE=inf AGENT_MIN_CTX=0
run_split tracelab_c8_DEFAULT "" tracelab 8
run_split tracelab_c8_A3 a3 tracelab 8
echo "A3G_DONE $(date -Is)"
