#!/bin/bash
# 2026-10-15 18:1x: rerun with up_edf_reserve2 (edf_reserve_matched.patch). The first run of this
# reservation fired ~never: the FCFS/priority policy never prefix-matches waiting requests on this
# radix cache (supports_fast_match_prefix False), so num_matched_prefix_tokens was 0 and a 99%-cached
# 27K round was charged 27K tokens > the 2K budget -> loop break. The patch matches the first 16
# waiting requests before scoring them. C=4 run twice with a same-tree control between (single C=4
# runs of identical code differed 3.9 vs 4.6 s mean TTFT). Queued behind a3_guard0_20261015.sh.
# EDF chunk reservation A/B (2026-10-15). The concurrency sweep showed head-of-line blocking at C=4:
# cache-hit rounds (<1K uncached tokens) keep a ~150-200 ms median TTFT but their mean rises to
# 1.7-3.1 s in the split arms, because an in-progress chunked prefill takes the whole 2-3K chunk
# budget every step and new requests only get the remainder -- EDF orders the waiting queue but never
# ranks it against the chunked request. SGLANG_EDF_RESERVE_EARLIER=1 (edf_reserve_earlier.patch,
# tree up_edf_reserve = up_edf_floor + patch) reserves budget for waiting requests whose deadline,
# rescored on uncached work, is earlier than the chunked request's. Program concurrency only.
# Full-length TraceLab (8 sessions x 30 rounds, seed 1014, YaRN x16): C=4 EDF control (rerun, same
# tree, reserve off) | EDF+reserve; C=8 EDF+reserve | EDF control. Then the frozen 32-64K cohort +
# ShareGPT 8 req/s at C=16: EDF+reserve only. Controls (EDF, reserve off, same up_edf_floor code)
# come from the parallel session's edf_reserve_ab_20261015.sh run in the same window (16:00-17:30).
# Requeued 16:4x behind that A/B; serializes with its queue on a2_kernel.lock. Claude, 2026-10-15.
set -uo pipefail
W=/dev/shm/w
root=$W/edf_reserve_matched_20261015
until grep -q 'A3G_EXIT' $W/a3_guard0_20261015.log 2>/dev/null; do sleep 60; done
exec 9>"$W/a2_kernel.lock"; flock 9
trap 'printf "RES_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
if pgrep -f '[s]glang.launch_server|[k]ernel_concurrency_driver|[f]used_reference|[r]un_mixed' >/dev/null; then echo "GPU predecessor"; exit 1; fi
mkdir -p "$root"
cd "$W"
export PYTHONPATH=$W/up_edf_reserve2/python HF_HOME=$W/hf MODEL=$W/qwen7b
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
  echo "RES_END $1 $(date -Is) tracebacks=$(grep -c Traceback $L) fallback=$(grep -ci fallback $L) split=[$(grep -o 'FUSED_SPLIT_CALLS [0-9]*' $L | tail -1)] edf_chunk=[$(grep -o 'EDF_RUNTIME_CHUNK calls=[0-9]* requested=[0-9]* admitted=[0-9]* prefill=[0-9]*' $L | tail -1)] programs=[$(grep -ho '"concurrency_limit": [0-9]*, "sessions": [0-9]*, "rounds": [0-9]*, "peak_active_programs": [0-9]*, "mean_active_programs": [0-9.]*' $2/bench_$3_agent.log | head -1)] inflight=[$(grep -h '^Concurrency:' $2/bench_$3_agent.log | awk '{print $2}')]"
}

export TRACE_PATH=$W/workloads/tracelab/syfi_full_min.jsonl.gz
run_arm() {  # tag C reserve scenario driver
  local tag=$1 C=$2 res=$3 out=$root/$1
  [ -s $W/codex_needs_gpu ] && { echo "RES_YIELD $tag"; exit 0; }
  [ -s "$out/MTPMIX_SPLIT_EDFR_agent.json" ] && return
  mkdir -p "$out"; unset BENCH_CHUNK
  echo "RES_START $tag C=$C reserve=$res $(date -Is)"
  if [ "$4" = tracelab ]; then
    ( cd "$(dirname "$runner")" && env ITL_SLO_MS=50 SGLANG_EDF_PREFILL_FLOOR=2048 SGLANG_EDF_RESERVE_EARLIER=$res \
        BENCH_SESSION_CONCURRENCY=$C BENCH_SESSION_AUDIT_FILE=$out/MTPMIX_SPLIT_EDFR_agent_programs.json \
        timeout --kill-after=30 7200 "$W/venv/bin/python" -u "$runner" --scenario tracelab_full_env \
        --methods MTPMIX_SPLIT_EDFR --outdir "$out" > "$out/driver.log" 2>&1 )
  else
    env ITL_SLO_MS=50 SGLANG_EDF_PREFILL_FLOOR=2048 SGLANG_EDF_RESERVE_EARLIER=$res BENCH_SESSION_CONCURRENCY=$C \
      timeout --kill-after=30 5400 "$W/venv/bin/python" -u drivers/kernel_concurrency_driver.py \
      --scenario kernel_concurrency_env --methods MTPMIX_SPLIT_EDFR --outdir "$out" > "$out/driver.log" 2>&1
  fi
  local rc=$?
  end_line "$tag rc=$rc reserve=[$(grep -o 'EDF_RESERVE calls=[0-9]* budget=[0-9]* reserved=[0-9]*' $out/server_MTPMIX_SPLIT_EDFR.log | tail -1)]" "$out" MTPMIX_SPLIT_EDFR
}
export YARN_FACTOR=16.0 CONTEXT_LEN=524288 AGENT_MAX_CTX=1040000 AGENT_N=240 AGENT_GROUPS=8 AGENT_SEED=1014
export CHAT_RATE=0 AGENT_RATE=inf
run_arm tracelab_c4_EDF_RES_1 4 1 tracelab
run_arm tracelab_c4_EDF_CTRL 4 0 tracelab
run_arm tracelab_c4_EDF_RES_2 4 1 tracelab
run_arm tracelab_c8_EDF_RES 8 1 tracelab
export AGENT_N=96 AGENT_GROUPS=48 AGENT_SEED=20261014 OTHER_N=2048 OTHER_RATE=8
export AGENT_MIN_CTX=32768 AGENT_MAX_CTX=65536 CONTEXT_LEN=131072 YARN_FACTOR=4.0
export BENCH_TRACE_MAX_TOTAL_TOKENS=131072
unset AGENT_RATE CHAT_RATE
run_arm cohort_c16_EDF_RES 16 1 cohort
echo "RES_DONE $(date -Is)"
