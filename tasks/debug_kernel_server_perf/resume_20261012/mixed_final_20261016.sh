#!/bin/bash
# EDF over the chunked prefill: chunk preemption A/B (2026-10-16). Root cause of the agent TTFT miss
# (edf_reserve_* runs, server-log trace): a cache-hit round (105,672 tokens, 2,377 uncached) sat in
# the waiting queue for 61 s while a cold ~390K prompt was prefilled 2,048 tokens per step; sglang
# keeps one chunked request and gives it the whole budget, and 2,377 > budget so no reservation can
# admit it. SGLANG_EDF_PREEMPT_CHUNK=1 (edf_preempt_chunk.patch, tree up_edf_preempt): rescore waiting
# deadlines on uncached work and, if the earliest is due before the chunked request and short
# (<= 16K uncached, chunked remainder >= 2x), park the chunked request (prefix checkpointed in the
# radix tree, row released as a retraction, re-queued next step). Gate (EDF control | EDF+preempt)
# with the EDF serving gate vs the MTP_LPT_Q4G floor, then interleaved arms in ONE tree/queue:
# THIS run (2026-10-16 06:4x): the final single-GPU config -- split + EDF 50 ms + floor 2048 + uncapped
# chunk preemption (edf_preempt_chunk_v2, park gate v2 PASS) -- vs MTP on the mixed workload: frozen
# 32-64K cohort at program concurrency C=8/16 + natural ShareGPT chat at 4 (near capacity) and
# 8 req/s (overload; OTHER_N = 256 s of arrivals). Replaces the void cohort C=16 preempt point.
set -uo pipefail
W=/dev/shm/w
root=$W/mixed_final_20261016
exec 9>"$W/a2_kernel.lock"; flock 9
trap 'printf "MIXF_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
for _ in $(seq 60); do pgrep -f '[s]glang.launch_server|[k]ernel_concurrency_driver|[f]used_reference|[r]un_mixed' >/dev/null || break; sleep 5; done
if pgrep -f '[s]glang.launch_server|[k]ernel_concurrency_driver|[f]used_reference|[r]un_mixed' >/dev/null; then echo "GPU predecessor"; exit 1; fi
mkdir -p "$root"
cd "$W"
export PYTHONPATH=$W/up_edf_preempt/python HF_HOME=$W/hf MODEL=$W/qwen7b
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
  echo "MIXF_END $1 $(date -Is) tracebacks=$(grep -c Traceback $L) fallback=$(grep -ci fallback $L) split=[$(grep -o 'FUSED_SPLIT_CALLS [0-9]*' $L | tail -1)] edf_chunk=[$(grep -o 'EDF_RUNTIME_CHUNK calls=[0-9]* requested=[0-9]* admitted=[0-9]* prefill=[0-9]*' $L | tail -1)] programs=[$(grep -ho '"concurrency_limit": [0-9]*, "sessions": [0-9]*, "rounds": [0-9]*, "peak_active_programs": [0-9]*, "mean_active_programs": [0-9.]*' $2/bench_$3_agent.log | head -1)] inflight=[$(grep -h '^Concurrency:' $2/bench_$3_agent.log | awk '{print $2}')]"
}

export TRACE_PATH=$W/workloads/tracelab/syfi_full_min.jsonl.gz
run_arm() {  # tag C method preempt scenario [max_tokens]
  local tag=$1 C=$2 method=$3 pre=$4 out=$root/$1; export SGLANG_EDF_PREEMPT_MAX_TOKENS=${6:-16384}
  [ -s $W/codex_needs_gpu ] && { echo "MIXF_YIELD $tag"; exit 0; }
  [ -s "$out/${method}_agent.json" ] && return
  mkdir -p "$out"; unset BENCH_CHUNK
  echo "MIXF_START $tag C=$C method=$method preempt=$pre $(date -Is)"
  if [ "$5" = tracelab ]; then
    ( cd "$(dirname "$runner")" && env ITL_SLO_MS=50 SGLANG_EDF_PREFILL_FLOOR=2048 SGLANG_EDF_PREEMPT_CHUNK=$pre SGLANG_EDF_RESERVE_EARLIER=$pre \
        BENCH_SESSION_CONCURRENCY=$C BENCH_SESSION_AUDIT_FILE=$out/${method}_agent_programs.json \
        timeout --kill-after=30 7200 "$W/venv/bin/python" -u "$runner" --scenario tracelab_full_env \
        --methods "$method" --outdir "$out" > "$out/driver.log" 2>&1 )
  else
    env ITL_SLO_MS=50 SGLANG_EDF_PREFILL_FLOOR=2048 SGLANG_EDF_PREEMPT_CHUNK=$pre SGLANG_EDF_RESERVE_EARLIER=$pre BENCH_SESSION_CONCURRENCY=$C \
      timeout --kill-after=30 5400 "$W/venv/bin/python" -u drivers/kernel_concurrency_driver.py \
      --scenario kernel_concurrency_env --methods "$method" --outdir "$out" > "$out/driver.log" 2>&1
  fi
  local rc=$?
  end_line "$tag rc=$rc park=[$(grep -o 'EDF_PARK calls=[0-9]*[^\n]*' $out/server_$method.log | tail -1)]" "$out" "$method"
}
export TRACE_PATH=$W/workloads/tracelab/syfi_full_min.jsonl.gz
export AGENT_N=96 AGENT_GROUPS=48 AGENT_SEED=20261014
export AGENT_MIN_CTX=32768 AGENT_MAX_CTX=65536 CONTEXT_LEN=131072 YARN_FACTOR=4.0
export BENCH_TRACE_MAX_TOTAL_TOKENS=131072
unset AGENT_RATE CHAT_RATE
for rate in 4 8; do
  export OTHER_RATE=$rate OTHER_N=$((rate * 256))
  for C in 8 16; do
    run_arm cohort_c${C}_chat${rate}_MTP $C MTP_LPT 0 cohort
    run_arm cohort_c${C}_chat${rate}_EDFP $C MTPMIX_SPLIT_EDFR 1 cohort 1000000000
  done
done
echo "MIXF_DONE $(date -Is)"
