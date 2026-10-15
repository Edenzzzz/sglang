#!/bin/bash
# Agent load by program concurrency only (ThunderAgent protocol; user, 2026-10-15): C programs are
# kept active, the next session starts when one finishes (closed loop, rounds back to back,
# BENCH_SESSION_CONCURRENCY via sglang.benchmark.session_concurrency, ported into up_edf_floor by
# edf_floor_session_concurrency.patch -- the EDF trees lacked it, so earlier EDF frozen-cohort runs
# released every program at once). Every arm must log PROGRAM_CONCURRENCY with peak == C.
#  P1 full-length TraceLab, agent alone: 8 sessions x 30 rounds, seed 1014 (context p10/50/90/99
#     38K/123K/223K/390K), YaRN x16 / 524288, C = 2 / 4 / 8 (KV pool holds the 8 sessions' peaks).
#  P2 frozen 32-64K cohort (48 programs / 96 rounds) + natural ShareGPT at 8 req/s, C = 8 / 16 / 24.
# Arms: MTP | split static chunk 3072 | split + EDF 50 ms + prefill floor 2048; tree up_edf_floor,
# guard 0, no barrier. Yields to /dev/shm/w/codex_needs_gpu before every arm. Claude, 2026-10-15.
set -uo pipefail
W=/dev/shm/w
root=$W/concurrency_sweep_20261015
exec 9>"$W/a2_kernel.lock"; flock 9
trap 'printf "CONC_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
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
  echo "CONC_END $1 $(date -Is) tracebacks=$(grep -c Traceback $L) fallback=$(grep -ci fallback $L) split=[$(grep -o 'FUSED_SPLIT_CALLS [0-9]*' $L | tail -1)] edf_chunk=[$(grep -o 'EDF_RUNTIME_CHUNK calls=[0-9]* requested=[0-9]* admitted=[0-9]* prefill=[0-9]*' $L | tail -1)] programs=[$(grep -ho '"concurrency_limit": [0-9]*, "sessions": [0-9]*, "rounds": [0-9]*, "peak_active_programs": [0-9]*, "mean_active_programs": [0-9.]*' $2/bench_$3_agent.log | head -1)] inflight=[$(grep -h '^Concurrency:' $2/bench_$3_agent.log | awk '{print $2}')]"
}

# P1: full-length TraceLab, agent alone, C = 2 / 4 / 8.
export TRACE_PATH=$W/workloads/tracelab/syfi_full_min.jsonl.gz
export YARN_FACTOR=16.0 CONTEXT_LEN=524288 AGENT_MAX_CTX=1040000 AGENT_N=240 AGENT_GROUPS=8 AGENT_SEED=1014
export CHAT_RATE=0 AGENT_RATE=inf
for C in 2 4 8; do
  for v in "${ARMS[@]}"; do
    IFS='|' read -r label method chunk floor <<< "$v"
    out=$root/tracelab_c$C/$label
    [ -s $W/codex_needs_gpu ] && { echo "CONC_YIELD tracelab_c$C $label"; exit 0; }
    [ -s "$out/${method}_agent.json" ] && continue
    mkdir -p "$out"
    if [ -n "$chunk" ]; then export BENCH_CHUNK=$chunk; else unset BENCH_CHUNK; fi
    echo "CONC_START tracelab_c$C $label method=$method $(date -Is)"
    ( cd "$(dirname "$runner")" && env ITL_SLO_MS=50 SGLANG_EDF_PREFILL_FLOOR=$floor \
        BENCH_SESSION_CONCURRENCY=$C BENCH_SESSION_AUDIT_FILE=$out/${method}_agent_programs.json \
        timeout --kill-after=30 7200 "$W/venv/bin/python" -u "$runner" --scenario tracelab_full_env \
        --methods "$method" --outdir "$out" > "$out/driver.log" 2>&1 )
    end_line "tracelab_c${C}_$label rc=$?" "$out" "$method"
  done
done

# P2: frozen 32-64K cohort + ShareGPT 8 req/s, C = 8 / 16 / 24.
export TRACE_PATH=$W/workloads/tracelab/syfi_full_min.jsonl.gz
export AGENT_N=96 AGENT_GROUPS=48 AGENT_SEED=20261014 OTHER_N=2048 OTHER_RATE=8
export AGENT_MIN_CTX=32768 AGENT_MAX_CTX=65536 CONTEXT_LEN=131072 YARN_FACTOR=4.0
export BENCH_TRACE_MAX_TOTAL_TOKENS=131072
unset AGENT_RATE CHAT_RATE
for C in 8 16 24; do
  for v in "${ARMS[@]}"; do
    IFS='|' read -r label method chunk floor <<< "$v"
    out=$root/cohort_c$C/$label
    [ -s $W/codex_needs_gpu ] && { echo "CONC_YIELD cohort_c$C $label"; exit 0; }
    [ -s "$out/${method}_agent.json" ] && continue
    mkdir -p "$out"
    if [ -n "$chunk" ]; then export BENCH_CHUNK=$chunk; else unset BENCH_CHUNK; fi
    echo "CONC_START cohort_c$C $label method=$method $(date -Is)"
    env ITL_SLO_MS=50 SGLANG_EDF_PREFILL_FLOOR=$floor BENCH_SESSION_CONCURRENCY=$C \
      timeout --kill-after=30 5400 "$W/venv/bin/python" -u drivers/kernel_concurrency_driver.py \
      --scenario kernel_concurrency_env --methods "$method" --outdir "$out" > "$out/driver.log" 2>&1
    end_line "cohort_c${C}_$label rc=$?" "$out" "$method"
  done
done
echo "CONC_DONE $(date -Is)"
