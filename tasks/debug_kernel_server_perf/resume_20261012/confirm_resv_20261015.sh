#!/bin/bash
# Confirmation (2026-10-15), queued after edf_floor_resv_20261015.sh: the candidate config (split +
# EDF 50 ms + floor 2048 + earlier-deadline reservation, tree up_edf_resv; and mode 2, which also
# reserves for short later-deadline requests, tree up_edf_resv2, edf_reserve_short_mode2.patch -- at
# TraceLab C=8 mode 1 fired <200 times because long prompts were already past their deadline) vs MTP and static split
# 3072, all in ONE queue and interleaved twice (rep 1, rep 2), so the verdict does not mix runs: the
# EDF control repeated 17% apart in mean TTFT across queues. TraceLab C = 4 / 8, cohort C = 16 + chat.
# Earlier header follows.
# EDF reservation + lower prefill floor (2026-10-15), queued after edf_reserve_ab_20261015.sh. The
# reservation targets agent TTFT; chat mean ITL still misses (-24% at cohort C=16) because the 2048
# floor binds every step, so this lowers the floor to 1024 and 512 with the reservation on.
# Original header of edf_reserve_ab_20261015.sh follows.
# EDF earlier-deadline reservation A/B (2026-10-15), queued after concurrency_sweep_20261015.sh.
# The sweep showed the split arms lose agent mean TTFT at C=4 only on short append rounds (<4K
# uncached: 926 ms MTP -> 1994 split / 2344 EDF) while long cold prefills improve 20-26%: sglang
# always continues the in-progress chunked prefill with the whole step budget, so a short append
# waits for every remaining chunk of a long prompt. edf_reserve_earlier_deadlines.patch (tree
# up_edf_resv, SGLANG_EDF_RESERVE=1) caps the chunked request's share so earlier-deadline waiting
# requests are admitted in the same step. Arms per point: EDF control (up_edf_floor) | EDF +
# reservation (up_edf_resv); TraceLab C = 4 / 8, cohort C = 16 + ShareGPT 8 req/s.
# Original header of concurrency_sweep_20261015.sh follows.
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
root=$W/confirm_resv_20261015
until grep -q 'FLRV_EXIT' $W/edf_floor_resv_20261015.log 2>/dev/null; do sleep 60; done
exec 9>"$W/a2_kernel.lock"; flock 9
trap 'printf "CONF_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
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
grep -q EDF_RESERVE_EARLIER_DEADLINES $W/up_edf_resv/python/sglang/srt/managers/scheduler.py
grep -q EDF_RESERVE_MODE $W/up_edf_resv2/python/sglang/srt/managers/scheduler.py
test -f $PYTHONPATH/sglang/benchmark/session_concurrency.py
runner=$W/up_racefix/tasks/realistic_workload_analysis/run_mixed_workloads.py
sha256sum "$PYTHONPATH/sglang/srt/managers/scheduler.py" "$PYTHONPATH/sglang/srt/managers/edf_scheduler.py" \
  "$PYTHONPATH/sglang/benchmark/session_replay.py" "$PYTHONPATH/sglang/benchmark/session_concurrency.py" \
  "$runner" drivers/kernel_concurrency_driver.py "$MTP_STEP_MODEL" \
  $W/up_edf_resv/python/sglang/srt/managers/scheduler.py $W/up_edf_resv2/python/sglang/srt/managers/scheduler.py > "$root/sources.sha256"
end_line() {  # tag out method
  local L=$2/server_$3.log
  echo "CONF_END $1 $(date -Is) tracebacks=$(grep -c Traceback $L) fallback=$(grep -ci fallback $L) split=[$(grep -o 'FUSED_SPLIT_CALLS [0-9]*' $L | tail -1)] edf_chunk=[$(grep -o 'EDF_RUNTIME_CHUNK calls=[0-9]* requested=[0-9]* admitted=[0-9]* prefill=[0-9]*' $L | tail -1)] reserve=[$(grep -o 'EDF_RESERVE calls=[0-9]* reserved=[0-9]* budget=[0-9]* total_reserved=[0-9]*' $L | tail -1)] programs=[$(grep -ho '"concurrency_limit": [0-9]*, "sessions": [0-9]*, "rounds": [0-9]*, "peak_active_programs": [0-9]*, "mean_active_programs": [0-9.]*' $2/bench_$3_agent.log | head -1)] inflight=[$(grep -h '^Concurrency:' $2/bench_$3_agent.log | awk '{print $2}')]"
}
ARMS=("MTP|up_edf_resv|0|0|MTP_LPT" "SPLIT_C3072|up_edf_resv|0|0|MTPMIX_SPLIT" "EDF_RESV|up_edf_resv|1|2048|MTPMIX_SPLIT_EDFR" "EDF_RESV2|up_edf_resv2|2|2048|MTPMIX_SPLIT_EDFR")
run_arm() {  # tag out scenario-driver(runner|driver)
  local tag=$1 out=$2 kind=$3 label=$4 tree=$5 resv=$6 C=$7 floor=$8 method=$9
  [ -s $W/codex_needs_gpu ] && { echo "CONF_YIELD $tag"; exit 0; }
  [ -s "$out/${method}_agent.json" ] && return 0
  mkdir -p "$out"; unset BENCH_CHUNK; [ "$label" = SPLIT_C3072 ] && export BENCH_CHUNK=3072
  echo "CONF_START $tag method=$method tree=$tree $(date -Is)"
  if [ "$kind" = runner ]; then
    ( cd "$(dirname "$runner")" && env PYTHONPATH=$W/$tree/python ITL_SLO_MS=50 SGLANG_EDF_PREFILL_FLOOR=$floor \
        SGLANG_EDF_RESERVE=$resv BENCH_SESSION_CONCURRENCY=$C BENCH_SESSION_AUDIT_FILE=$out/${method}_agent_programs.json \
        timeout --kill-after=30 7200 "$W/venv/bin/python" -u "$runner" --scenario tracelab_full_env \
        --methods "$method" --outdir "$out" > "$out/driver.log" 2>&1 )
  else
    env PYTHONPATH=$W/$tree/python ITL_SLO_MS=50 SGLANG_EDF_PREFILL_FLOOR=$floor SGLANG_EDF_RESERVE=$resv \
      BENCH_SESSION_CONCURRENCY=$C timeout --kill-after=30 5400 "$W/venv/bin/python" -u \
      drivers/kernel_concurrency_driver.py --scenario kernel_concurrency_env --methods "$method" \
      --outdir "$out" > "$out/driver.log" 2>&1
  fi
  end_line "$tag rc=$?" "$out" "$method"
}

# TraceLab full-length, agent alone, C = 4 / 8.
export TRACE_PATH=$W/workloads/tracelab/syfi_full_min.jsonl.gz
export YARN_FACTOR=16.0 CONTEXT_LEN=524288 AGENT_MAX_CTX=1040000 AGENT_N=240 AGENT_GROUPS=8 AGENT_SEED=1014
export CHAT_RATE=0 AGENT_RATE=inf
 for rep in 1 2; do for C in 4 8; do for v in "${ARMS[@]}"; do IFS='|' read -r label tree resv floor method <<< "$v"
  run_arm "tracelab_c${C}_${label}_r$rep" "$root/rep$rep/tracelab_c$C/$label" runner "$label" "$tree" "$resv" "$C" "$floor" "$method"; done; done

# Frozen 32-64K cohort + ShareGPT 8 req/s, C = 16.
export AGENT_N=96 AGENT_GROUPS=48 AGENT_SEED=20261014 OTHER_N=2048 OTHER_RATE=8
export AGENT_MIN_CTX=32768 AGENT_MAX_CTX=65536 CONTEXT_LEN=131072 YARN_FACTOR=4.0
export BENCH_TRACE_MAX_TOTAL_TOKENS=131072
unset AGENT_RATE CHAT_RATE
for v in "${ARMS[@]}"; do IFS='|' read -r label tree resv floor method <<< "$v"
  run_arm "cohort_c16_${label}_r$rep" "$root/rep$rep/cohort_c16/$label" driver "$label" "$tree" "$resv" 16 "$floor" "$method"; done
  export TRACE_PATH=$W/workloads/tracelab/syfi_full_min.jsonl.gz
  export YARN_FACTOR=16.0 CONTEXT_LEN=524288 AGENT_MAX_CTX=1040000 AGENT_N=240 AGENT_GROUPS=8 AGENT_SEED=1014
  export CHAT_RATE=0 AGENT_RATE=inf AGENT_MIN_CTX=0; unset BENCH_TRACE_MAX_TOTAL_TOKENS OTHER_N OTHER_RATE
done
echo "CONF_DONE $(date -Is)"
