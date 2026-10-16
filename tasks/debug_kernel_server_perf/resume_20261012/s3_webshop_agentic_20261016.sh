#!/bin/bash
# S3 agentic web browsing (user, 2026-10-16: standalone S3 = ShareGPT, math, coding and agentic web
# browsing only). THUDM/AgentInstruct WebShop: 351 multi-turn web-shopping trajectories (4-10 turns,
# mean 4.7), converted by drivers/serving/prep_webshop_agentic.py to the agentic-trace format and replayed
# round by round with the server's own replies (backend sglang-oai-chat, 64 output tokens per turn).
# Load is program concurrency (ThunderAgent): C agents in flight, closed loop, C = 8 / 16 / 32 / 64.
# Arms: MTP | split static 3072 | split + EDF 50 ms + floor 2048 | split + EDF + floor + chunk preemption
# (SRPT, the TraceLab winner), all in tree up_edf_preempt, guard 0. Runner: run_agentic_standalone.py
# (run_mixed_workloads.py + an "agentic:<json>" workload and BENCH_BACKEND). Claude, 2026-10-16.
set -uo pipefail
W=/dev/shm/w
root=$W/s3_webshop_agentic_20261016
exec 9>"$W/a2_kernel.lock"; flock 9
trap 'printf "WEB_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
for _ in $(seq 60); do pgrep -f '[s]glang.launch_server|[k]ernel_concurrency_driver|[r]un_mixed|[r]un_agentic' >/dev/null || break; sleep 5; done
if pgrep -f '[s]glang.launch_server|[k]ernel_concurrency_driver|[r]un_mixed|[r]un_agentic' >/dev/null; then echo "GPU predecessor"; exit 1; fi
mkdir -p "$root"
export PYTHONPATH=$W/up_edf_preempt/python HF_HOME=$W/hf MODEL=$W/qwen7b HF_DATASETS_OFFLINE=1
export SGLANG_FUSED_FI_ROOT=$W/port07
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112 SGLANG_SPLIT_OVERLAP=0
export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3 SGLANG_FUSED_DECODE_GUARD=0
export TTFT_SLO_MS=2000 SGLANG_EDF_ACCEPT_LOG_EVERY=20000
export MTP_STEP_MODEL=$W/edf_dynchunk_20261015/models/MTPMIX_SPLIT_step_model.json
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export YARN_FACTOR=4.0 CONTEXT_LEN=131072 BENCH_BACKEND=sglang-oai-chat AGENTIC_OUTPUT_LEN=64
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH BENCH_CHUNK SERVER_EXTRA_ARGS STREAMS BENCH_SESSION_CONCURRENCY
runner=$W/up_racefix/tasks/realistic_workload_analysis/run_agentic_standalone.py
data=$W/workloads/standalone/webshop_agentic.json
test -s "$data" && test -f "$runner" && test -s "$MTP_STEP_MODEL"
sha256sum "$runner" "$data" "$PYTHONPATH/sglang/srt/managers/scheduler.py" "$MTP_STEP_MODEL" > "$root/sources.sha256"
export STANDALONE=agentic:$data STANDALONE_N=351 STANDALONE_RATE=inf
for C in 8 16 32 64; do
  export AGENTIC_CONCURRENCY=$C
  out=$root/c$C; mkdir -p "$out"
  # label|method|chunk|floor|preempt
  for v in "MTP|MTP_LPT||0|0" "SPLIT_C3072|MTPMIX_SPLIT|3072|0|0" "EDF|MTPMIX_SPLIT_EDFR||2048|0" "EDFP|MTPMIX_SPLIT_EDFR||2048|1"; do
    IFS='|' read -r label method chunk floor pre <<< "$v"
    [ -s $W/codex_needs_gpu ] && { echo "WEB_YIELD c$C $label"; exit 0; }
    d=$out/$label; mkdir -p "$d"
    [ -f "$d/${method}_main.json" ] && continue
    if [ -n "$chunk" ]; then export BENCH_CHUNK=$chunk; else unset BENCH_CHUNK; fi
    echo "WEB_START c$C $label method=$method $(date -Is)"
    ( cd "$(dirname "$runner")" && env ITL_SLO_MS=50 SGLANG_EDF_PREFILL_FLOOR=$floor SGLANG_EDF_PREEMPT_CHUNK=$pre \
        SGLANG_EDF_RESERVE_EARLIER=$pre SGLANG_EDF_PREEMPT_MAX_TOKENS=1000000000 timeout --kill-after=30 3600 \
        "$W/venv/bin/python" -u "$runner" --scenario standalone_env --methods "$method" --outdir "$d" > "$d/driver.log" 2>&1 )
    rc=$?; L=$d/server_$method.log
    echo "WEB_END c$C $label rc=$rc $(date -Is) tracebacks=$(grep -c Traceback $L) fallback=$(grep -ci fallback $L) split=[$(grep -o 'FUSED_SPLIT_CALLS [0-9]*' $L | tail -1)] park=[$(grep -o 'EDF_PARK calls=[0-9]*' $L | tail -1)] conv=[$(grep -h 'Successful requests' $d/bench_${method}_main.log | awk '{print $NF}')]"
  done
done
echo "WEB_DONE $(date -Is)"
