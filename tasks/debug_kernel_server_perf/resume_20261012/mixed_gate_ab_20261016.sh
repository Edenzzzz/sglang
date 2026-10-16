#!/bin/bash
# 4-mode context gate A/B (2026-10-16). S3 showed the mixed step loses 7-10% on short-prompt workloads
# (ShareGPT, math, coding, WebShop) and wins where long-context prefill shares the GPU with decode.
# SGLANG_MIXED_SPEC_MIN_CTX (mixed_spec_ctx_gate.patch, tree up_edf_gate = up_edf_preempt + gate): a step
# co-schedules its prefill with the verify rows only if some prefill row's prefix + chunk >= the gate;
# otherwise the prefill runs as its own step, as in stock MTP. Arms per point: MTP | EDF + floor +
# preemption (ungated) | same + gate 8192. Points: WebShop C=16, ShareGPT 16 req/s, cohort C=16 + chat
# 8 req/s, TraceLab C=8. Claude, 2026-10-16.
set -uo pipefail
W=/dev/shm/w
root=$W/mixed_gate_ab_20261016
exec 9>"$W/a2_kernel.lock"; flock 9
trap 'printf "GATEAB_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
for _ in $(seq 60); do pgrep -f '[s]glang.launch_server|[k]ernel_concurrency_driver|[r]un_mixed|[r]un_agentic' >/dev/null || break; sleep 5; done
if pgrep -f '[s]glang.launch_server|[k]ernel_concurrency_driver|[r]un_mixed|[r]un_agentic' >/dev/null; then echo "GPU predecessor"; exit 1; fi
mkdir -p "$root"; cd "$W"
export PYTHONPATH=$W/up_edf_gate/python HF_HOME=$W/hf MODEL=$W/qwen7b HF_DATASETS_OFFLINE=1
export SGLANG_FUSED_FI_ROOT=$W/port07
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112 SGLANG_SPLIT_OVERLAP=0
export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3 SGLANG_FUSED_DECODE_GUARD=0
export TTFT_SLO_MS=2000 SGLANG_EDF_ACCEPT_LOG_EVERY=20000 ITL_SLO_MS=50
export SGLANG_EDF_MAX_PROFILED_PREFILL_PREFIX=262144 SGLANG_EDF_LONG_PREFIX_PROJECTION_THRESHOLD=262145
export MTP_STEP_MODEL=$W/edf_dynchunk_20261015/models/MTPMIX_SPLIT_step_model.json
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH BENCH_CHUNK SERVER_EXTRA_ARGS STREAMS BENCH_SESSION_CONCURRENCY BENCH_BACKEND
grep -q MIXED_SPEC_MIN_CTX $PYTHONPATH/sglang/srt/managers/scheduler.py
sha256sum $PYTHONPATH/sglang/srt/managers/scheduler.py "$MTP_STEP_MODEL" > "$root/sources.sha256"
RUN=$W/up_racefix/tasks/realistic_workload_analysis
ARMS=("MTP|MTP_LPT|0|0|0" "EDFP|MTPMIX_SPLIT_EDFR|2048|1|0" "EDFP_G8K|MTPMIX_SPLIT_EDFR|2048|1|8192")
arm() {  # point label method floor preempt gate kind
  local point=$1 label=$2 method=$3 floor=$4 pre=$5 gate=$6 kind=$7 d=$root/$1/$2
  [ -s $W/codex_needs_gpu ] && { echo "GATEAB_YIELD $point $label"; exit 0; }
  mkdir -p "$d"; echo "GATEAB_START $point $label $(date -Is)"
  local E="SGLANG_EDF_PREFILL_FLOOR=$floor SGLANG_EDF_PREEMPT_CHUNK=$pre SGLANG_EDF_RESERVE_EARLIER=$pre SGLANG_EDF_PREEMPT_MAX_TOKENS=1000000000 SGLANG_MIXED_SPEC_MIN_CTX=$gate"
  case $kind in
    web) ( cd $RUN && env $E BENCH_BACKEND=sglang-oai-chat STANDALONE=agentic:$W/workloads/standalone/webshop_agentic.json \
           STANDALONE_N=351 STANDALONE_RATE=inf AGENTIC_CONCURRENCY=16 AGENTIC_OUTPUT_LEN=64 YARN_FACTOR=4.0 CONTEXT_LEN=131072 \
           timeout --kill-after=30 3600 $W/venv/bin/python -u run_agentic_standalone.py --scenario standalone_env --methods $method --outdir $d > $d/driver.log 2>&1 ) ;;
    sgpt) ( cd $RUN && env $E STANDALONE=sharegpt STANDALONE_N=1920 STANDALONE_RATE=16 YARN_FACTOR=4.0 CONTEXT_LEN=131072 \
           timeout --kill-after=30 3600 $W/venv/bin/python -u run_mixed_workloads.py --scenario standalone_env --methods $method --outdir $d > $d/driver.log 2>&1 ) ;;
    cohort) env $E BENCH_SESSION_CONCURRENCY=16 TRACE_PATH=$W/workloads/tracelab/syfi_full_min.jsonl.gz AGENT_N=96 AGENT_GROUPS=48 \
           AGENT_SEED=20261014 OTHER_N=2048 OTHER_RATE=8 AGENT_MIN_CTX=32768 AGENT_MAX_CTX=65536 CONTEXT_LEN=131072 YARN_FACTOR=4.0 \
           BENCH_TRACE_MAX_TOTAL_TOKENS=131072 timeout --kill-after=30 5400 $W/venv/bin/python -u drivers/kernel_concurrency_driver.py \
           --scenario kernel_concurrency_env --methods $method --outdir $d > $d/driver.log 2>&1 ;;
    trace) ( cd $RUN && env $E BENCH_SESSION_CONCURRENCY=8 BENCH_SESSION_AUDIT_FILE=$d/${method}_agent_programs.json \
           TRACE_PATH=$W/workloads/tracelab/syfi_full_min.jsonl.gz YARN_FACTOR=16.0 CONTEXT_LEN=524288 AGENT_MAX_CTX=1040000 \
           AGENT_N=240 AGENT_GROUPS=8 AGENT_SEED=1014 CHAT_RATE=0 AGENT_RATE=inf timeout --kill-after=30 7200 \
           $W/venv/bin/python -u run_mixed_workloads.py --scenario tracelab_full_env --methods $method --outdir $d > $d/driver.log 2>&1 ) ;;
  esac
  local rc=$? L=$d/server_$method.log
  echo "GATEAB_END $point $label rc=$rc $(date -Is) tracebacks=$(grep -c Traceback $L) fallback=$(grep -ci fallback $L) mixed=[$(grep -o 'MIXEDSPEC_STEPS [0-9]*' $L | tail -1)] gated=[$(grep -o 'MIXEDSPEC_GATED [0-9]*' $L | tail -1)] park=[$(grep -o 'EDF_PARK calls=[0-9]*' $L | tail -1)]"
}
for pk in "webshop_c16|web" "sharegpt_r16|sgpt" "cohort_c16|cohort" "tracelab_c8|trace"; do
  IFS='|' read -r point kind <<< "$pk"
  for v in "${ARMS[@]}"; do IFS='|' read -r label method floor pre gate <<< "$v"; arm $point $label $method $floor $pre $gate $kind; done
done
echo "GATEAB_DONE $(date -Is)"
