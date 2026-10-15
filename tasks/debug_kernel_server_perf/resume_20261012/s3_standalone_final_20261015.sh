#!/bin/bash
# S3 standalone workloads with the final serving configuration (2026-10-15), replacing the arms of
# s3_standalone_20261013.sh (never run): MTP | split static chunk 3072 | split + EDF 50 ms + prefill
# floor 2048, all in tree up_edf_floor with guard 0 and no mixed-step barrier (race fix), CUDA graphs
# on, YaRN x4 / 131072. The EDF arm uses the split step model fitted on the 32-64K cohort
# (edf_dynchunk_20261015); short standalone prompts are inside its token range but below its
# context range. Layout <root>/<workload>_r<point>/<ARM>_main.json for drivers/serving/eval_curves.py.
# Load axis: request rate (req/s) only -- no session-start rates. Mooncake replays its recorded
# per-request timestamps compressed by 1/slowdown (Crossflow's protocol); each END line reports the
# achieved request rate so its point is stated in req/s. Queued behind resv_gate_20261015.sh;
# yields to /dev/shm/w/codex_needs_gpu before every arm.
# Claude, 2026-10-15.
set -uo pipefail
W=/dev/shm/w
root=$W/s3_standalone_final_20261015
until grep -q 'GATE_DONE' $W/resv_gate_20261015.log 2>/dev/null; do sleep 60; done
exec 9>"$W/a2_kernel.lock"; flock 9
trap 'printf "S3F_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
# A predecessor's server can still be exiting after it releases the lock; wait up to 5 min.
for _ in $(seq 60); do pgrep -f '[s]glang.launch_server|[k]ernel_concurrency_driver|[f]used_reference|[r]un_mixed' >/dev/null || break; sleep 5; done
if pgrep -f '[s]glang.launch_server|[k]ernel_concurrency_driver|[f]used_reference|[r]un_mixed' >/dev/null; then echo "GPU predecessor"; exit 1; fi
mkdir -p "$root"
export PYTHONPATH=$W/up_edf_floor/python HF_HOME=$W/hf MODEL=$W/qwen7b HF_DATASETS_OFFLINE=1
export SGLANG_FUSED_FI_ROOT=$W/port07
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112 SGLANG_SPLIT_OVERLAP=0
export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3 SGLANG_FUSED_DECODE_GUARD=0
export TTFT_SLO_MS=2000 SGLANG_EDF_ACCEPT_LOG_EVERY=20000
export MTP_STEP_MODEL=$W/edf_dynchunk_20261015/models/MTPMIX_SPLIT_step_model.json
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export YARN_FACTOR=4.0 CONTEXT_LEN=131072
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH BENCH_CHUNK SGLANG_FUSED_NATIVE_PDL SGLANG_FUSED_KERNEL_EXEC
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE SGLANG_FUSED_VERIFY_SNAPSHOT SGLANG_FUSED_DISPATCH
unset FLASHINFER_FUSED_PLAN_DEBUG SGLANG_STEP_TRACE SGLANG_EDF_SCHEDULING SGLANG_EDF_LATENCY_MODEL SGLANG_EDF_ACCEPT_MODE
unset FUSED_NOSPLIT FUSED_FA3_UNPACKED FUSED_FA3_CLUSTER_PAIR FUSED_PAGED_TMA_BUILD SERVER_EXTRA_ARGS STREAMS
unset BENCH_SESSION_CONCURRENCY AGENT_RATE CODEX_RATE
test -s "$MTP_STEP_MODEL"
runner=$W/up_edf_floor/tasks/realistic_workload_analysis/run_mixed_workloads.py
test -f "$runner" || runner=$W/up_racefix/tasks/realistic_workload_analysis/run_mixed_workloads.py
sha256sum "$PYTHONPATH/sglang/srt/managers/scheduler.py" "$PYTHONPATH/sglang/srt/managers/edf_scheduler.py" \
  "$runner" "$MTP_STEP_MODEL" > "$root/sources.sha256"
S=$W/workloads/standalone
# name|STANDALONE|points|seconds-per-point (N = point * seconds, at least 100; 0 = Mooncake, N 600)
WORKLOADS=("sharegpt|sharegpt|4 8 16 32|120"
           "gsm8k|sharegpt:$S/gsm8k.json|8 16 32|90"
           "math500|sharegpt:$S/math500.json|8 16 32|60"
           "humaneval|sharegpt:$S/humaneval.json|8 16 32|20"
           "codefeedback|sharegpt:$S/codefeedback.json|8 16 32|90"
           "mooncake_conversation|mooncake_conversation|0.5 1 2|0"
           "mooncake_toolagent|mooncake_toolagent|0.5 1 2|0"
           "longbench_v2|longbench_v2|0.25 0.5 1|400")
for wl in "${WORKLOADS[@]}"; do
  IFS='|' read -r name kind points secs <<< "$wl"
  for pt in $points; do
    export STANDALONE=$kind STANDALONE_RATE=$pt
    if [ "$secs" = 0 ]; then
      export STANDALONE_N=600 MOONCAKE_SLOWDOWN=$(python3 -c "print(1/$pt)")
    else
      export STANDALONE_N=$(python3 -c "print(max(100, int($pt * $secs)))")
    fi
    out=$root/${name}_r$pt; mkdir -p "$out"
    for v in "MTP_LPT||0" "MTPMIX_SPLIT|3072|0" "MTPMIX_SPLIT_EDFR||2048"; do
      IFS='|' read -r method chunk floor <<< "$v"
      [ -s $W/codex_needs_gpu ] && { echo "S3F_YIELD $name r$pt $method"; exit 0; }
      [ -f "$out/${method}_main.json" ] && continue
      if [ -n "$chunk" ]; then export BENCH_CHUNK=$chunk; else unset BENCH_CHUNK; fi
      echo "S3F_START $name r$pt $method n=$STANDALONE_N $(date -Is)"
      ( cd "$(dirname "$runner")" && env ITL_SLO_MS=50 SGLANG_EDF_PREFILL_FLOOR=$floor timeout --kill-after=30 3600 \
          "$W/venv/bin/python" -u "$runner" --scenario standalone_env --methods "$method" --outdir "$out" > "$out/driver_$method.log" 2>&1 )
      rc=$?; L=$out/server_$method.log
      echo "S3F_END $name r$pt $method rc=$rc $(date -Is) tracebacks=$(grep -c Traceback $L) fallback=$(grep -ci fallback $L) split=[$(grep -o 'FUSED_SPLIT_CALLS [0-9]*' $L | tail -1)] edf_chunk=[$(grep -o 'EDF_RUNTIME_CHUNK calls=[0-9]* requested=[0-9]* admitted=[0-9]* prefill=[0-9]*' $L | tail -1)] req_s=[$(grep -h 'Request throughput' $out/bench_${method}_main.log 2>/dev/null | awk '{print $NF}')]"
    done
  done
done
echo "S3F_DONE $(date -Is)"
