#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
root=$workdir/stock_fa3_serving_profile_20261014
profile_pid=
cleanup() {
  status=$?
  if [[ -n "$profile_pid" ]]; then
    kill "$profile_pid" 2>/dev/null || true
    wait "$profile_pid" 2>/dev/null || true
  fi
  printf 'STOCK_FA3_PROFILE_EXIT status=%s time=%s\n' "$status" "$(date -Is)"
}
trap cleanup EXIT
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
grep -q '^STOCK_FA3_SERVING_AB_DONE ' "$workdir/stock_fa3_serving_ab_20261014.log"
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[k]ernel_concurrency_driver' >/dev/null; then exit 1; fi
cd "$workdir"
sha256sum -c stock_fa3_serving_ab_overlay_20261014/candidate.sha256
test ! -e "$root"
mkdir "$root"
sha256sum up_stock_fa3_ab/python/sglang/srt/layers/attention/{flashinfer_backend,fused_fa3_loader}.py \
  up_stock_fa3_ab/python/sglang/benchmark/{serving,session_replay,session_concurrency,workload_protocol}.py \
  up_stock_fa3_ab/python/sglang/benchmark/datasets/tracelab.py \
  stock_fa3_pdl_work_20261014/{load_stock_fa3.py,stock_fa3_pdl.cu} \
  drivers/kernel_concurrency_driver.py > "$root/sources.sha256"
source "$workdir/tree_env.sh"
export PYTHONPATH=$workdir/up_stock_fa3_ab/python:$workdir/stock_fa3_pdl_work_20261014
export HF_HOME=$workdir/hf MODEL=$workdir/qwen7b
export SGLANG_FUSED_FI_ROOT=$workdir/fi_fa3_kv_geometry_v2
export SGLANG_FUSED_MERGED_DIR=$workdir/merged_stock_serving_native_release
export TORCH_EXTENSIONS_DIR=$workdir/torch_ext_stock_serving_native_release
export STOCK_FA3_EXTENSIONS=$workdir/torch_ext_stock_fa3_pdl
export STOCK_FA3_MERGED=$workdir/merged_stock_fa3_pdl
export STOCK_FLASH_ATTENTION=$workdir/stock_fa3_pdl_work_20261014/flash-attention
export NATIVE_RELEASE_BUILD=1 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 SGLANG_FUSED_SM_FA3=112
export FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7 FA3_CTA_KV=128 FA3_STAGES=2
export SGLANG_FUSED_FA3_CONSUMER_REGS=216 FUSED_FA3_DEDICATED_PRIMARY=1 FUSED_FA3_CLUSTER_PAIR=1
export FUSED_PAGED_TMA_BUILD=0 FUSED_CLUSTER=1 FLASHINFER_FUSED_CLUSTER=1
export SGLANG_SPLIT_OVERLAP=0 MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
export CHAT_DATASET=$workdir/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export TRACE_PATH=$workdir/workloads/tracelab/syfi_full_min.jsonl.gz
export AGENT_N=96 AGENT_GROUPS=48 AGENT_SEED=20261014 OTHER_N=2048 OTHER_RATE=8
export AGENT_MIN_CTX=32768 AGENT_MAX_CTX=65536 CONTEXT_LEN=131072 YARN_FACTOR=4.0
export BENCH_TRACE_MAX_TOTAL_TOKENS=131072 BENCH_SESSION_CONCURRENCY=24
unset FUSED_NOSPLIT FUSED_KV_SHARE_BUILD FUSED_FA3_COMPACT_ABI FUSED_FA3_NOSPLIT_SPECIALIZE
unset SGLANG_FUSED_FA3_STATIC_GQA FUSED_KV_CHUNK_MULT FUSED_FA3_UNPACKED
unset SGLANG_EDF_SCHEDULING SGLANG_EDF_LATENCY_MODEL SGLANG_EDF_ACCEPT_MODE SGLANG_DEBUG_STEP_SYNC
unset SGLANG_FUSED_CHAIN_TREE SGLANG_FUSED_VERIFY_EPILOGUE DISABLE_CUDA_GRAPH SGLANG_STEP_TRACE
unset SGLANG_TWO_EXEC SGLANG_MIXED_ATTENTION_KERNEL SGLANG_ALLOW_MIXED_CHUNK_SPEC SGLANG_MIXED_SPEC
unset SGLANG_FUSED_KERNEL_EXEC SGLANG_FUSED_DISPATCH SGLANG_FUSED_DECODE_GUARD
unset BENCH_WARMUP_CHAIN BENCH_MAX_CONCURRENCY BENCH_NO_BARRIER BENCH_CHUNK SERVER_EXTRA_ARGS STREAMS
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_SNAPSHOT SGLANG_FUSED_TILE_STATS
for label in SPLIT STOCK_PDL112; do
  method=MTPMIX_SPLIT
  native=0
  if [[ "$label" == STOCK_PDL112 ]]; then method=MTPMIXFUSED; native=1; fi
  export SGLANG_FUSED_NATIVE_PDL=$native SGLANG_STOCK_FA3_PDL=$native SGLANG_STOCK_FA3_PHASE=2
  output=$root/$label
  mkdir -p "$output/prof"
  export SGLANG_TORCH_PROFILER_DIR=$output/prof
  echo "STOCK_FA3_PROFILE_START $label $(date -Is)"
  (
    ready=0
    for attempt in $(seq 1 300); do
      if grep -q 'barrier: 2/2 streams ready; released' "$output/driver.log" 2>/dev/null; then ready=1; break; fi
      sleep 1
    done
    [[ "$ready" == 1 ]]
    sleep 30
    date -Is > "$output/profile_trigger_time.txt"
    curl --fail --silent --show-error --max-time 180 -X POST http://127.0.0.1:30013/start_profile \
      -H 'Content-Type: application/json' \
      -d "{\"output_dir\":\"$output/prof\",\"num_steps\":80,\"activities\":[\"CPU\",\"GPU\"]}" \
      > "$output/start_profile.json"
  ) &
  profile_pid=$!
  timeout --kill-after=30 5400 "$workdir/venv/bin/python" -u drivers/kernel_concurrency_driver.py \
    --scenario kernel_concurrency_env --methods "$method" --outdir "$output" > "$output/driver.log" 2>&1
  wait "$profile_pid"
  profile_pid=
  "$workdir/venv/bin/python" stock_fa3_serving_ab_overlay_20261014/audit_stock_serving.py \
    "$output" "$method" --stock "$native" --phase 2 --budget 112 \
    --reference "$workdir/stock_fa3_serving_ab_20261014/MTP" > "$output/audit.log" 2>&1
  shopt -s nullglob
  traces=("$output"/prof/*.json "$output"/prof/*.json.gz)
  (( ${#traces[@]} > 0 ))
  "$workdir/venv/bin/python" drivers/serving/parse_kernel_tail.py "${traces[@]}" > "$output/kernel_tail.txt"
  echo "STOCK_FA3_PROFILE_DONE $label $(date -Is)"
done
echo "STOCK_FA3_PROFILE_COMPLETE $(date -Is)"
