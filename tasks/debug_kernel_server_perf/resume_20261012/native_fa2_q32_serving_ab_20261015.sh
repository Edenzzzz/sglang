#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
prototype=$workdir/native_fa2_q32_serving_work_20261015
overlay=$workdir/native_fa2_q32_serving_ab_overlay_20261015
root=$workdir/native_fa2_q32_serving_ab_20261015
trap 'status=$?; printf "NATIVE_Q32_SERVING_AB_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
grep -q '^NATIVE_Q32_SERVING_GATE_COMPLETE comparison_status=0 ' "$workdir/native_fa2_q32_serving_gate_20261015.log"
if pgrep -f '[s]glang.launch_server|[b]ench_native_q32.py|[t]est_native_q32.py|[c]oupled_reference.py|[k]ernel_concurrency_driver' >/dev/null; then exit 1; fi
cd "$workdir"
sha256sum -c "$prototype/candidate.sha256"
test ! -e "$root"
mkdir "$root"
cp "$prototype/candidate.sha256" "$root/"
sha256sum "$overlay/"*.py "$workdir/torch_ext_native_fa2_q32_v2/native_fa2_q32/native_fa2_q32.so" \
  "$workdir/torch_ext_stock_fa3_unsplit/stock_fa3_pdl/stock_fa3_pdl.so" > "$root/runtime.sha256"
source "$workdir/tree_env.sh"
export PYTHONPATH=$workdir/up_stock_fa3_dispatch/python:$workdir/stock_fa3_pdl_work_20261014
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
export SGLANG_KERNEL_DISPATCH_AUDIT=1
unset FUSED_NOSPLIT FUSED_KV_SHARE_BUILD FUSED_FA3_COMPACT_ABI FUSED_FA3_NOSPLIT_SPECIALIZE
unset SGLANG_FUSED_FA3_STATIC_GQA FUSED_KV_CHUNK_MULT FUSED_FA3_UNPACKED
unset SGLANG_EDF_SCHEDULING SGLANG_EDF_LATENCY_MODEL SGLANG_EDF_ACCEPT_MODE SGLANG_DEBUG_STEP_SYNC
unset SGLANG_FUSED_CHAIN_TREE SGLANG_FUSED_VERIFY_EPILOGUE DISABLE_CUDA_GRAPH SGLANG_STEP_TRACE
unset SGLANG_TWO_EXEC SGLANG_MIXED_ATTENTION_KERNEL SGLANG_ALLOW_MIXED_CHUNK_SPEC SGLANG_MIXED_SPEC
unset SGLANG_FUSED_KERNEL_EXEC SGLANG_FUSED_DISPATCH SGLANG_FUSED_DECODE_GUARD
unset BENCH_WARMUP_CHAIN BENCH_MAX_CONCURRENCY BENCH_NO_BARRIER BENCH_CHUNK SERVER_EXTRA_ARGS STREAMS
unset SGLANG_TORCH_PROFILER_DIR SGLANG_FUSED_TILE_STATS

export PYTHONPATH=$workdir/up_native_q32_serving/python:$workdir/stock_fa3_unsplit_work_20261014:$workdir/native_fa2_q32_v2_work_20261015
export NATIVE_Q32_MERGED=$workdir/merged_native_fa2_q32_v2 NATIVE_Q32_EXTENSIONS=$workdir/torch_ext_native_fa2_q32_v2
export STOCK_FA3_EXTENSIONS=$workdir/torch_ext_stock_fa3_unsplit STOCK_FA3_MERGED=$workdir/merged_stock_fa3_unsplit
export STOCK_FLASH_ATTENTION=$workdir/stock_fa3_pdl_work_20261014/flash-attention STOCK_FA3_UNSPLIT_SPECIALIZE=1
export SGLANG_FUSED_DECODE_GUARD=0 SGLANG_FUSED_SM_FA3=132
export SGLANG_FUSED_NATIVE_PDL=1 SGLANG_STOCK_FA3_PDL=1 SGLANG_STOCK_FA3_PHASE=2 SGLANG_STOCK_FA3_SPLIT_OVERRIDE=1
unset SGLANG_KERNEL_LOGIT_AUDIT SGLANG_MIXED_VERIFY_CHECK SGLANG_KERNEL_ROLE_CONTROL
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_SNAPSHOT
for variant in 'NATIVE64_1|64' 'NATIVE32_1|32' 'NATIVE32_2|32' 'NATIVE64_2|64'; do
  IFS='|' read -r label query_tile <<< "$variant"
  export SGLANG_NATIVE_FA2_QUERY_TILE=$query_tile
  output=$root/$label
  mkdir "$output"
  echo "NATIVE_QUERY_TILE_SERVING_START $label $(date -Is)"
  timeout --kill-after=30 5400 "$workdir/venv/bin/python" -u drivers/kernel_concurrency_driver.py \
    --scenario kernel_concurrency_env --methods MTPMIXFUSED --outdir "$output" > "$output/driver.log" 2>&1
  "$workdir/venv/bin/python" "$overlay/audit_native_q32_serving.py" "$output" --query-tile "$query_tile" \
    --reference "$workdir/stock_fa3_single_serving_20261015/MTP" \
    --dispatch-auditor "$workdir/stock_fa3_single_serving_overlay_20261015/audit_stock_dispatch.py" \
    --base-auditor "$workdir/stock_fa3_single_serving_overlay_20261015/audit_stock_serving.py" > "$output/audit.log" 2>&1
  echo "NATIVE_QUERY_TILE_SERVING_DONE $label $(date -Is)"
done
echo "NATIVE_Q32_SERVING_AB_COMPLETE $(date -Is)"
