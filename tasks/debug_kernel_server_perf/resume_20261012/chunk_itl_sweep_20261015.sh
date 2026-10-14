#!/bin/bash
# Chat mean ITL is the one metric short of the bar on Codex's best point (split, guard0, vs MTP:
# -25.6%). Chat ITL is the step time, and with guard0 the steps carry long agent prefill chunks on
# FA3, so test the mixed-step prefill chunk (BENCH_CHUNK; default 8192) on the same frozen workload
# and auditors as stock_fa3_single_serving_20261015.sh (Codex). MTP also runs at the smallest chunk
# so the comparison does not tune only our side. The split's output gate already passed; chunk size
# changes batching, not kernel math. Claude, 2026-10-15.
set -euo pipefail
workdir=/dev/shm/w
prototype=$workdir/stock_fa3_unsplit_work_20261014
overlay=$workdir/stock_fa3_single_serving_overlay_20261015
root=$workdir/chunk_itl_sweep_20261015
trap 'status=$?; printf "CHUNK_ITL_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
grep -q '^STOCK_UNSPLIT_COMPLETE comparison_status=0 ' "$workdir/stock_fa3_unsplit_resume_20261014.log"
if pgrep -f '[s]glang.launch_server|[c]orrectness.py|[k]ernel_concurrency_driver' >/dev/null; then exit 1; fi
cd "$workdir"
sha256sum -c "$prototype/candidate.sha256"
test ! -e "$root"
mkdir "$root"
cp "$prototype/"{baseline,candidate}.sha256 "$root/"
sha256sum "$prototype/correctness.py" "$overlay/"*.py \
  "$workdir/torch_ext_stock_fa3_pdl/stock_fa3_pdl/stock_fa3_pdl.so" \
  "$workdir/torch_ext_stock_fa3_unsplit/stock_fa3_pdl/stock_fa3_pdl.so" > "$root/runtime.sha256"
grep -q -- '-DSTOCK_FA3_UNSPLIT_SPECIALIZE=1' "$workdir/torch_ext_stock_fa3_unsplit/stock_fa3_pdl/build.ninja"
cp "$workdir/torch_ext_stock_fa3_unsplit/stock_fa3_pdl/build.ninja" "$root/specialized_build.ninja"
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

export SGLANG_FUSED_DECODE_GUARD=0
export STOCK_FLASH_ATTENTION=$workdir/stock_fa3_pdl_work_20261014/flash-attention
unset SGLANG_KERNEL_LOGIT_AUDIT SGLANG_MIXED_VERIFY_CHECK SGLANG_KERNEL_ROLE_CONTROL
configure_stock() {
  specialized=$1
  export STOCK_FA3_UNSPLIT_SPECIALIZE=$specialized
  export PYTHONPATH=$workdir/up_stock_fa3_unsplit/python:$workdir/stock_fa3_pdl_work_20261014
  export STOCK_FA3_EXTENSIONS=$workdir/torch_ext_stock_fa3_pdl STOCK_FA3_MERGED=$workdir/merged_stock_fa3_pdl
  if [[ "$specialized" == 1 ]]; then
    export PYTHONPATH=$workdir/up_stock_fa3_unsplit/python:$prototype
    export STOCK_FA3_EXTENSIONS=$workdir/torch_ext_stock_fa3_unsplit STOCK_FA3_MERGED=$workdir/merged_stock_fa3_unsplit
  fi
}
gate=$root/sm112_gate
mkdir "$gate"
driver=$prototype/correctness.py
baseline=$workdir/stock_fa3_unsplit_20261014/serving_gate/MTPMIX_SPLIT_UNSPLIT_CONTROL.json
cp "$baseline" "$gate/MTPMIX_SPLIT_UNSPLIT_CONTROL.json"
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_SNAPSHOT
variants=(
  'MTP|MTP_LPT|0|0|2|112|'
  'SPLIT_C8192_1|MTPMIX_SPLIT|0|0|2|112|'
  'SPLIT_C4096|MTPMIX_SPLIT|0|0|2|112|4096'
  'SPLIT_C2048|MTPMIX_SPLIT|0|0|2|112|2048'
  'MTP_C2048|MTP_LPT|0|0|2|112|2048'
  'SPLIT_C8192_2|MTPMIX_SPLIT|0|0|2|112|'
)
for variant in "${variants[@]}"; do
  IFS='|' read -r label method native specialized phase budget chunk <<< "$variant"
  configure_stock "$specialized"
  if [ -n "$chunk" ]; then export BENCH_CHUNK=$chunk; else unset BENCH_CHUNK; fi
  if grep -q CODEX_NEEDS_GPU /dev/shm/w/codex_needs_gpu 2>/dev/null; then echo "CHUNK_ITL_YIELD $label"; exit 0; fi
  export SGLANG_FUSED_NATIVE_PDL=$native SGLANG_STOCK_FA3_PDL=$native
  export SGLANG_STOCK_FA3_SPLIT_OVERRIDE=$native SGLANG_STOCK_FA3_PHASE=$phase SGLANG_FUSED_SM_FA3=$budget
  output=$root/$label
  mkdir "$output"
  echo "CHUNK_ITL_START $label method=$method chunk=${chunk:-8192} $(date -Is)"
  timeout --kill-after=30 5400 "$workdir/venv/bin/python" -u drivers/kernel_concurrency_driver.py \
    --scenario kernel_concurrency_env --methods "$method" --outdir "$output" > "$output/driver.log" 2>&1
  if [[ "$method" == MTPMIX_SPLIT || "$method" == MTPMIXFUSED ]]; then
    "$workdir/venv/bin/python" "$overlay/audit_stock_dispatch.py" "$output" "$method" \
      --stock "$native" --phase "$phase" --budget "$budget" --guard 0 \
      --reference "$root/MTP" --base-auditor "$overlay/audit_stock_serving.py" > "$output/audit.log" 2>&1
  else
    compare=()
    if [[ "$label" != MTP ]]; then compare=(--reference "$root/MTP"); fi
    "$workdir/venv/bin/python" "$overlay/audit_stock_serving.py" "$output" "$method" \
      --stock 0 --phase "$phase" --budget "$budget" "${compare[@]}" > "$output/audit.log" 2>&1
  fi
  echo "CHUNK_ITL_DONE $label $(date -Is)"
done
echo "CHUNK_ITL_COMPLETE $(date -Is)"
