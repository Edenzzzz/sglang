#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
prototype=$workdir/stock_fa3_unsplit_work_20261014
root=$workdir/stock_fa3_unsplit_20261014
trap 'status=$?; printf "STOCK_UNSPLIT_RESUME_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
grep -q '^STOCK_UNSPLIT_PHASE_PASS ' "$workdir/stock_fa3_unsplit_20261014.log"
grep -q '^STOCK_UNSPLIT_EXIT status=1 ' "$workdir/stock_fa3_unsplit_20261014.log"
if pgrep -f '[s]glang.launch_server|[c]orrectness.py|[k]ernel_concurrency_driver' >/dev/null; then exit 1; fi
cd "$workdir"
sha256sum -c "$prototype/candidate.sha256"
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

export STOCK_FLASH_ATTENTION=$workdir/stock_fa3_pdl_work_20261014/flash-attention
unset SGLANG_KERNEL_LOGIT_AUDIT SGLANG_MIXED_VERIFY_CHECK
gate=$root/serving_gate
driver=$prototype/correctness.py
test ! -e "$gate/server_MTPMIXFUSED_UNSPLIT_SPECIALIZED_VERIFY.log"
sha256sum "$driver" "$prototype/test_correctness.py" > "$root/correctness_fix.sha256"
"$workdir/venv/bin/python" "$prototype/test_correctness.py" > "$root/correctness_tests.log" 2>&1
"$workdir/venv/bin/python" "$driver" --audit "$gate" MTPMIX_SPLIT_UNSPLIT_CONTROL
"$workdir/venv/bin/python" "$driver" --audit "$gate" MTPMIXFUSED_UNSPLIT_GENERIC_VERIFY
grep -q 'STOCK_FA3_SPLIT_POLICY.*selected=1' "$gate/server_MTPMIXFUSED_UNSPLIT_GENERIC_VERIFY.log"
grep -q 'FUSED_KERNEL_VERIFY PASS.*native_pdl=1' "$gate/server_MTPMIXFUSED_UNSPLIT_GENERIC_VERIFY.log"
comparison_status=0
if "$workdir/venv/bin/python" "$driver" --compare "$gate" MTPMIX_SPLIT_UNSPLIT_CONTROL MTPMIXFUSED_UNSPLIT_GENERIC_VERIFY > "$gate/MTPMIXFUSED_UNSPLIT_GENERIC_VERIFY.compare.log" 2>&1; then
  echo "STOCK_UNSPLIT_COMPARE_PASS MTPMIXFUSED_UNSPLIT_GENERIC_VERIFY"
else
  comparison_status=1
  echo "STOCK_UNSPLIT_COMPARE_FAIL MTPMIXFUSED_UNSPLIT_GENERIC_VERIFY"
fi
for variant in 'MTPMIXFUSED_UNSPLIT_SPECIALIZED_VERIFY|1|1|1' 'MTPMIXFUSED_UNSPLIT_SPECIALIZED|1|1|0' 'MTPMIX_SPLIT_UNSPLIT_REPEAT|0|0|0'; do
  IFS='|' read -r arm native specialized verify <<< "$variant"
  export STOCK_FA3_UNSPLIT_SPECIALIZE=$specialized
  export PYTHONPATH=$workdir/up_stock_fa3_unsplit/python:$workdir/stock_fa3_pdl_work_20261014
  export STOCK_FA3_EXTENSIONS=$workdir/torch_ext_stock_fa3_pdl STOCK_FA3_MERGED=$workdir/merged_stock_fa3_pdl
  if [[ "$specialized" == 1 ]]; then
    export PYTHONPATH=$workdir/up_stock_fa3_unsplit/python:$prototype
    export STOCK_FA3_EXTENSIONS=$workdir/torch_ext_stock_fa3_unsplit STOCK_FA3_MERGED=$workdir/merged_stock_fa3_unsplit
  fi
  echo "STOCK_UNSPLIT_GATE_START $arm specialized=$specialized $(date -Is)"
  CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET="$CHAT_DATASET" BENCH_CHUNK=512 \
    CORRECTNESS_LONG_TOKENS=49152 CORRECTNESS_DECODE_TOKENS=1200 \
    SGLANG_FUSED_DECODE_GUARD=0 SGLANG_FUSED_SM_FA3=132 \
    SGLANG_FUSED_NATIVE_PDL=$native SGLANG_STOCK_FA3_PDL=$native SGLANG_STOCK_FA3_PHASE=2 \
    SGLANG_STOCK_FA3_SPLIT_OVERRIDE=$native \
    SGLANG_FUSED_SPLIT_VERIFY=$verify SGLANG_FUSED_VERIFY_SNAPSHOT=$verify SGLANG_FUSED_SPLIT_TOL=0.02 \
    timeout --kill-after=30 1800 "$workdir/venv/bin/python" -u "$driver" "$arm" "$gate" > "$gate/$arm.run.log" 2>&1
  grep -q 'MIXEDSPEC_STEPS' "$gate/server_$arm.log"
  if [[ "$native" == 1 ]]; then
    grep -q 'STOCK_FA3_SPLIT_POLICY.*selected=1' "$gate/server_$arm.log"
    grep -q 'STOCK_FA3_SERVING path=stock_fa3_calls.*splits=1' "$gate/server_$arm.log"
    if grep -E 'STOCK_FA3_SERVING.*splits=([2-9]|[1-9][0-9])' "$gate/server_$arm.log"; then exit 1; fi
  fi
  if [[ "$verify" == 1 ]]; then grep -q 'FUSED_KERNEL_VERIFY PASS.*native_pdl=1' "$gate/server_$arm.log"; fi
  if [[ "$arm" != MTPMIX_SPLIT_UNSPLIT_CONTROL ]]; then
    if "$workdir/venv/bin/python" "$driver" --compare "$gate" MTPMIX_SPLIT_UNSPLIT_CONTROL "$arm" > "$gate/$arm.compare.log" 2>&1; then
      echo "STOCK_UNSPLIT_COMPARE_PASS $arm"
    else
      comparison_status=1
      echo "STOCK_UNSPLIT_COMPARE_FAIL $arm"
    fi
  fi
done
echo "STOCK_UNSPLIT_COMPLETE comparison_status=$comparison_status $(date -Is)"
exit "$comparison_status"
