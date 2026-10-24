#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
root=$workdir/stock_fa3_serving_ab_20261014
overlay=$workdir/stock_fa3_serving_ab_overlay_20261014
trap 'status=$?; printf "STOCK_FA3_SERVING_AB_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
grep -q '^STOCK_FA3_SERVING_GATE_DONE ' "$workdir/stock_fa3_serving_gate_20261014.log"
grep -q '^B1_KERNEL_CONCURRENCY_DONE ' "$workdir/b1_kernel_concurrency_20261014.log"
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[k]ernel_concurrency_driver|[s]tock_fa3_reference.py' >/dev/null; then exit 1; fi
cd "$workdir"
sha256sum -c "$overlay/baseline.sha256"
test ! -e up_stock_fa3_ab
test ! -e "$root"
mkdir up_stock_fa3_ab "$root"
cp -a up_stock_fa3_serving/python up_stock_fa3_ab/python
cp "$overlay/flashinfer_backend.py" up_stock_fa3_ab/python/sglang/srt/layers/attention/flashinfer_backend.py
for name in serving.py session_replay.py session_concurrency.py workload_protocol.py; do
  cp "up_b1_kernel_serving/python/sglang/benchmark/$name" "up_stock_fa3_ab/python/sglang/benchmark/$name"
done
cp up_b1_kernel_serving/python/sglang/benchmark/datasets/tracelab.py up_stock_fa3_ab/python/sglang/benchmark/datasets/tracelab.py
sha256sum -c "$overlay/candidate.sha256"
cp "$overlay/"{baseline,candidate}.sha256 "$root/"
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
gate=$root/full_sm_gate
mkdir "$gate"
driver=$workdir/drivers/srv_correctness_edf_20261014.py
for arm in MTPMIX_SPLIT_FULL_CONTROL MTPMIXFUSED_STOCK_FULL_VERIFY; do
  native=0
  verify=0
  if [[ "$arm" == *VERIFY ]]; then native=1; verify=1; fi
  echo "STOCK_FULL_GATE_START $arm $(date -Is)"
  CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET="$CHAT_DATASET" BENCH_CHUNK=512 \
    CORRECTNESS_DECODE_TOKENS=1200 SGLANG_FUSED_DECODE_GUARD=0 SGLANG_FUSED_SM_FA3=132 \
    SGLANG_FUSED_NATIVE_PDL=$native SGLANG_STOCK_FA3_PDL=$native SGLANG_STOCK_FA3_PHASE=2 \
    SGLANG_FUSED_SPLIT_VERIFY=$verify SGLANG_FUSED_VERIFY_SNAPSHOT=$verify SGLANG_FUSED_SPLIT_TOL=0.02 \
    timeout --kill-after=30 1800 "$workdir/venv/bin/python" -u "$driver" "$arm" "$gate" > "$gate/$arm.run.log" 2>&1
done
"$workdir/venv/bin/python" "$driver" --compare "$gate" MTPMIX_SPLIT_FULL_CONTROL MTPMIXFUSED_STOCK_FULL_VERIFY \
  > "$gate/compare.log" 2>&1
grep -q 'FUSED_KERNEL_VERIFY PASS.*native_pdl=1' "$gate/server_MTPMIXFUSED_STOCK_FULL_VERIFY.log"
grep -q 'STOCK_FA3_SERVING path=stock_fa3_calls' "$gate/server_MTPMIXFUSED_STOCK_FULL_VERIFY.log"
grep -q 'sm_split=132' "$gate/server_MTPMIXFUSED_STOCK_FULL_VERIFY.log"
echo "STOCK_FULL_GATE_PASS $(date -Is)"
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_SNAPSHOT
variants=(
  'MTP|MTP_LPT|0|0|2|112'
  'UNFUSED4|MTPMIX_LPT|0|0|2|112'
  'SPLIT_1|MTPMIX_SPLIT|0|0|2|112'
  'NATIVE_RELEASE|MTPMIXFUSED|1|0|2|112'
  'STOCK_SERIAL112|MTPMIXFUSED|1|1|0|112'
  'STOCK_PDL112|MTPMIXFUSED|1|1|2|112'
  'STOCK_SERIAL132|MTPMIXFUSED|1|1|0|132'
  'STOCK_PDL132|MTPMIXFUSED|1|1|2|132'
  'SPLIT_2|MTPMIX_SPLIT|0|0|2|112'
)
for variant in "${variants[@]}"; do
  IFS='|' read -r label method native stock phase budget <<< "$variant"
  export SGLANG_FUSED_NATIVE_PDL=$native SGLANG_STOCK_FA3_PDL=$stock
  export SGLANG_STOCK_FA3_PHASE=$phase SGLANG_FUSED_SM_FA3=$budget
  output=$root/$label
  mkdir "$output"
  echo "STOCK_SERVING_START $label method=$method budget=$budget $(date -Is)"
  timeout --kill-after=30 5400 "$workdir/venv/bin/python" -u drivers/kernel_concurrency_driver.py \
    --scenario kernel_concurrency_env --methods "$method" --outdir "$output" > "$output/driver.log" 2>&1
  compare=()
  if [ "$label" != MTP ]; then compare=(--reference "$root/MTP"); fi
  "$workdir/venv/bin/python" "$overlay/audit_stock_serving.py" "$output" "$method" \
    --stock "$stock" --phase "$phase" --budget "$budget" "${compare[@]}" > "$output/audit.log" 2>&1
  echo "STOCK_SERVING_DONE $label $(date -Is)"
done
echo "STOCK_FA3_SERVING_AB_DONE $(date -Is)"
