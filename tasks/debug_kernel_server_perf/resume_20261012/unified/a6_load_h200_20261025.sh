#!/bin/bash
# Higher agent load for the throughput bar (H200, 2026-10-25), same stack as the A6 stage 2: our full single-node stack (EDF with per-tenant deadlines,
# demote-on-miss, 4-mode MTP step) vs stock sglang on FA3, once with native v3 as the kernel and once with the unified
# dispatcher (SGLANG_A6_UNIFIED=1: one plan, stock FA3 at the plan's split, v3's FA2 body with the in-launch merge).
# Same workloads as ours_full_stack_v3_h200_20261024.sh (long 128-256K agent conc 4 / chat 4 req/s; short 32-64K agent
# conc 8 / chat 8 req/s), two opposite-order repeats, audited; the A6 arm must log a6_unified=1. Claude.
set -euo pipefail
SRC=/dev/shm/w/qwen25_v3_common_screen_h200_20261023
BASE=/opt/tiger/w/a6_load_h200_20261025
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
mkdir -p "$BASE"
make_root() {
  local R=$1
  mkdir -p "$R"
  test -d "$R/python" || cp -a "$SRC/python" "$R/python"
  cp "$SRC"/*.py "$SRC"/*.sh "$SRC"/tracelab_unique.jsonl.gz "$R/"
  cp /dev/shm/w/ub/native_fa3_fa2_pdl_20261023.py /dev/shm/w/ub/run_native_serving_pair_20261023.py \
     /dev/shm/w/ub/native_serving_protocol_20261023.py /dev/shm/w/ub/audit_native_serving_pair_20261023.py \
     /dev/shm/w/ub/unified_executor_20261024.py "$R/"
  /dev/shm/w/venv/bin/python /dev/shm/w/perf/rss/apply_edf_late_demote_20261024.py "$R"
  grep -q SGLANG_EDF_LATE_DEMOTE "$R/python/sglang/srt/managers/scheduler.py"
  grep -q prepare_unified "$R/native_fa3_fa2_pdl_20261023.py"
  grep -q native_env "$R/run_native_serving_pair_20261023.py"
}
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_np2s_stock/unified_dispatch_ext/unified_dispatch_ext.so
export UNIFIED_FA3_STOCK=1 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_MIN=33
COMMON="--split-policy geometry_v3 --direct-output --packed-query-limit 64 --repeats 2 --ours"
FIX="--ours-env SGLANG_TTFT_SLO_MS=500 --ours-env SGLANG_EDF_LONG_PROMPT_TOKENS=8192 --ours-env SGLANG_EDF_LONG_TTFT_SLO_MS=10000 --ours-env SGLANG_EDF_LATE_DEMOTE=1 --ours-env SGLANG_EDF_LATE_DEMOTE_TOK_S=1e12"
for kernel in a6; do
  KENV=""; [ $kernel = a6 ] && KENV="--native-env SGLANG_A6_UNIFIED=1"
  for spec in "c8 mix128 --concurrency 8 --sessions 16 --chat-rate 4 --chat-requests 960" "c16 mix128 --concurrency 16 --sessions 32 --chat-rate 4 --chat-requests 960"; do
    set -- $spec; name=$1; stage=$2; shift 2
    [ -z "${ONLY:-}" ] || [ "$ONLY" = "${name}_$kernel" ] || continue
    R=$BASE/${name}_$kernel
    make_root "$R"
    export PYTHONPATH=$R/python:$R:/dev/shm/w/fi:/dev/shm/w/drivers/serving TRACE_DATASET_PATH=$R/tracelab_unique.jsonl.gz
    cd "$R"
    echo "##### $name $stage kernel=$kernel"
    /dev/shm/w/venv/bin/python -u run_native_serving_pair_20261023.py "$R" --stage $stage --modes four "$@" $COMMON $FIX $KENV
    /dev/shm/w/venv/bin/python audit_native_serving_pair_20261023.py "$R" "$R/audit.json"
    if [ $kernel = a6 ]; then
      n=$(cat $R/*_native/server.log 2>/dev/null | grep -c "a6_unified=1" || true)
      echo "a6 composition: $n NATIVE_PDL_CALL a6_unified=1 lines"; [ "$n" -gt 0 ] || { echo A6_NOT_ACTIVE; exit 1; }
    fi
    /dev/shm/w/venv/bin/python /dev/shm/w/perf/bench_per_tenant.py ${stage}_*/requests || true
  done
done
echo A6_LOAD_DONE
