#!/bin/bash
# Serving A/B of the A6 plan path (H200 job 8574c727fab6ffbd): the old plan (five blocking host-to-device copies per
# mixed step, each draining the stream, so the overlap scheduler's CPU work waits for the previous step) vs the new one
# (device metadata, double-buffered pinned staging). Same full single-node stack and workload as stage 2's short set
# (mix32, agent C=8 + chat 8 req/s, four-mode), two opposite-order repeats per root; each root also runs the stock FA3
# arm as a common reference. Claude, 2026-10-25.
set -euo pipefail
SRC=/dev/shm/w/qwen25_v3_common_screen_h200_20261023
BASE=/opt/tiger/w/a6_plan_ab_h200_20261025
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
mkdir -p "$BASE"
grep -q pinned_done /dev/shm/w/ub/unified_executor_20261024.py
! grep -q pinned_done /dev/shm/w/ub/planold/unified_executor_20261024.py
make_root() {  # root, dir holding that version's native_fa3_fa2_pdl / unified_executor
  local R=$1 V=$2
  mkdir -p "$R"
  test -d "$R/python" || cp -a "$SRC/python" "$R/python"
  cp "$SRC"/*.py "$SRC"/*.sh "$SRC"/tracelab_unique.jsonl.gz "$R/"
  cp /dev/shm/w/ub/run_native_serving_pair_20261023.py /dev/shm/w/ub/native_serving_protocol_20261023.py \
     /dev/shm/w/ub/audit_native_serving_pair_20261023.py "$R/"
  cp "$V/native_fa3_fa2_pdl_20261023.py" "$V/unified_executor_20261024.py" "$R/"
  /dev/shm/w/venv/bin/python /dev/shm/w/perf/rss/apply_edf_late_demote_20261024.py "$R"
  grep -q SGLANG_EDF_LATE_DEMOTE "$R/python/sglang/srt/managers/scheduler.py"
}
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_h200_np_u/unified_dispatch_ext/unified_dispatch_ext.so
export UNIFIED_FA3_STOCK=1 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_MIN=33
COMMON="--split-policy geometry_v3 --direct-output --packed-query-limit 64 --repeats 2 --ours"
FIX="--ours-env SGLANG_TTFT_SLO_MS=500 --ours-env SGLANG_EDF_LONG_PROMPT_TOKENS=8192 --ours-env SGLANG_EDF_LONG_TTFT_SLO_MS=10000 --ours-env SGLANG_EDF_LATE_DEMOTE=1 --ours-env SGLANG_EDF_LATE_DEMOTE_TOK_S=1e12"
for v in new old; do
  V=/dev/shm/w/ub; [ $v = old ] && V=/dev/shm/w/ub/planold
  R=$BASE/plan_$v
  make_root "$R" "$V"
  export PYTHONPATH=$R/python:$R:/dev/shm/w/fi:/dev/shm/w/drivers/serving TRACE_DATASET_PATH=$R/tracelab_unique.jsonl.gz
  cd "$R"
  echo "##### plan $v"
  /dev/shm/w/venv/bin/python -u run_native_serving_pair_20261023.py "$R" --stage mix32 --modes four $COMMON $FIX --native-env SGLANG_A6_UNIFIED=1 2>&1 | grep -E "NATIVE_SERVING_ARM_DONE|Error|Traceback" | cut -c1-900
  n=$(cat $R/*_native/server.log 2>/dev/null | grep -c "a6_unified=1" || true)
  echo "a6 composition: $n"; [ "$n" -gt 0 ] || { echo A6_NOT_ACTIVE; exit 1; }
  /dev/shm/w/venv/bin/python /dev/shm/w/perf/bench_per_tenant.py mix32_*/requests 2>&1 | tail -12 || true
done
echo PLAN_AB_DONE
