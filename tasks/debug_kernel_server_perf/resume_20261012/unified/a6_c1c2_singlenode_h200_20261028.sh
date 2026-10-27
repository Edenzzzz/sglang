#!/bin/bash
# Single-node attribution of C1 (A6 kernel) and C2 (four-mode step) on one H200, 2026-10-28.
# The cluster replay shows +25% agent goodput and -45% TPOT over stock colocated sglang at an EQUAL prefix-cache hit
# rate (91.8% vs 92.7%), so the gain is in execution, not in cache. C1 is only 4-9% of attention time on captures and
# 1.04x of whole-step time in serving, so C2 (verify rows merged into the prefill chunk's forward) is the suspect.
# This separates them on one GPU, where there is no router, no pool and no cross-node effect to confound:
#   --modes four  the four-mode step   --modes two  stock MTP steps (no C2)
# and each mode runs our A6 arm against the stock FA3 reference arm the harness launches, giving C1 within each mode.
# Both context stages, two repeats, opposite order, composition counter checked (memory: fused A/B validity rule).
# Claude.
set -euo pipefail
SRC=/dev/shm/w/qwen25_v3_common_screen_h200_20261023
BASE=/opt/tiger/w/a6_c1c2_singlenode_20261028
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
mkdir -p "$BASE"
R=$BASE/root
mkdir -p "$R"
test -d "$R/python" || cp -a "$SRC/python" "$R/python"
cp "$SRC"/*.py "$SRC"/*.sh "$SRC"/tracelab_unique.jsonl.gz "$R/"
cp /dev/shm/w/ub/run_native_serving_pair_20261023.py /dev/shm/w/ub/native_serving_protocol_20261023.py \
   /dev/shm/w/ub/audit_native_serving_pair_20261023.py /dev/shm/w/ub/native_fa3_fa2_pdl_20261023.py \
   /dev/shm/w/ub/unified_executor_20261024.py "$R/"
/dev/shm/w/venv/bin/python /dev/shm/w/perf/rss/apply_edf_late_demote_20261024.py "$R"
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_np64_h200x4/unified_dispatch_ext/unified_dispatch_ext.so
[ -f "$UNIFIED_SO" ] || UNIFIED_SO=$(ls -t /opt/tiger/w/torch_ext_unified_*/unified_dispatch_ext/unified_dispatch_ext.so | head -1)
export UNIFIED_SO
export UNIFIED_FA3_STOCK=1 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_MIN=65
echo "UNIFIED_SO=$UNIFIED_SO"
COMMON="--split-policy geometry_v3 --direct-output --packed-query-limit 64 --repeats 2 --ours"
FIX="--ours-env SGLANG_TTFT_SLO_MS=500 --ours-env SGLANG_EDF_LONG_PROMPT_TOKENS=8192 --ours-env SGLANG_EDF_LONG_TTFT_SLO_MS=10000"
export PYTHONPATH=$R/python:$R:/dev/shm/w/fi:/dev/shm/w/drivers/serving TRACE_DATASET_PATH=$R/tracelab_unique.jsonl.gz
cd "$R"
for stage in mix32 mix128; do
  echo "##### stage $stage"
  /dev/shm/w/venv/bin/python -u run_native_serving_pair_20261023.py "$R" --stage $stage --modes four two \
    $COMMON $FIX --native-env SGLANG_A6_UNIFIED=1 2>&1 | grep -E "NATIVE_SERVING_ARM_DONE|Error|Traceback" | cut -c1-900
  n=$(cat $R/${stage}_*_native/server.log 2>/dev/null | grep -c "a6_unified=1" || true)
  echo "a6 composition ($stage): $n"; [ "$n" -gt 0 ] || { echo A6_NOT_ACTIVE; exit 1; }
  /dev/shm/w/venv/bin/python -u audit_native_serving_pair_20261023.py "$R" 2>&1 | tail -14 || true
done
echo C1C2_SINGLENODE_DONE
