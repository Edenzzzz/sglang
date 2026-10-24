#!/bin/bash
# Stage 1, split-policy comparison for the unified dispatcher (A6): on the 96 trace-sampled captures, the FA3
# baselines and native v3 once, then the unified arm under each KV-split policy (flashinfer's persistent-LPT
# kv_len_limit on both executors / makespan search / the first fixed rules). 2-class FA2 build (32 / 16),
# rows > 32 packed on FA3. H200 34c7bd503d461dc1. Claude, 2026-10-24.
set -u
cd /dev/shm/w/ub
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified2_20261024/unified_dispatch_ext/unified_dispatch_ext.so UNIFIED_FA3_MIN=33 UNIFIED_FA2_CPS=2
OUT=${OUT:-/opt/tiger/w/unified_policy_h200_20261024}; mkdir -p $OUT
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || { echo "GPU busy"; exit 1; }
for d in mix32_two mix32_four mix128_two mix128_four; do
  mkdir -p $OUT/$d
  first=1
  for pol in flashinfer dynamic fixed; do
    extra="--unified-only --append"; [ $first = 1 ] && extra=""
    UNIFIED_SPLIT_POLICY=$pol /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py \
      /opt/tiger/w/trace_sampled_h200_20261023/$d/captures $OUT/$d/kernel_results.json --label unified_$pol $extra 2>&1 \
      | grep -E "UNIFIED_BENCH|Error|error|Traceback"
    first=0
  done
done
for arm in unified_flashinfer unified_dynamic unified_fixed; do
  for b in default fixed best arm:native_pdl_geometry_v3 arm:fa3_dual_split32; do
    /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py $arm --baseline $b $OUT/*/kernel_results.json
  done
done
echo UNIFIED_POLICY_DONE
