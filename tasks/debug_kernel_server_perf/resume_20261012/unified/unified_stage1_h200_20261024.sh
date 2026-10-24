#!/bin/bash
# Stage 1 for the unified tile dispatcher (A6): the 96 trace-sampled captures (24 per context x mode),
# unified vs every FA3 baseline and native v3. H200 34c7bd503d461dc1. Claude, 2026-10-24.
set -u
cd /dev/shm/w/ub
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
OUT=${OUT:-/opt/tiger/w/unified_stage1_h200_20261024}; mkdir -p $OUT
export UNIFIED_SO UNIFIED_FA3_MIN UNIFIED_FA2_CPS
for d in mix32_two mix32_four mix128_two mix128_four; do
  mkdir -p $OUT/$d; /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py /opt/tiger/w/trace_sampled_h200_20261023/$d/captures $OUT/$d/kernel_results.json 2>&1 | grep -E "UNIFIED_BENCH|Error|error|Traceback"
done
for b in default fixed best arm:native_pdl_geometry_v3 arm:fa3_dual_split32; do
  /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py unified --baseline $b $OUT/*/kernel_results.json
done
echo UNIFIED_STAGE1_DONE
