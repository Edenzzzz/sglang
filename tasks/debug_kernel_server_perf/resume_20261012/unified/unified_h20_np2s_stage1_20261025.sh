#!/bin/bash
# Stage 1 on H20 with the race-fixed executor (np2s: v3 body, chunk-parallel ticketed merge items, stock FA3 with the
# plan's split, no early PDL trigger), under geometry and geometry_fa3, vs native v3 / sglang FA3. Claude, 2026-10-25.
cd /dev/shm/w/ub
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SO=/opt/tiger/w/torch_ext_unified_np2s_stock/unified_dispatch_ext/unified_dispatch_ext.so
B=/opt/tiger/w/trace_sampled_h20_20261024; OUT=/opt/tiger/w/unified_h20_20261024
for pol in geometry geometry_fa3; do
  label=u_np2s_stock_$pol
  for d in mix32_two mix32_four mix128_two mix128_four; do
    UNIFIED_SPLIT_POLICY=$pol /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json \
      --unified-only --append --label $label 2>&1 | grep -E "FAIL|rror|Traceback"
  done
  for b in arm:native_pdl_geometry_v3 default arm:fa3_dual_split32; do
    /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py $label --baseline $b $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
  done
done
echo NP2S_S1_DONE
