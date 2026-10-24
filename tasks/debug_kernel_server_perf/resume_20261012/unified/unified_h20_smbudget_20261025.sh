#!/bin/bash
# SM pre-assignment ceiling (A6, H20 78 SMs): stage 1 with the race-fixed executor (np2s, stock FA3, geometry) at fixed
# FA3 SM budgets; the per-capture best budget is the oracle a per-step cost model could reach. Claude, 2026-10-25.
cd /dev/shm/w/ub
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry UNIFIED_SO=/opt/tiger/w/torch_ext_unified_np2s_stock/unified_dispatch_ext/unified_dispatch_ext.so
B=/opt/tiger/w/trace_sampled_h20_20261024; OUT=/opt/tiger/w/unified_h20_20261024
for sm in 70 62 54 46; do
  label=u_np2s_sm$sm
  for d in mix32_two mix32_four mix128_two mix128_four; do
    UNIFIED_FA3_SM=$sm /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json \
      --unified-only --append --label $label 2>&1 | grep -E "FAIL|rror|Traceback"
  done
  /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py $label --baseline arm:native_pdl_geometry_v3 $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
done
echo SMB_DONE
