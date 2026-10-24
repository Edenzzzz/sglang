#!/bin/bash
# Stage 1 with plan-scheduled merge items (A6): the 96 trace-sampled captures, unified (fixed split policy, merge
# items of 4 and 2 rows) appended beside the baselines already measured in unified_policy_v2. Claude, 2026-10-24.
cd /dev/shm/w/ub
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_base2/unified_dispatch_ext/unified_dispatch_ext.so UNIFIED_SPLIT_POLICY=fixed UNIFIED_FA3_MIN=33
SRC=/opt/tiger/w/unified_policy_v2_h200_20261024; OUT=/opt/tiger/w/unified_mi_h200_20261024; mkdir -p $OUT
for d in mix32_two mix32_four mix128_two mix128_four; do
  mkdir -p $OUT/$d; cp $SRC/$d/kernel_results.json $OUT/$d/kernel_results.json
  for r in 4 2; do
    UNIFIED_MERGE_ROWS=$r /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py /opt/tiger/w/trace_sampled_h200_20261023/$d/captures \
      $OUT/$d/kernel_results.json --unified-only --append --label unified_mi$r 2>&1 | grep -E "FAIL|rror|Traceback"
  done
done
for arm in unified_mi4 unified_mi2 unified_fixed; do
  for b in default arm:native_pdl_geometry_v3 arm:fa3_dual_split32; do
    /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py $arm --baseline $b $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
  done
done
echo MI_STAGE1_DONE
