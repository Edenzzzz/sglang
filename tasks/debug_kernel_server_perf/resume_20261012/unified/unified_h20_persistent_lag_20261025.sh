#!/bin/bash
# Persistent overlap + LPT: merge items interleaved L groups behind their compute (UNIFIED_MERGE_LAG), FA2 rows vs v3's FA2 launch (A6, H20). The
# compute-only bound is 4.72 / 6.19 ms (two / four); with 4-row items the persistent merge costs 1.10 / 0.73 ms against
# the v3-body executor's 0.20 / 0.25 ms. Claude, 2026-10-25.
cd /dev/shm/w/ub
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry UNIFIED_SO=/opt/tiger/w/torch_ext_unified_pers_stock/unified_dispatch_ext/unified_dispatch_ext.so
B=/opt/tiger/w/trace_sampled_h20_20261024
for lag in 0 4 16 64; do rows=4
  for t in "two:mix32_two mix128_two" "four:mix32_four mix128_four"; do
    dirs=""; for s in ${t#*:}; do dirs="$dirs $B/$s/captures"; done
    UNIFIED_MERGE_ROWS=4 UNIFIED_MERGE_LAG=$lag /dev/shm/w/venv/bin/python -u fa2_rows_compare_20261024.py $dirs > fa2rows_pers_lag${lag}_${t%%:*}.log 2>&1
    echo "lag $lag ${t%%:*} $(grep -E 'TOTAL|Error' fa2rows_pers_lag${lag}_${t%%:*}.log | tail -1 | sed 's/batchprefill=[0-9.]*ms //' | cut -c1-110)"
  done
done
echo LAG_DONE
