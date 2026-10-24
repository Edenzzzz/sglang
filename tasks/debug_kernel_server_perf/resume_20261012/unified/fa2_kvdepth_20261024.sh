#!/bin/bash
# Per-class FA2 KV depth (A6): 16-row tile MMA_KV in {2, 4, 8} with the 32-row tile at 2, on the FA2 rows of all four
# sets, vs stock BatchAttention and BatchPrefill. Claude, 2026-10-24.
cd /dev/shm/w/ub
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_SPLIT_POLICY=fixed UNIFIED_MERGE_ROWS=4
B=/opt/tiger/w/trace_sampled_h200_20261023
for v in base2 kv16_4 kv16_8; do
  UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$v/unified_dispatch_ext/unified_dispatch_ext.so \
    /dev/shm/w/venv/bin/python -u fa2_rows_compare_20261024.py $B/mix32_two/captures $B/mix128_two/captures 2>&1 | grep -E "TOTAL|rror" | sed "s/^/$v two-mode /"
  UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$v/unified_dispatch_ext/unified_dispatch_ext.so \
    /dev/shm/w/venv/bin/python -u fa2_rows_compare_20261024.py $B/mix32_four/captures $B/mix128_four/captures 2>&1 | grep -E "TOTAL|rror" | sed "s/^/$v four-mode /"
done
echo KVDEPTH_DONE
