#!/bin/bash
# Plan-scheduled merge items vs last-arriver JIT merge (A6), FA2 rows vs stock BatchAttention; then a correctness
# sweep of the full unified path (graph replays) with merge items on. Claude, 2026-10-24.
cd /dev/shm/w/ub
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_base2/unified_dispatch_ext/unified_dispatch_ext.so UNIFIED_SPLIT_POLICY=fixed
B=/opt/tiger/w/trace_sampled_h200_20261023
for v in "items 8" "items 4" "items 16" "jit 8"; do
  set -- $v
  UNIFIED_MERGE_MODE=$1 UNIFIED_MERGE_ROWS=$2 /dev/shm/w/venv/bin/python -u fa2_rows_compare_20261024.py $B/mix128_two/captures $B/mix32_four/captures 2>&1 | grep -E "TOTAL|rror" | sed "s/^/$1 rows=$2 /"
done
UNIFIED_FA3_MIN=33 UNIFIED_GRAPH=1 UNIFIED_REPLAYS=50 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/mix32_four/captures/*.pt $B/mix128_two/captures/*.pt 2>&1 | grep -cE "PASS" | sed 's/^/PASS count: /'
UNIFIED_FA3_MIN=33 UNIFIED_GRAPH=1 UNIFIED_REPLAYS=50 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/mix32_four/captures/*.pt $B/mix128_two/captures/*.pt 2>&1 | grep -E "FAIL|GRAPH replay|rror" | head -5
echo MERGEITEMS_DONE
