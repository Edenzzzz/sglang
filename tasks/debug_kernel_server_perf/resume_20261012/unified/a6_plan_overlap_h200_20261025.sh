#!/bin/bash
# A6 plan on device metadata (no blocking host-to-device copies, double-buffered pinned staging), H200: plan host time
# with ~20 steps queued, host-only vs device-metadata plan, same-output check; FP32 + 20 graph replays on every capture
# with the new executor. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
grep -q pinned_done unified_executor_20261024.py || { echo STALE; exit 1; }
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_SO=/opt/tiger/w/torch_ext_unified_h200_np_u/unified_dispatch_ext/unified_dispatch_ext.so
B=/opt/tiger/w/trace_sampled_h200_20261025
/dev/shm/w/venv/bin/python -u plan_overlap_20261025.py $B/mix32_two/captures $B/mix32_four/captures $B/mix128_two/captures $B/mix128_four/captures > plan_overlap.log 2>&1
grep -E "TOTAL|Error|differs" plan_overlap.log | tail -3
for d in mix32_two mix32_four mix128_two mix128_four; do
  UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_overlap_$d.log 2>&1
  echo "check $d: FAIL $(grep -v GRAPH_FAILS check_overlap_$d.log | grep -c FAIL), PASS $(grep -c ' PASS' check_overlap_$d.log)"
done
echo PLAN_OVERLAP_DONE
