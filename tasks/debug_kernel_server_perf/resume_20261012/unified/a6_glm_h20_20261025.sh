#!/bin/bash
# Stage 1 for GLM-4.5-Air (TP4, H20 pod 0, rank-0 attention; 24 q / 2 kv heads per rank, verify rows pack 48 -> FA3): capture 32-64K two/four-mode
# steps from real 4-mode MTP serving, re-bench native v3, then A6 (np2s_stock, geometry_fa3): FP32 + graph check and
# stage 1 vs v3 / sglang FA3. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
bash trace_sampled_capture_glm_h20_20261025.sh > /opt/tiger/w/trace_sampled_glm_h20_20261025.log 2>&1
grep -q TRACE_SAMPLED_CAPTURE_DONE /opt/tiger/w/trace_sampled_glm_h20_20261025.log || { echo CAPTURE_FAILED; tail -20 /opt/tiger/w/trace_sampled_glm_h20_20261025.log; exit 1; }
B=/opt/tiger/w/trace_sampled_glm_h20_20261025
echo "captures $(ls $B/*/captures/*.pt | wc -l)"
bash rebench_trace_sampled_v3_glm_h20_20261025.sh > rebench_v3_glm.log 2>&1
grep -E "time-weighted|REBENCH" rebench_v3_glm.log | sed -E 's/ \|.*//'
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_SO=/opt/tiger/w/torch_ext_unified_np_u/unified_dispatch_ext/unified_dispatch_ext.so
OUT=/opt/tiger/w/unified_glm_h20_20261025
for d in mix32_two mix32_four; do
  mkdir -p $OUT/$d; cp $B/$d/kernel_results_v3.json $OUT/$d/kernel_results.json
  UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_glm_$d.log 2>&1
  echo "check $d: FAIL $(grep -v GRAPH_FAILS check_glm_$d.log | grep -c FAIL), PASS $(grep -c ' PASS' check_glm_$d.log), graph fails $(grep GRAPH_FAILS check_glm_$d.log | awk '{s+=$2} END {print s+0}'), err $(grep -cE 'Error|Traceback' check_glm_$d.log)"
  /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json --unified-only --append --label u_np_u_geometry_fa3 2>&1 | grep -E "FAIL|rror|Traceback"
done
for b in arm:native_pdl_geometry_v3 default arm:fa3_dual_split32; do
  /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_np_u_geometry_fa3 --baseline $b $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
done
echo A6_GLM_DONE
