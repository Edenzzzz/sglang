#!/bin/bash
# Gate of the deployed serving A6 library (bc43aedeb9ed) on the Llama-3.3-70B trace-sampled captures (pod 0): FP32 + 20
# graph replays with FA3_MIN=65, then stage 1 vs native v3. Claude, 2026-10-26.
set -u
cd /dev/shm/w/ub
export PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi CUDA_VISIBLE_DEVICES=0
A=/opt/tiger/w/a6lib
export NATIVE_PDL_FA2_LIBRARY=$A/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=$A/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=$A/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=$A/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=65 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_SO=$A/unified_dispatch_ext.so
echo "lib $(md5sum $UNIFIED_SO | cut -c1-12)"
B=/opt/tiger/w/trace_sampled_llama70b_h20_20261025; OUT=/opt/tiger/w/srv_gate_llama_20261026; mkdir -p $OUT
for d in $(cd $B && ls -d */ | tr -d /); do
  [ -d $B/$d/captures ] || continue
  UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 timeout 3600 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_srv_llama_$d.log 2>&1
  echo "check Llama $d rc=$?: FAIL $(grep -v GRAPH_FAILS check_srv_llama_$d.log | grep -c FAIL), PASS $(grep -c ' PASS' check_srv_llama_$d.log)"
  mkdir -p $OUT/$d
  /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json --label u_srv 2>&1 | grep -E "FAIL|rror|Traceback" | head -3
done
echo "Llama stage 1, deployed lib vs native v3:"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_srv --baseline arm:native_pdl_geometry_v3 $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
echo LLAMA_GATE_DONE
