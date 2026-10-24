#!/bin/bash
# Persistent overlap + LPT with the ready-queue merge (A6, H20): the member that completes a group publishes its merge
# items; CTAs drain published merges before pulling more compute (LPT order kept). FP32 + 20 graph replays on all 92,
# FA2 rows x3 vs v3's FA2 launch, stage 1 under geometry. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
name=pers_ready
grep -q "Ready-queue merge item" port07_unified/tests/unified_dispatch_ext.cu || { echo STALE; exit 1; }
rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
UNIFIED_FA2_NP=0 UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name \
  /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
grep -q UNIFIED_BUILD_OK build_$name.log || { grep -E " error" build_$name.log | head; echo BUILD_FAILED; exit 1; }
for w in UNIFIED_FA2_NP=0 UNIFIED_FA3_STOCK=1; do grep -q -- "-D$w" build_$name.log || { echo FLAG_MISSING $w; exit 1; }; done
echo "build ok"
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry UNIFIED_MERGE_MODE=ready UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
B=/opt/tiger/w/trace_sampled_h20_20261024; OUT=/opt/tiger/w/unified_h20_20261024
for d in mix32_two mix32_four mix128_two mix128_four; do
  UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 timeout 1200 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_pready_$d.log 2>&1
  echo "check $d: rc=$? FAIL $(grep -v GRAPH_FAILS check_pready_$d.log | grep -c FAIL), PASS $(grep -c ' PASS' check_pready_$d.log), graph fails $(grep GRAPH_FAILS check_pready_$d.log | awk '{s+=$2} END {print s+0}'), err $(grep -cE 'Error|Traceback' check_pready_$d.log)"
done
for rep in 1 2 3; do
  for t in "two:mix32_two mix128_two" "four:mix32_four mix128_four"; do
    dirs=""; for s in ${t#*:}; do dirs="$dirs $B/$s/captures"; done
    timeout 1200 /dev/shm/w/venv/bin/python -u fa2_rows_compare_20261024.py $dirs > fa2rows_pready_${rep}_${t%%:*}.log 2>&1
    echo "rep $rep ${t%%:*} $(grep -E 'TOTAL|Error' fa2rows_pready_${rep}_${t%%:*}.log | tail -1 | sed 's/batchprefill=[0-9.]*ms //' | cut -c1-120)"
  done
done
for d in mix32_two mix32_four mix128_two mix128_four; do
  timeout 1800 /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json --unified-only --append --label u_pready_geometry 2>&1 | grep -E "FAIL|rror|Traceback"
done
/dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_pready_geometry --baseline arm:native_pdl_geometry_v3 $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
echo PREADY_DONE
