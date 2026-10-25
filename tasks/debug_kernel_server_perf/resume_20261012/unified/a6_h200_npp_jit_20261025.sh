#!/bin/bash
# Persistent BatchPrefill loop + JIT reduction (A6, H200, 96 captures, geometry_fa3): ready mode publishes each split
# group's merge items the moment its last member completes, so merges overlap the remaining compute. vs the same loop
# with merge items queued after compute (ticket mode), and vs the hardware-launched NP executor. FP32 + 20 graph replays
# on every capture for ready mode; per-item timelines of both modes. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
grep -q "UNIFIED_NPP_READY_STUCK" port07_unified/tests/unified_dispatch_ext.cu || { echo STALE; exit 1; }
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
build() {  # name, flags...
  local name=$1; shift
  rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
  env "$@" UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
  grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED $name; exit 1; }
  for w in "$@" UNIFIED_FA3_STOCK=1; do grep nvcc build_$name.log | grep -q -- "-D$w" || { echo FLAG_MISSING $name $w; exit 1; }; done
  echo "built $name"
}
build h200_nppj UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_NP_PERSIST=1
build h200_nppj_tl UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_NP_PERSIST=1 UNIFIED_TIMELINE=1
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3
unset UNIFIED_MERGE_MODE UNIFIED_FA2_ORDER UNIFIED_PDL
B=/opt/tiger/w/trace_sampled_h200_20261025; OUT=/opt/tiger/w/unified_h200_20261025
SETS="mix32_two mix32_four mix128_two mix128_four"
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_h200_nppj/unified_dispatch_ext/unified_dispatch_ext.so
for d in $SETS; do
  UNIFIED_MERGE_MODE=ready UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 timeout 1800 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_nppj.$d.log 2>&1
  echo "check nppj ready $d: FAIL $(grep -v GRAPH_FAILS check_nppj.$d.log | grep -c FAIL), PASS $(grep -c ' PASS' check_nppj.$d.log), stuck $(grep -c STUCK check_nppj.$d.log)"
done
for mode in ticket ready; do
  for d in $SETS; do
    if [ $mode = ready ]; then export UNIFIED_MERGE_MODE=ready; else unset UNIFIED_MERGE_MODE; fi
    /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json \
      --unified-only --append --label u_h200_nppj_$mode 2>&1 | grep -E "FAIL|rror|Traceback|STUCK"
  done
  unset UNIFIED_MERGE_MODE
  for b in arm:native_pdl_geometry_v3 arm:u_h200_np_u arm:u_h200_npp; do
    echo "stage1 nppj_$mode vs ${b#arm:}:"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_h200_nppj_$mode --baseline $b $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
  done
done
echo "ready vs ticket:"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_h200_nppj_ready --baseline arm:u_h200_nppj_ticket $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_h200_nppj_tl/unified_dispatch_ext/unified_dispatch_ext.so
for mode in ticket ready; do
  for t in "two:mix32_two mix128_two" "four:mix32_four mix128_four"; do
    dirs=""; for s in ${t#*:}; do dirs="$dirs $B/$s/captures"; done
    if [ $mode = ready ]; then export UNIFIED_MERGE_MODE=ready; else unset UNIFIED_MERGE_MODE; fi
    timeout 1800 /dev/shm/w/venv/bin/python -u timeline_fa2_rows_20261025.py tl_nppj_$mode.${t%%:*}.pt $dirs > tl_nppj_$mode.${t%%:*}.log 2>&1
    echo "timeline nppj $mode ${t%%:*} $(grep -E 'TL_TOTAL|Error' tl_nppj_$mode.${t%%:*}.log | tail -1)"
  done
done
unset UNIFIED_MERGE_MODE
echo NPPJ_DONE
