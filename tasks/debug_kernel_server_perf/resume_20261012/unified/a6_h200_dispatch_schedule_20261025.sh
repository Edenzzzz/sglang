#!/bin/bash
# Separating dispatch, tile code and schedule (A6, H200 job 8574c727fab6ffbd, 96 trace-sampled captures, geometry_fa3):
#   NP     v3's BatchPrefill body, one hardware-launched CTA per queue ticket          (u_h200_np_u, earlier run)
#   NPP    the same body and merge, a persistent grid pulling the same LPT queue        (UNIFIED_NP_PERSIST=1)
#   PERS   flashinfer's persistent runner pulling the same queue                       (u_h200_pers_dual, earlier run)
# and the schedule itself, on the NP executor: LPT order vs enumeration order (UNIFIED_FA2_ORDER=fifo), and the FA2
# launch overlapped with FA3 by PDL vs serial (UNIFIED_PDL=0). FP32 + 20 graph replays on the new build; stage 1 for
# every arm against native v3 and against NP. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
grep -q "UNIFIED_NP_PERSIST" port07_unified/tests/unified_dispatch_ext.cu || { echo STALE; exit 1; }
grep -q "UNIFIED_FA2_ORDER" port07_unified/include/flashinfer/attention/fused_plan.cuh || { echo STALE; exit 1; }
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
build() {  # name, flags...
  local name=$1; shift
  rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
  env "$@" UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
  grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED $name; exit 1; }
  for w in "$@" UNIFIED_FA3_STOCK=1; do grep nvcc build_$name.log | grep -q -- "-D$w" || { echo FLAG_MISSING $name $w; exit 1; }; done
  echo "built $name"
}
build h200_npp UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_NP_PERSIST=1
build h200_np_o UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_PRINT_OCC=1
unset UNIFIED_MERGE_MODE UNIFIED_FA2_ORDER UNIFIED_PDL
B=/opt/tiger/w/trace_sampled_h200_20261025; OUT=/opt/tiger/w/unified_h200_20261025
SETS="mix32_two mix32_four mix128_two mix128_four"
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_h200_npp/unified_dispatch_ext/unified_dispatch_ext.so
for d in $SETS; do
  UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 timeout 1800 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_npp.$d.log 2>&1
  echo "check npp $d: FAIL $(grep -v GRAPH_FAILS check_npp.$d.log | grep -c FAIL), PASS $(grep -c ' PASS' check_npp.$d.log), $(grep -m1 UNIFIED_NPP check_npp.$d.log)"
done
arm() {  # label, so, extra env...
  local label=$1 so=$2; shift 2
  for d in $SETS; do
    env "$@" UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$so/unified_dispatch_ext/unified_dispatch_ext.so \
      /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json \
      --unified-only --append --label $label 2>&1 | grep -E "FAIL|rror|Traceback"
  done
  for b in arm:native_pdl_geometry_v3 arm:u_h200_np_u; do
    echo "stage1 $label vs ${b#arm:}:"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py $label --baseline $b $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
  done
}
arm u_h200_npp h200_npp
arm u_h200_np_fifo h200_np_o UNIFIED_FA2_ORDER=fifo
arm u_h200_np_nopdl h200_np_o UNIFIED_PDL=0
arm u_h200_np_rerun h200_np_o
echo "persistent runner vs NPP:"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_h200_pers_dual --baseline arm:u_h200_npp $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
echo DISPATCH_SCHED_DONE
