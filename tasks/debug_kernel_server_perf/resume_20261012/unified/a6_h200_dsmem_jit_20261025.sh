#!/bin/bash
# DSMEM-cluster JIT reduction (A6, H200, 96 captures): UNIFIED_MERGE_MODE=dsmem -- the FA2 launch runs in clusters of C,
# each cluster one plan task holding whole split groups (chunks capped at C), one hardware cluster barrier, then the
# group's CTAs reduce its rows; no global fences / counter atomics / spin-waits. C = 2 / 4 / 8 / 16 (16 non-portable).
# FP32 + 20 graph replays per C, stage 1 vs the ticketed merge items (u_h200_np_u) and native v3. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
grep -q UnifiedFA2NPClusterKernel port07_unified/tests/unified_dispatch_ext.cu || { echo STALE; exit 1; }
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
name=h200_np_dsmem
rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name \
  /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED; exit 1; }
echo "built $name"
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_MERGE_MODE=dsmem
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
B=/opt/tiger/w/trace_sampled_h200_20261025; OUT=/opt/tiger/w/unified_h200_20261025
SETS="mix32_two mix32_four mix128_two mix128_four"
for c in ${DSMEM_CLUSTERS:-2 4 8 16}; do
  export UNIFIED_DSMEM_CLUSTER=$c
  ok=1
  for d in $SETS; do
    UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 timeout 1200 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_dsmem_c${c}_$d.log 2>&1
    rc=$?
    f=$(grep -v GRAPH_FAILS check_dsmem_c${c}_$d.log | grep -c FAIL); p=$(grep -c ' PASS' check_dsmem_c${c}_$d.log)
    echo "check C=$c $d rc=$rc: FAIL $f, PASS $p $(grep -m1 -E 'Error|error' check_dsmem_c${c}_$d.log | cut -c1-160)"
    [ $rc != 0 ] || [ $f != 0 ] && ok=0
  done
  [ $ok = 1 ] || { echo "C=$c not exact, skipping its timing"; continue; }
  for d in $SETS; do
    /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json --unified-only --append --label u_dsmem_c$c 2>&1 | grep -E "FAIL|rror|Traceback"
  done
  echo "stage1 C=$c vs ticketed merge items:"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_dsmem_c$c --baseline arm:u_h200_np_u $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
  echo "stage1 C=$c vs native v3:"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_dsmem_c$c --baseline arm:native_pdl_geometry_v3 $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
done
echo DSMEM_JIT_DONE
