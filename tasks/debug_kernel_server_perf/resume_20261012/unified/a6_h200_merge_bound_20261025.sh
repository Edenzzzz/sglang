#!/bin/bash
# Full-step merge-free bound (A6, H200, 96 captures): the NP executor with no group check-ins (fence + counter atomic)
# and no merge items (UNIFIED_NP_NOSYNC=1, output wrong for split rows; timing bound only) vs the same executor with the
# unified merge -- everything the merge and its global-memory synchronization cost in the step. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
grep -q UNIFIED_NP_NOSYNC port07_unified/tests/unified_dispatch_ext.cu || { echo STALE; exit 1; }
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
for v in sync nosync; do
  n=0; [ $v = nosync ] && n=1
  rm -rf /opt/tiger/w/torch_ext_unified_h200_np_$v/unified_dispatch_ext
  UNIFIED_NP_NOSYNC=$n UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_h200_np_$v \
    /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_np_$v.log 2>&1
  grep -q UNIFIED_BUILD_OK build_np_$v.log || { grep " error" build_np_$v.log | head; echo BUILD_FAILED; exit 1; }
  grep nvcc build_np_$v.log | grep -q -- "-DUNIFIED_NP_NOSYNC=$n" || { echo FLAG_MISSING; exit 1; }
done
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3
B=/opt/tiger/w/trace_sampled_h200_20261025; OUT=/opt/tiger/w/unified_h200_20261025
for v in sync nosync; do
  b=0; [ $v = nosync ] && b=1
  for d in mix32_two mix32_four mix128_two mix128_four; do
    UNIFIED_SO=/opt/tiger/w/torch_ext_unified_h200_np_$v/unified_dispatch_ext/unified_dispatch_ext.so UNIFIED_TIMING_BOUND=$b \
      /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json --unified-only --append --label u_np_$v 2>&1 | grep -E "FAIL|rror|Traceback"
  done
done
echo "merge-free bound vs with the unified merge (speedup = everything the merge + its sync cost):"
/dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_np_nosync --baseline arm:u_np_sync $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
echo MBOUND_DONE
