#!/bin/bash
# FA3 scheduling (A6): v3's geometry_v3 prefix split for FA3 inside the unified plan (UNIFIED_SPLIT_POLICY=geometry_fa3),
# FA2 on the fixed rule, for both FA2 executors; stage 1 on all 96 captures vs native v3. Claude, 2026-10-24.
cd /dev/shm/w/ub
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8 VERBOSE=1 UNIFIED_FA2_CLASSES=2
export PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
grep -q '"geometry_fa3"' port07_unified/include/flashinfer/attention/fused_plan.cuh || { echo STALE; exit 1; }
for v in "g3_fa3items UNIFIED_FA2_NP=0" "g3_np UNIFIED_FA2_NP=1"; do
  set -- $v; name=$1; shift
  rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
  env "$@" UNIFIED_FA2_DYNAMIC=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name /dev/shm/w/venv/bin/python -u \
    build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
  echo "build $name $(grep -c UNIFIED_BUILD_OK build_$name.log)"
  grep -q UNIFIED_BUILD_OK build_$name.log || { grep -E " error" build_$name.log | head -5; echo BUILD_FAILED; exit 1; }; grep -E " error" build_$name.log | head -3
done
B=/opt/tiger/w/trace_sampled_h200_20261023
export UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_FA3_MIN=33 UNIFIED_MERGE_ROWS=4
OUT=/opt/tiger/w/unified_fa3items_h200_20261024
for name in g3_fa3items g3_np; do
  for d in mix32_two mix32_four mix128_two mix128_four; do
    UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so UNIFIED_GRAPH=1 UNIFIED_REPLAYS=5 \
      /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_${name}_$d.log 2>&1
    echo "check $name $d: FAIL lines $(grep -c FAIL check_${name}_$d.log), PASS lines $(grep -c PASS check_${name}_$d.log)"
    UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so /dev/shm/w/venv/bin/python -u \
      bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json --unified-only --append --label unified_$name 2>&1 | grep -E "FAIL|rror|Traceback"
  done
done
for arm in unified_g3_fa3items unified_g3_np unified_fa3items unified_np; do
  for b in arm:native_pdl_geometry_v3 default; do
    /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py $arm --baseline $b $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
  done
done
echo GEOMFA3_DONE
