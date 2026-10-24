#!/bin/bash
# v3-body FA2 executor with the last-arriver merge and blockIdx dispatch (UNIFIED_NP_MERGE=0: no ticket atomics, no
# merge CTAs), under v3's FA3 split (geometry_fa3) and v3's full split (geometry); FA2 rows, FP32 check, stage 1. Claude, 2026-10-24.
cd /dev/shm/w/ub
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8 VERBOSE=1 UNIFIED_FA2_CLASSES=2
export PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
grep -q '"geometry_fa3"' port07_unified/include/flashinfer/attention/fused_plan.cuh || { echo STALE; exit 1; }
grep -q UNIFIED_NP_MERGE port07_unified/tests/unified_dispatch_ext.cu || { echo STALE; exit 1; }
for v in "np_last UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=0"; do
  set -- $v; name=$1; shift
  rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
  env "$@" UNIFIED_FA2_DYNAMIC=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name /dev/shm/w/venv/bin/python -u \
    build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
  echo "build $name $(grep -c UNIFIED_BUILD_OK build_$name.log)"
  grep -q UNIFIED_BUILD_OK build_$name.log || { grep -E " error" build_$name.log | head -5; echo BUILD_FAILED; exit 1; }
  # The executor switch must reach the compiler: the build log's nvcc line carries every -D flag.
  for want in $(echo "$@" | grep -oE 'UNIFIED_[A-Z0-9_]+=[0-9]+'); do
    grep -q -- "-D$want" build_$name.log || { echo "FLAG_MISSING $want"; echo BUILD_FAILED; exit 1; }
  done; grep -E " error" build_$name.log | head -3
done
B=/opt/tiger/w/trace_sampled_h200_20261023
export UNIFIED_FA3_MIN=33 UNIFIED_MERGE_ROWS=4
OUT=/opt/tiger/w/unified_fa3items_h200_20261024
# geometry_fa3: v3's FA3 split, fixed FA2 split. geometry: v3's split for both (FA2 = row_split_size).
for pol in geometry_fa3 geometry; do
  for name in np_last; do
    label=unified_${name}_${pol}
    for d in mix32_two mix32_four mix128_two mix128_four; do
      UNIFIED_SPLIT_POLICY=$pol UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so UNIFIED_GRAPH=1 UNIFIED_REPLAYS=5 \
        /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_${label}_$d.log 2>&1
      echo "check $label $d: worst $(grep -oE 'worst[^ ]* [0-9.e-]+' check_${label}_$d.log | awk '{print $2}' | sort -g | tail -1), FAIL $(grep -v GRAPH_FAILS check_${label}_$d.log | grep -c FAIL), graph fails $(grep GRAPH_FAILS check_${label}_$d.log | awk '{s+=$2} END {print s+0}')"
      UNIFIED_SPLIT_POLICY=$pol UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so /dev/shm/w/venv/bin/python -u \
        bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json --unified-only --append --label $label 2>&1 | grep -E "FAIL|rror|Traceback"
    done
    for sets in "two:mix32_two mix128_two" "four:mix32_four mix128_four"; do
      tag=${sets%%:*}; dirs=""; for s in ${sets#*:}; do dirs="$dirs $B/$s/captures"; done
      UNIFIED_SPLIT_POLICY=$pol UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so \
        /dev/shm/w/venv/bin/python -u fa2_rows_compare_20261024.py $dirs > fa2rows_${label}_${tag}.log 2>&1
      echo "fa2rows $label $tag $(grep -E 'TOTAL|Error' fa2rows_${label}_${tag}.log | tail -1)"
    done
    for b in arm:native_pdl_geometry_v3 default; do
      /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py $label --baseline $b $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
    done
  done
done
echo NPLAST_DONE
