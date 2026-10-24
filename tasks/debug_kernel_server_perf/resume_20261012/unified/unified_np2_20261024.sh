#!/bin/bash
# Non-persistent FA2 executor with ticketed merge items (A6): v3 body, merge spread over CTAs like the persistent kernel.
# the same mechanism as FA2's merge items (no tail loop). Build, FP32 + graph-replay check on all 96 captures, then
# stage 1 vs native v3 beside the earlier arms. Claude, 2026-10-24.
cd /dev/shm/w/ub
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8 VERBOSE=1 UNIFIED_FA2_CLASSES=2
export PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
grep -c "num_fa3_merge_items" port07_unified/include/flashinfer/attention/fused_plan.cuh >/dev/null || { echo STALE; exit 1; }
grep -c "ENABLE_NATIVE_Q32" port07_unified/include/flashinfer/attention/prefill.cuh >/dev/null || { echo STALE; exit 1; }
grep -q ticket_state port07_unified/tests/unified_dispatch_ext.cu || { echo STALE; exit 1; }
# fa3items: persistent overlap + LPT FA2 kernel; np: v3's non-persistent BatchPrefill body. Same plan, same merge
# schedule (FA2 groups + FA3 merge items), no merge launches.
for v in "np2 UNIFIED_FA2_NP=1"; do
  set -- $v; name=$1; shift
  rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
  env "$@" UNIFIED_FA2_DYNAMIC=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name /dev/shm/w/venv/bin/python -u \
    build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
  echo "build $name $(grep -c UNIFIED_BUILD_OK build_$name.log)"
  grep -q UNIFIED_BUILD_OK build_$name.log || { grep -E " error" build_$name.log | head -5; echo BUILD_FAILED; exit 1; }
  # The executor switch must reach the compiler: the build log's nvcc line carries every -D flag.
  want=$(echo "$@" | grep -oE 'UNIFIED_FA2_NP=[01]')
  grep -q -- "-D$want" build_$name.log || { echo "FLAG_MISSING $want"; echo BUILD_FAILED; exit 1; }; grep -B1 -A3 "entry function '_Z\(16UnifiedFA2Kernel\|18UnifiedFA2NPKernel\)" build_$name.log | grep -E 'spill|registers'
  grep -E " error" build_$name.log | head -5
done
B=/opt/tiger/w/trace_sampled_h200_20261023
export UNIFIED_SPLIT_POLICY=fixed UNIFIED_FA3_MIN=33 UNIFIED_MERGE_ROWS=4
for name in np2; do
  export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
  for sets in "two:mix32_two mix128_two" "four:mix32_four mix128_four"; do
    tag=${sets%%:*}; dirs=""; for s in ${sets#*:}; do dirs="$dirs $B/$s/captures"; done
    /dev/shm/w/venv/bin/python -u fa2_rows_compare_20261024.py $dirs > fa2rows_${name}_${tag}.log 2>&1
    echo "fa2rows $name $tag $(grep -E 'TOTAL|Error' fa2rows_${name}_${tag}.log | tail -1)"
  done
  for d in mix32_two mix32_four mix128_two mix128_four; do
    UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_${name}_$d.log 2>&1
    echo "check $name $d: FAIL lines $(grep -c FAIL check_${name}_$d.log), PASS lines $(grep -c PASS check_${name}_$d.log), $(grep GRAPH_FAILS check_${name}_$d.log | sort | uniq -c | tr '\n' ' ')"
  done
done
SRC=/opt/tiger/w/unified_mi_h200_20261024; OUT=/opt/tiger/w/unified_fa3items_h200_20261024; mkdir -p $OUT
for d in mix32_two mix32_four mix128_two mix128_four; do
  mkdir -p $OUT/$d; [ -f $OUT/$d/kernel_results.json ] || cp $SRC/$d/kernel_results.json $OUT/$d/kernel_results.json
  for name in np2; do
    UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so /dev/shm/w/venv/bin/python -u \
      bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json --unified-only --append --label unified_$name 2>&1 | grep -E "FAIL|rror|Traceback"
  done
done
for arm in unified_np2 unified_np unified_fa3items; do
  for b in default arm:native_pdl_geometry_v3 arm:fa3_dual_split32; do
    /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py $arm --baseline $b $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
  done
done
echo NP2_DONE
