#!/bin/bash
# Persistent overlap + LPT FA2 kernel (A6, H20): plan-chosen instantiation -- 16-row-only plans run a 3-CTA/SM register
# cap (MIN_BLOCKS16=3), plans with 32-row items the 2-CTA build with the 32-row class at MMA_KV 4. FA2 rows, occupancy
# printed, FP32 + 20 graph replays and stage 1. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
grep -q "UNIFIED_OCC variant" port07_unified/tests/unified_dispatch_ext.cu || { echo STALE; exit 1; }
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
build() {  # name, flags...
  local name=$1; shift
  rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
  env "$@" UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
  grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED $name; exit 1; }
  for w in "$@" UNIFIED_FA3_STOCK=1; do grep nvcc build_$name.log | grep -q -- "-D$w" || { echo FLAG_MISSING $name $w; exit 1; }; done
  echo "built $name: $(grep -A2 "Compiling entry function .*UnifiedFA2Kernel" build_$name.log | grep -E "registers|spill" | head -2 | tr "\n" " ")"
}
build pers_dual UNIFIED_FA2_NP=0 UNIFIED_MMA_KV16=2 UNIFIED_MMA_KV32=4 UNIFIED_FA2_MIN_BLOCKS=1 UNIFIED_FA2_MIN_BLOCKS16=3
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry UNIFIED_PRINT_OCC=1
unset UNIFIED_MERGE_MODE
B=/opt/tiger/w/trace_sampled_h20_20261024; OUT=/opt/tiger/w/unified_h20_20261024
best=""; bestt=999
for name in pers_dual; do
  export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
  for c in 2; do
    sum=0
    for t in "two:mix32_two mix128_two" "four:mix32_four mix128_four"; do
      dirs=""; for s in ${t#*:}; do dirs="$dirs $B/$s/captures"; done
      log=fa2rows_dual_${name}_c$c.${t%%:*}.log
      UNIFIED_FA2_CTAS_PER_SM=$c UNIFIED_FA2_CTAS_PER_SM16=3 timeout 1800 /dev/shm/w/venv/bin/python -u fa2_rows_compare_20261024.py $dirs > $log 2>&1
      echo "rows $name ctas/SM $c ${t%%:*} $(grep UNIFIED_OCC $log | sort -u | cut -d" " -f2,9- | tr "\n" " ") $(grep -E 'TOTAL|Error|STUCK' $log | tail -1 | sed 's/batchprefill=[0-9.]*ms //; s/batchattention=[0-9.]*ms //' | cut -c1-70)"
      u=$(grep -o 'FA2ROWS_TOTAL unified=[0-9.]*' $log | grep -o '[0-9.]*$'); sum=$(python3 -c "print(${sum}+${u:-999})")
    done
    if python3 -c "import sys; sys.exit(0 if $sum < $bestt else 1)"; then best="$name:$c"; bestt=$sum; fi
  done
done
echo "best $best total $bestt"
name=${best%%:*}; c=${best#*:}
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so UNIFIED_FA2_CTAS_PER_SM=$c UNIFIED_FA2_CTAS_PER_SM16=3
for d in mix32_two mix32_four mix128_two mix128_four; do
  UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 timeout 1800 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_dual.$d.log 2>&1
  echo "check $best $d: FAIL $(grep -v GRAPH_FAILS check_dual.$d.log | grep -c FAIL), PASS $(grep -c ' PASS' check_dual.$d.log)"
done
label=u_pers_dual_${name}_c$c
for d in mix32_two mix32_four mix128_two mix128_four; do
  /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json \
    --unified-only --append --label $label 2>&1 | grep -E "FAIL|rror|Traceback"
done
echo "stage1 $best vs v3:"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py $label --baseline arm:native_pdl_geometry_v3 $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
echo DUAL_DONE
