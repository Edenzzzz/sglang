#!/bin/bash
# A6 on the H200 (job 8574c727fab6ffbd, 132 SMs): the persistent overlap + LPT FA2 executor with the plan-chosen
# instantiation (16-row-only plans: 3 CTAs/SM, wide runners compiled out; else 2 CTAs/SM, 32-row class at MMA_KV 4) vs
# the v3-body (NP) executor, both with the batched merge rows, under geometry_fa3 (the H200's best policy). FA2 rows,
# FP32 + 20 graph replays on every capture, stage 1 vs native v3 / sglang FA3. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
grep -q "bool ONLY16 = false" port07_unified/tests/unified_dispatch_ext.cu || { echo STALE; exit 1; }
grep -q UnifiedMergeRowCG port07_unified/include/flashinfer/attention/unified_merge.cuh || { echo STALE; exit 1; }
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
build() {  # name, flags...
  local name=$1; shift
  rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
  env "$@" UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
  grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED $name; exit 1; }
  for w in "$@" UNIFIED_FA3_STOCK=1; do grep nvcc build_$name.log | grep -q -- "-D$w" || { echo FLAG_MISSING $name $w; exit 1; }; done
  echo "built $name"
}
build h200_np_u UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1
build h200_pers_dual UNIFIED_FA2_NP=0 UNIFIED_MMA_KV16=2 UNIFIED_MMA_KV32=4 UNIFIED_FA2_MIN_BLOCKS=1 UNIFIED_FA2_MIN_BLOCKS16=3
export CUDA_VISIBLE_DEVICES=0
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_PRINT_OCC=1
unset UNIFIED_MERGE_MODE
B=/opt/tiger/w/trace_sampled_h200_20261025; OUT=/opt/tiger/w/unified_h200_20261025
for name in h200_np_u h200_pers_dual; do
  export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so UNIFIED_FA2_CTAS_PER_SM=2 UNIFIED_FA2_CTAS_PER_SM16=3
  for t in "two:mix32_two mix128_two" "four:mix32_four mix128_four"; do
    dirs=""; for s in ${t#*:}; do dirs="$dirs $B/$s/captures"; done
    log=fa2rows_$name.${t%%:*}.log
    timeout 1800 /dev/shm/w/venv/bin/python -u fa2_rows_compare_20261024.py $dirs > $log 2>&1
    echo "rows $name ${t%%:*} $(grep UNIFIED_OCC $log | sort -u | cut -d' ' -f2,9- | tr '\n' ' ') $(grep -E 'TOTAL|Error|STUCK' $log | tail -1 | sed 's/batchprefill=[0-9.]*ms //; s/batchattention=[0-9.]*ms //' | cut -c1-70)"
  done
  for d in mix32_two mix32_four mix128_two mix128_four; do
    UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 timeout 1800 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_$name.$d.log 2>&1
    echo "check $name $d: FAIL $(grep -v GRAPH_FAILS check_$name.$d.log | grep -c FAIL), PASS $(grep -c ' PASS' check_$name.$d.log)"
    /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json \
      --unified-only --append --label u_$name 2>&1 | grep -E "FAIL|rror|Traceback"
  done
  for b in arm:native_pdl_geometry_v3 default; do
    /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_$name --baseline $b $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
  done
done
echo "persistent vs NP:"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_h200_pers_dual --baseline arm:u_h200_np_u $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
echo A6_H200_PERS_DONE
