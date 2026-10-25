#!/bin/bash
# Batched online-max merge rows (UnifiedMergeRowCG: 4 chunks per chunk group in flight, no separate max pass) in both
# FA2 executors (A6, H20). Per build: FP32 + 20 graph replays on all 92 captures, FA2 rows vs v3's FA2 launch; the
# persistent kernel under tail and ready merges plus its per-item timeline; stage 1 vs v3 for both executors.
# Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
grep -q "UnifiedMergeRowCG" port07_unified/include/flashinfer/attention/unified_merge.cuh || { echo STALE; exit 1; }
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
build() {  # name, flags...
  local name=$1; shift
  rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
  env "$@" UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
  grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED $name; exit 1; }
  for w in "$@" UNIFIED_FA3_STOCK=1; do grep nvcc build_$name.log | grep -q -- "-D$w" || { echo FLAG_MISSING $name $w; exit 1; }; done
}
build pers_u UNIFIED_FA2_NP=0 UNIFIED_TIMELINE=0
build pers_utl UNIFIED_FA2_NP=0 UNIFIED_TIMELINE=1
build np_u UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry
B=/opt/tiger/w/trace_sampled_h20_20261024; OUT=/opt/tiger/w/unified_h20_20261024
for arm in "pers_u:tail" "pers_u:ready" "np_u:tail"; do
  name=${arm%%:*}; mode=${arm#*:}
  export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
  if [ $mode = ready ]; then export UNIFIED_MERGE_MODE=ready; else unset UNIFIED_MERGE_MODE; fi
  tag=${name}_$mode
  for d in mix32_two mix32_four mix128_two mix128_four; do
    UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 timeout 1800 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_$tag.$d.log 2>&1
    echo "check $tag $d: FAIL $(grep -v GRAPH_FAILS check_$tag.$d.log | grep -c FAIL), PASS $(grep -c ' PASS' check_$tag.$d.log), stuck $(grep -c STUCK check_$tag.$d.log)"
  done
  for t in "two:mix32_two mix128_two" "four:mix32_four mix128_four"; do
    dirs=""; for s in ${t#*:}; do dirs="$dirs $B/$s/captures"; done
    timeout 1800 /dev/shm/w/venv/bin/python -u fa2_rows_compare_20261024.py $dirs > fa2rows_$tag.${t%%:*}.log 2>&1
    echo "rows $tag ${t%%:*} $(grep -E 'TOTAL|Error|STUCK' fa2rows_$tag.${t%%:*}.log | tail -1 | sed 's/batchprefill=[0-9.]*ms //' | cut -c1-110)"
  done
  if [ $name = pers_u ]; then
    export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_pers_utl/unified_dispatch_ext/unified_dispatch_ext.so
    for t in "two:mix32_two mix128_two" "four:mix32_four mix128_four"; do
      dirs=""; for s in ${t#*:}; do dirs="$dirs $B/$s/captures"; done
      timeout 1800 /dev/shm/w/venv/bin/python -u timeline_fa2_rows_20261025.py tl_u_${mode}_${t%%:*}.pt $dirs > tl_u_${mode}_${t%%:*}.log 2>&1
      echo "timeline $tag ${t%%:*} $(grep -E 'TL_TOTAL|Error' tl_u_${mode}_${t%%:*}.log | tail -1)"
    done
    export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
  fi
  label=u_${tag}_merge_unroll
  for d in mix32_two mix32_four mix128_two mix128_four; do
    /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json \
      --unified-only --append --label $label 2>&1 | grep -E "FAIL|rror|Traceback"
  done
  echo "stage1 $tag vs v3:"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py $label --baseline arm:native_pdl_geometry_v3 $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
done
unset UNIFIED_MERGE_MODE
echo MU_DONE
