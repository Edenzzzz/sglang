#!/bin/bash
# A6 on H20, FA3 through stock flash-attention (UNIFIED_FA3_STOCK=1): the unified plan picks the FA3 split count, FA3
# runs with its own tile scheduler as native v3 launches it (stock_fa3_pdl, PDL primary), and the FA2 launch (v3 body,
# last-arriver merge) reduces FA2 groups and FA3 split rows with the plan's merge schedule; no combine / merge launches.
# FP32 + graph-replay check and stage 1 vs native v3 under geometry_fa3 and geometry. Claude, 2026-10-25.
cd /dev/shm/w/ub
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=${MAX_JOBS:-4} VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1
export PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
if [ "${1:-}" != run ]; then
  for v in "np_last_stock UNIFIED_NP_MERGE=0" "np2_stock UNIFIED_NP_MERGE=1"; do
    set -- $v; name=$1; shift
    [ -f /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so ] && { echo "build $name cached"; continue; }
    env UNIFIED_FA2_NP=1 "$@" UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name \
      /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
    grep -q UNIFIED_BUILD_OK build_$name.log || { grep -E " error" build_$name.log | head -5; echo BUILD_FAILED; exit 1; }
    for want in UNIFIED_FA2_NP=1 "$@" UNIFIED_FA3_STOCK=1; do
      grep -q -- "-D$want" build_$name.log || { echo "FLAG_MISSING $want"; echo BUILD_FAILED; exit 1; }
    done
    echo "build $name ok"
  done
  exit 0
fi
export CUDA_VISIBLE_DEVICES=0
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_FA3_MIN=33 UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1
B=/opt/tiger/w/trace_sampled_h20_20261024
OUT=/opt/tiger/w/unified_h20_20261024
for name in np_last_stock np2_stock; do
so=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
for pol in geometry_fa3 geometry; do
  label=u_${name}_${pol}
  for d in mix32_two mix32_four mix128_two mix128_four; do
    UNIFIED_SPLIT_POLICY=$pol UNIFIED_SO=$so UNIFIED_GRAPH=1 UNIFIED_REPLAYS=5 \
      /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_${label}_$d.log 2>&1
    echo "check $label $d: FAIL $(grep -v GRAPH_FAILS check_${label}_$d.log | grep -c FAIL), PASS $(grep -c ' PASS' check_${label}_$d.log), graph fails $(grep GRAPH_FAILS check_${label}_$d.log | awk '{s+=$2} END {print s+0}'), err $(grep -cE 'Error|Traceback' check_${label}_$d.log)"
    UNIFIED_SPLIT_POLICY=$pol UNIFIED_SO=$so /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures \
      $OUT/$d/kernel_results.json --unified-only --append --label $label 2>&1 | grep -E "FAIL|rror|Traceback"
  done
  for b in arm:native_pdl_geometry_v3 default; do
    /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py $label --baseline $b $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
  done
done
done
echo H20STOCK_DONE
