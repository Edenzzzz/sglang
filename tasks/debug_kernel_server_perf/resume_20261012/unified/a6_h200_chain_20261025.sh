#!/bin/bash
# A6 on the re-forked H200 (job 8574c727fab6ffbd, 132 SMs), 2026-10-25: re-capture the trace-sampled steps (real 4-mode
# MTP serving, same intervals as 2026-10-23), re-bench native v3 on them, build the race-fixed executor (v3 body,
# chunk-parallel ticketed merge items, stock FA3 with the plan's split, no early PDL trigger), then FA2 rows vs v3's FA2
# launch, FP32 + graph-replay check on every capture, and stage 1 vs v3 / sglang FA3 under geometry and geometry_fa3.
# Claude.
set -u
cd /dev/shm/w/ub
sed -e "s#ROOT=/opt/tiger/w/trace_sampled_h20_20261024#ROOT=/opt/tiger/w/trace_sampled_h200_20261025#" trace_sampled_capture_h20_20261024.sh > trace_sampled_capture_h200_20261025.sh
bash trace_sampled_capture_h200_20261025.sh > /opt/tiger/w/trace_sampled_h200_20261025.log 2>&1
grep -q TRACE_SAMPLED_CAPTURE_DONE /opt/tiger/w/trace_sampled_h200_20261025.log || { echo CAPTURE_FAILED; tail -20 /opt/tiger/w/trace_sampled_h200_20261025.log; exit 1; }
echo "captures $(ls /opt/tiger/w/trace_sampled_h200_20261025/*/captures/*.pt | wc -l)"
bash rebench_trace_sampled_v3_h200_20261025.sh > rebench_v3_h200.log 2>&1
grep -E "time-weighted|REBENCH" rebench_v3_h200.log | sed -E 's/ \|.*//'
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
name=np2s_stock
rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name \
  /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
grep -q UNIFIED_BUILD_OK build_$name.log || { grep -E " error" build_$name.log | head; echo BUILD_FAILED; exit 1; }
for w in UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1; do grep -q -- "-D$w" build_$name.log || { echo FLAG_MISSING $w; exit 1; }; done
grep -q suppress_pdl_trigger port07_unified/include/flashinfer/attention/prefill.cuh || { echo STALE_PORT07; exit 1; }
echo "build $name ok"
export CUDA_VISIBLE_DEVICES=0
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
B=/opt/tiger/w/trace_sampled_h200_20261025; OUT=/opt/tiger/w/unified_h200_20261025; mkdir -p $OUT
for d in mix32_two mix32_four mix128_two mix128_four; do mkdir -p $OUT/$d; cp $B/$d/kernel_results_v3.json $OUT/$d/kernel_results.json; done
for pol in geometry geometry_fa3; do
  for t in "two:mix32_two mix128_two" "four:mix32_four mix128_four"; do
    dirs=""; for s in ${t#*:}; do dirs="$dirs $B/$s/captures"; done
    UNIFIED_SPLIT_POLICY=$pol /dev/shm/w/venv/bin/python -u fa2_rows_compare_20261024.py $dirs > fa2rows_h200_${pol}_${t%%:*}.log 2>&1
    echo "fa2rows $pol ${t%%:*} $(grep -E 'TOTAL|Error' fa2rows_h200_${pol}_${t%%:*}.log | tail -1 | sed 's/batchprefill=[0-9.]*ms //')"
  done
  for d in mix32_two mix32_four mix128_two mix128_four; do
    UNIFIED_SPLIT_POLICY=$pol UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_h200_${pol}_$d.log 2>&1
    echo "check $pol $d: FAIL $(grep -v GRAPH_FAILS check_h200_${pol}_$d.log | grep -c FAIL), PASS $(grep -c ' PASS' check_h200_${pol}_$d.log), graph fails $(grep GRAPH_FAILS check_h200_${pol}_$d.log | awk '{s+=$2} END {print s+0}'), err $(grep -cE 'Error|Traceback' check_h200_${pol}_$d.log)"
    UNIFIED_SPLIT_POLICY=$pol /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json \
      --unified-only --append --label u_np2s_$pol 2>&1 | grep -E "FAIL|rror|Traceback"
  done
  for b in arm:native_pdl_geometry_v3 default arm:fa3_dual_split32; do
    /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_np2s_$pol --baseline $b $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
  done
done
echo A6_H200_DONE
