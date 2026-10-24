#!/bin/bash
# A6 on H20 (job 2f608c1b1a67cf09 pod 0, GPU 0; 78 SMs), 2026-10-24, while the H200 re-fork is queued. Three FA2
# executors under one unified plan and its in-launch merge schedule:
#   np_last  = v3's BatchPrefill body, each split group reduced by its last-arriving chunk (blockIdx, no ticket atomics)
#   np2      = v3's body, ticketed merge items
#   fa3items = persistent overlap + LPT kernel, merge items
# under v3's FA3 split (geometry_fa3) and the fixed rule. Per arm: FA2 rows vs v3's FA2 launch, FP32 + graph-replay
# check, stage 1 vs native v3 / sglang's FA3 call on the H20 trace-sampled captures. Claude.
cd /dev/shm/w/ub
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_FA3_MIN=33 UNIFIED_MERGE_ROWS=4
B=/opt/tiger/w/trace_sampled_h20_20261024
OUT=/opt/tiger/w/unified_h20_20261024; mkdir -p $OUT
for d in mix32_two mix32_four mix128_two mix128_four; do
  mkdir -p $OUT/$d; [ -f $OUT/$d/kernel_results.json ] || cp $B/$d/kernel_results_v3.json $OUT/$d/kernel_results.json
done
for pol in geometry; do
  for name in np_last np2; do
    label=u_${name}_${pol}
    so=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
    for sets in "two:mix32_two mix128_two" "four:mix32_four mix128_four"; do
      tag=${sets%%:*}; dirs=""; for s in ${sets#*:}; do dirs="$dirs $B/$s/captures"; done
      UNIFIED_SPLIT_POLICY=$pol UNIFIED_SO=$so /dev/shm/w/venv/bin/python -u fa2_rows_compare_20261024.py $dirs > fa2rows_${label}_${tag}.log 2>&1
      echo "fa2rows $label $tag $(grep -E 'TOTAL|Error' fa2rows_${label}_${tag}.log | tail -1)"
    done
    for d in mix32_two mix32_four mix128_two mix128_four; do
      UNIFIED_SPLIT_POLICY=$pol UNIFIED_SO=$so UNIFIED_GRAPH=1 UNIFIED_REPLAYS=5 \
        /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_${label}_$d.log 2>&1
      echo "check $label $d: FAIL $(grep -v GRAPH_FAILS check_${label}_$d.log | grep -c FAIL), PASS $(grep -c ' PASS' check_${label}_$d.log), graph fails $(grep GRAPH_FAILS check_${label}_$d.log | awk '{s+=$2} END {print s+0}')"
      UNIFIED_SPLIT_POLICY=$pol UNIFIED_SO=$so /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures \
        $OUT/$d/kernel_results.json --unified-only --append --label $label 2>&1 | grep -E "FAIL|rror|Traceback"
    done
    for b in arm:native_pdl_geometry_v3 default; do
      /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py $label --baseline $b $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
    done
  done
done
echo H20GEOM_DONE
