#!/bin/bash
# Decode-only NP instantiation (UnifiedFA2NPKernel<false, true>: 32-row body compiled out, __launch_bounds__(128, 3),
# 16-row shared memory), chosen by the plan when every FA2 item is 16-row. Motivation: on synthetic decode batches the
# persistent runner's 3-CTA/SM decode instantiation beat the NP executor (2 CTAs/SM) by 1.08-1.14x at bs 256-1024.
# Build now; gate + time after the queued H200 runs. Ablation in the same build: UNIFIED_NP_ONLY16=0. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
grep -q 'UNIFIED_NP_MIN_BLOCKS16' port07_unified/tests/unified_dispatch_ext.cu || { echo STALE; exit 1; }
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
name=h200_np16
rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name \
  /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED; exit 1; }
echo "built $name; spills: $(grep -c 'bytes spill' build_$name.log) lines, nonzero: $(grep 'bytes spill' build_$name.log | grep -v ' 0 bytes spill stores, 0 bytes spill loads' | head -3 | tr '\n' ' ')"
until grep -q NOSYNC_PROBE_DONE a6_nosync_probe.log 2>/dev/null; do sleep 60; done
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_PRINT_OCC=1
unset UNIFIED_MERGE_MODE UNIFIED_FA2_ORDER UNIFIED_PDL UNIFIED_NP_ONLY16
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
S=/opt/tiger/w/synth_meld_20261025; OUT=/opt/tiger/w/synth_meld_results_20261025
B=/opt/tiger/w/trace_sampled_h200_20261025; R=/opt/tiger/w/unified_h200_20261025
for d in $B/mix32_two $B/mix32_four $B/mix128_two $B/mix128_four $S/sharegpt_two $S/math_two $S/sharedprefix_two $S/longtail_two; do
  UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 timeout 2400 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $d/captures/*.pt > check_np16_$(basename $d).log 2>&1
  echo "check np16 $(basename $d): FAIL $(grep -v GRAPH_FAILS check_np16_$(basename $d).log | grep -c FAIL), PASS $(grep -c ' PASS' check_np16_$(basename $d).log) $(grep -m1 UNIFIED_NP_OCC check_np16_$(basename $d).log)"
done
for d in $(cd $S && ls -d */ | tr -d /); do
  for v in 1 0; do
    UNIFIED_NP_ONLY16=$v timeout 3600 /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $S/$d/captures $OUT/$d/kernel_results.json --unified-only --append --label u_np16_$v > np16_${v}_$d.log 2>&1
  done
  echo "$d: $(grep -h -m1 UNIFIED_NP_OCC np16_1_$d.log)"
done
for d in mix32_two mix32_four mix128_two mix128_four; do
  for v in 1 0; do
    UNIFIED_NP_ONLY16=$v /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $R/$d/kernel_results.json --unified-only --append --label u_np16_$v 2>&1 | grep -E "FAIL|rror|Traceback"
  done
done
echo "captures, decode-only instantiation on vs off (same build):"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_np16_1 --baseline arm:u_np16_0 $R/*/kernel_results.json | sed -E 's/ \|.*//'
echo "captures, vs native v3:"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_np16_1 --baseline arm:native_pdl_geometry_v3 $R/*/kernel_results.json | sed -E 's/ \|.*//'
/dev/shm/w/venv/bin/python synth_peek.py > synth_peek_np16.out 2>&1; cat synth_peek_np16.out
echo NP16_DONE
