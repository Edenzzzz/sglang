#!/bin/bash
# FA2 chunk rounding (UNIFIED_FA2_CHUNK_ROUND=tile: per-slot share rounded up to 256 keys, not a power of two) vs the
# default, same build, H200: synthetic MELD-shaped sets + the 96 captures, then the FA2 balance bounds with tile
# rounding on the synthetic sets. Waits for the decode-only instantiation run. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
grep -q 'UNIFIED_FA2_CHUNK_ROUND' port07_unified/include/flashinfer/attention/fused_plan.cuh || { echo STALE; exit 1; }
until grep -q NP16_DONE a6_np16.log 2>/dev/null; do sleep 60; done
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
for name in h200_np_cr h200_np_cr_tl; do
  tl=0; [ $name = h200_np_cr_tl ] && tl=1
  rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
  UNIFIED_TIMELINE=$tl UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name \
    /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
  grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED; exit 1; }
done
echo built
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3
unset UNIFIED_MERGE_MODE UNIFIED_FA2_ORDER UNIFIED_PDL UNIFIED_NP_ONLY16 UNIFIED_FA2_CHUNK_ROUND
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_h200_np_cr/unified_dispatch_ext/unified_dispatch_ext.so
S=/opt/tiger/w/synth_meld_20261025; OUT=/opt/tiger/w/synth_meld_results_20261025
B=/opt/tiger/w/trace_sampled_h200_20261025; R=/opt/tiger/w/unified_h200_20261025
for d in $S/sharegpt_four $S/sharegpt_two $S/longtail_four $B/mix32_four $B/mix128_two; do
  UNIFIED_FA2_CHUNK_ROUND=tile UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 timeout 2400 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $d/captures/*.pt > check_cr_$(basename $d).log 2>&1
  echo "check tile-round $(basename $d): FAIL $(grep -v GRAPH_FAILS check_cr_$(basename $d).log | grep -c FAIL), PASS $(grep -c ' PASS' check_cr_$(basename $d).log)"
done
for d in $(cd $S && ls -d */ | tr -d /); do
  for v in pow2 tile; do
    UNIFIED_FA2_CHUNK_ROUND=$v timeout 3600 /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $S/$d/captures $OUT/$d/kernel_results.json --unified-only --append --label u_cr_$v > cr_${v}_$d.log 2>&1
  done
done
for d in mix32_two mix32_four mix128_two mix128_four; do
  for v in pow2 tile; do
    UNIFIED_FA2_CHUNK_ROUND=$v /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $R/$d/kernel_results.json --unified-only --append --label u_cr_$v 2>&1 | grep -E "FAIL|rror|Traceback"
  done
done
echo "captures, tile rounding vs power of two (same build):"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_cr_tile --baseline arm:u_cr_pow2 $R/*/kernel_results.json | sed -E 's/ \|.*//'
sed -i "s/'u_np16_1','u_np16_0'\]/'u_np16_1','u_np16_0','u_cr_pow2','u_cr_tile']/" synth_peek.py
/dev/shm/w/venv/bin/python synth_peek.py > synth_peek_cr.out 2>&1; cat synth_peek_cr.out
for d in sharegpt_four sharegpt_two longtail_four math_two; do
  UNIFIED_FA2_CHUNK_ROUND=tile UNIFIED_SO=/opt/tiger/w/torch_ext_unified_h200_np_cr_tl/unified_dispatch_ext/unified_dispatch_ext.so \
    timeout 1800 /dev/shm/w/venv/bin/python -u plan_balance_20261025.py $S/$d/captures > pb_cr_$d.log 2>&1
  echo "balance tile $d:"; grep -E "^PLAN " pb_cr_$d.log | sed -E 's/^PLAN [^ ]*\///; s/fa3_items.*\| //' | cut -c1-200; tail -1 pb_cr_$d.log
done
echo CHUNK_ROUND_DONE
