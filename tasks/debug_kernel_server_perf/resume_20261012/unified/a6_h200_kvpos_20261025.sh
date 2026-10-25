#!/bin/bash
# FA2 L2 order (UNIFIED_FA2_ORDER=kvpos: KV-position-major, longest first within a position) vs LPT, NP executor with
# the ticketed merge, H200: the 96 trace-sampled captures, the synthetic MELD-shaped sets, and a 48K shared-prefix set
# (sharedlong; the prefix exceeds L2). Waits for the synthetic sweep and the plan-balance rerun. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
until grep -q SYNTH_MELD_DONE a6_synth_meld.log 2>/dev/null && grep -q PLAN_BALANCE_DONE a6_plan_balance.log 2>/dev/null; do sleep 60; done
grep -q 'UNIFIED_FA2_ORDER=kvpos' port07_unified/include/flashinfer/attention/fused_plan.cuh || { echo STALE; exit 1; }
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
name=h200_np_kv
rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name \
  /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED; exit 1; }
echo "built $name"
S=/opt/tiger/w/synth_meld_20261025; OUT=/opt/tiger/w/synth_meld_results_20261025
[ -d $S/sharedlong_two ] || /dev/shm/w/venv/bin/python -u gen_synth_meld_20261025.py $S sharedlong > gen_sharedlong.log 2>&1
grep SYNTH gen_sharedlong.log
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3
unset UNIFIED_MERGE_MODE UNIFIED_FA2_ORDER UNIFIED_PDL
SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
bench() {  # captures dir, results json, label, first, env...
  local c=$1 j=$2 l=$3 f=$4; shift 4; local extra="--unified-only --append"; [ $f = 1 ] && { rm -f $j; extra=""; }
  env "$@" UNIFIED_SO=$SO timeout 3600 /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $c $j $extra --label $l > kv_${l}_$(basename $(dirname $j)).log 2>&1
  echo "$l $(basename $(dirname $j)): $(grep -c UNIFIED_BENCH_FAIL kv_${l}_$(basename $(dirname $j)).log) fails $(grep -m1 -E 'Error|Traceback' kv_${l}_$(basename $(dirname $j)).log | cut -c1-150)"
}
for d in sharedlong_two sharedlong_four; do mkdir -p $OUT/$d; bench $S/$d/captures $OUT/$d/kernel_results.json u_np_lpt2 1; done
for d in $(cd $S && ls -d */ | tr -d / ); do
  case $d in sharedlong_*) ;; *) bench $S/$d/captures $OUT/$d/kernel_results.json u_np_lpt2 0;; esac
  bench $S/$d/captures $OUT/$d/kernel_results.json u_np_kvpos 0 UNIFIED_FA2_ORDER=kvpos
  /dev/shm/w/venv/bin/python summarize_synth_20261025.py u_np_lpt2 $OUT/$d/kernel_results.json | grep -v fa3_
done
B=/opt/tiger/w/trace_sampled_h200_20261025; R=/opt/tiger/w/unified_h200_20261025
for d in mix32_two mix32_four mix128_two mix128_four; do
  bench $B/$d/captures $R/$d/kernel_results.json u_np_lpt2 0
  bench $B/$d/captures $R/$d/kernel_results.json u_np_kvpos 0 UNIFIED_FA2_ORDER=kvpos
done
echo "captures, kvpos vs LPT (same build):"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_np_kvpos --baseline arm:u_np_lpt2 $R/*/kernel_results.json | sed -E 's/ \|.*//'
echo KVPOS_DONE
