#!/bin/bash
# A6 executors on synthetic MELD-shaped steps (gen_synth_meld_20261025.py: ShareGPT / math / long-tail / shared-prefix
# decode and verify rows at batch 16-1024 behind one 512-token prefill chunk), H200. Arms, each FP32-gated by the bench:
#   u_np_u      NP grid + ticketed merge items (the A6 default)        native_pdl_geometry_v3 (and sglang FA3) baselines
#   u_np_nosync merge-free timing bound (no merge, no fences)          u_npp   persistent pull loop, same tile code
#   u_nppj      persistent loop + ready-queue JIT merge                u_pers  persistent runner (plan-chosen CTAs/SM)
#   u_dsmem_c8 / c16  DSMEM-cluster reduction
# Answers: where does the reduction cost more than ~1%, does a CTA-per-item grid cost launch overhead at large batch, and
# what does the persistent kernel buy there. Waits for the plan-balance run. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
until grep -q PLAN_BALANCE_DONE a6_plan_balance.log 2>/dev/null; do sleep 60; done
export PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi CUDA_VISIBLE_DEVICES=0
S=/opt/tiger/w/synth_meld_20261025; OUT=/opt/tiger/w/synth_meld_results_20261025
[ -f $S/SYNTH_DONE ] || { /dev/shm/w/venv/bin/python -u gen_synth_meld_20261025.py $S > gen_synth.log 2>&1 && grep -q SYNTH_DONE gen_synth.log && touch $S/SYNTH_DONE; } || { echo GEN_FAILED; tail gen_synth.log; exit 1; }
cat gen_synth.log
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3
unset UNIFIED_MERGE_MODE UNIFIED_FA2_ORDER UNIFIED_PDL
SETS=$(cd $S && ls -d */ | tr -d / | tr '\n' ' ')
so() { echo /opt/tiger/w/torch_ext_unified_$1/unified_dispatch_ext/unified_dispatch_ext.so; }
run() {  # label, so name, first(1 = with baselines), extra env...
  local label=$1 name=$2 first=$3; shift 3
  for d in $SETS; do
    mkdir -p $OUT/$d
    if [ $first = 1 ]; then rm -f $OUT/$d/kernel_results.json; extra=""; else extra="--unified-only --append"; fi
    env "$@" UNIFIED_SO=$(so $name) timeout 3600 /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $S/$d/captures \
      $OUT/$d/kernel_results.json $extra --label $label > synth_${label}_$d.log 2>&1
    echo "$label $d: $(grep -c UNIFIED_BENCH_FAIL synth_${label}_$d.log) fails, $(grep -c '^UNIFIED_BENCH ' synth_${label}_$d.log) captures $(grep -m1 -E 'Error|Traceback' synth_${label}_$d.log | cut -c1-160)"
  done
}
run u_np_u h200_np_u 1
run u_np_nosync h200_np_nosync 0 UNIFIED_TIMING_BOUND=1
run u_npp h200_npp 0
run u_nppj h200_nppj 0 UNIFIED_MERGE_MODE=ready
run u_pers h200_pers_dual 0 UNIFIED_FA2_CTAS_PER_SM=2 UNIFIED_FA2_CTAS_PER_SM16=3
run u_dsmem_c8 h200_np_dsmem 0 UNIFIED_MERGE_MODE=dsmem UNIFIED_DSMEM_CLUSTER=8
run u_dsmem_c16 h200_np_dsmem 0 UNIFIED_MERGE_MODE=dsmem UNIFIED_DSMEM_CLUSTER=16
for d in $SETS; do /dev/shm/w/venv/bin/python summarize_synth_20261025.py u_np_u $OUT/$d/kernel_results.json; done
for d in $SETS; do
  UNIFIED_SO=$(so h200_np_tl) PLAN_BALANCE_TIMELINE=1 timeout 1800 /dev/shm/w/venv/bin/python -u plan_balance_20261025.py $S/$d/captures > pb_synth_$d.log 2>&1
  echo "balance $d:"; grep -E "^PLAN " pb_synth_$d.log | sed -E 's/^PLAN [^ ]*\///' | cut -c1-260; tail -1 pb_synth_$d.log
done
echo SYNTH_MELD_DONE
