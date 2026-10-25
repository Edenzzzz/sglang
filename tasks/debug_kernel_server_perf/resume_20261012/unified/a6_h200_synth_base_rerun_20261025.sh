#!/bin/bash
# Re-run the baseline pass (native v3, sglang FA3, A6 default) on the synthetic four-mode sets whose large batches
# overflowed the native harness's 512 MB float workspace (now allowed to grow to 4 GB). Appends to the same records.
# Waits for the L2-order run. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
until grep -q KVPOS_DONE a6_kvpos.log 2>/dev/null; do sleep 60; done
grep -q "max_workspace_bytes=4 << 30" bench_captured_mixed_20261023.py || { echo STALE; exit 1; }
export PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi CUDA_VISIBLE_DEVICES=0
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3
unset UNIFIED_MERGE_MODE UNIFIED_FA2_ORDER UNIFIED_PDL
S=/opt/tiger/w/synth_meld_20261025; OUT=/opt/tiger/w/synth_meld_results_20261025
for d in $(cd $S && ls -d */ | tr -d /); do
  grep -q Traceback synth_u_np_u_$d.log 2>/dev/null || continue
  UNIFIED_SO=/opt/tiger/w/torch_ext_unified_h200_np_u/unified_dispatch_ext/unified_dispatch_ext.so timeout 3600 \
    /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $S/$d/captures $OUT/$d/kernel_results.json --append --label u_np_u > synth_base_rerun_$d.log 2>&1
  echo "rerun $d: $(grep -c '^UNIFIED_BENCH ' synth_base_rerun_$d.log) captures $(grep -m1 -E 'Error|Traceback' synth_base_rerun_$d.log | cut -c1-150)"
  /dev/shm/w/venv/bin/python summarize_synth_20261025.py u_np_u $OUT/$d/kernel_results.json | grep -v fa3_
done
echo SYNTH_RERUN_DONE
