#!/bin/bash
# Why is the merge-free bound (UNIFIED_NP_NOSYNC=1) SLOWER than the real merge on four-mode steps (captures 0.980 /
# 0.988x, synthetic 0.69-0.86x) while faster on two-mode? Per-item timelines (UNIFIED_TIMELINE=1) of the same steps with
# and without merges: FA3 grid span, FA2 compute end, per-item durations. Waits for the queued H200 runs. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
until grep -q SYNTH_RERUN_DONE a6_synth_rerun.log 2>/dev/null; do sleep 60; done
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
name=h200_np_nosync_tl
rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
UNIFIED_NP_NOSYNC=1 UNIFIED_TIMELINE=1 UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name \
  /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED; exit 1; }
for w in UNIFIED_NP_NOSYNC=1 UNIFIED_TIMELINE=1; do grep nvcc build_$name.log | grep -q -- "-D$w" || { echo FLAG_MISSING $w; exit 1; }; done
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3
unset UNIFIED_MERGE_MODE UNIFIED_FA2_ORDER UNIFIED_PDL
S=/opt/tiger/w/synth_meld_20261025; B=/opt/tiger/w/trace_sampled_h200_20261025
for b in h200_np_tl h200_np_nosync_tl; do
  for d in $S/sharegpt_four $S/sharegpt_two $B/mix32_four; do
    UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$b/unified_dispatch_ext/unified_dispatch_ext.so timeout 1800 \
      /dev/shm/w/venv/bin/python -u timeline_step_20261025.py $d/captures > nsp_${b}_$(basename $d).log 2>&1
    echo "== $b $(basename $d)"; grep -E "^STEP " nsp_${b}_$(basename $d).log | sed -E 's/^STEP [^ ]*\///' | cut -c1-230 | head -8
    grep -E "STEP_TOTAL|Error" nsp_${b}_$(basename $d).log | tail -1
  done
done
echo NOSYNC_PROBE_DONE
