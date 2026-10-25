#!/bin/bash
# Per-item timeline of the persistent overlap + LPT FA2 kernel (A6, H20): merge items at the queue tail vs the ready
# queue, on the FA2 rows of the 92 trace-sampled captures. Build with UNIFIED_TIMELINE=1 (checked on the nvcc line).
# Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
name=pers_tl
grep -q "g_timeline_n" port07_unified/tests/unified_dispatch_ext.cu || { echo STALE; exit 1; }
rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
UNIFIED_TIMELINE=1 UNIFIED_FA2_NP=0 UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED; exit 1; }
grep nvcc build_$name.log | grep -q -- "-DUNIFIED_TIMELINE=1" || { echo FLAG_NOT_ON_NVCC; exit 1; }
export CUDA_VISIBLE_DEVICES=0
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
B=/opt/tiger/w/trace_sampled_h20_20261024
for mode in tail ready; do
  for t in "two:mix32_two mix128_two" "four:mix32_four mix128_four"; do
    dirs=""; for s in ${t#*:}; do dirs="$dirs $B/$s/captures"; done
    if [ $mode = ready ]; then export UNIFIED_MERGE_MODE=ready; else unset UNIFIED_MERGE_MODE; fi
    timeout 1800 /dev/shm/w/venv/bin/python -u timeline_fa2_rows_20261025.py tl_${mode}_${t%%:*}.pt $dirs > tl_${mode}_${t%%:*}.log 2>&1
    echo "$mode ${t%%:*} $(grep -E 'TL_TOTAL|Error|STUCK' tl_${mode}_${t%%:*}.log | tail -1)"
  done
done
echo TL_DONE
