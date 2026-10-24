#!/bin/bash
# Per-kernel FA2 breakdown, ours vs v3's FA2 launch, on the FA2 rows of the two-mode and four-mode sets (A6). Ours with
# merge items (p_base2) and with the merge skipped (p_nomerge, wrong output, timing bound). Claude, 2026-10-24.
cd /dev/shm/w/ub
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8 VERBOSE=1 UNIFIED_FA2_CLASSES=2
export PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
if [ ! -f /opt/tiger/w/torch_ext_unified_p_nomerge/unified_dispatch_ext/unified_dispatch_ext.so ]; then
  UNIFIED_FA2_DYNAMIC=1 UNIFIED_SKIP_MERGE=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_p_nomerge \
    /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_p_nomerge.log 2>&1
fi
B=/opt/tiger/w/trace_sampled_h200_20261023
for v in p_base2 p_nomerge; do
  for sets in "two:mix32_two mix128_two" "four:mix32_four mix128_four"; do
    tag=${sets%%:*}; dirs=""; for s in ${sets#*:}; do dirs="$dirs $B/$s/captures"; done
    UNIFIED_SPLIT_POLICY=fixed UNIFIED_MERGE_ROWS=4 \
    UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$v/unified_dispatch_ext/unified_dispatch_ext.so \
      /dev/shm/w/venv/bin/python -u profile_fa2_rows_20261024.py $dirs > fa2prof_${v}_${tag}.log 2>&1
    echo "$v $tag"; grep -E "PROF_TOTAL|Error" fa2prof_${v}_${tag}.log | tail -3
  done
done
echo FA2PROF_DONE
