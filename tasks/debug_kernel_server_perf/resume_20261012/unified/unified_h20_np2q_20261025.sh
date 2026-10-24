#!/bin/bash
# A6 on H20: ticketed v3-body FA2 executor running the plan's queue with merge items interleaved L groups behind their
# compute (UNIFIED_MERGE_LAG), stock FA3, chunk-parallel merge. FA2 rows vs v3's FA2 launch per lag. Claude, 2026-10-25.
cd /dev/shm/w/ub
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
name=np2q_stock
rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name \
  /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
grep -q UNIFIED_BUILD_OK build_$name.log || { grep -E " error" build_$name.log | head; echo BUILD_FAILED; exit 1; }
for w in UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1; do grep -q -- "-D$w" build_$name.log || { echo FLAG_MISSING $w; exit 1; }; done
echo "build ok"
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_SPLIT_POLICY=geometry UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
B=/opt/tiger/w/trace_sampled_h20_20261024
for lag in -1 0 2 8; do
  for t in "two:mix32_two mix128_two" "four:mix32_four mix128_four"; do
    dirs=""; for s in ${t#*:}; do dirs="$dirs $B/$s/captures"; done
    UNIFIED_MERGE_LAG=$lag /dev/shm/w/venv/bin/python -u fa2_rows_compare_20261024.py $dirs > fa2rows_np2q_lag${lag}_${t%%:*}.log 2>&1
    echo "lag $lag ${t%%:*} $(grep -E 'TOTAL|Error' fa2rows_np2q_lag${lag}_${t%%:*}.log | tail -1 | sed 's/batchprefill=[0-9.]*ms //')"
  done
done
echo NP2Q_DONE
