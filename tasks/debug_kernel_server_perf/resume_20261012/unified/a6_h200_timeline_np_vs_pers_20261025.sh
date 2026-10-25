#!/bin/bash
# Persistent overlap + LPT vs the non-persistent (NP) FA2 executor under the same plan and the same merge (ticketed
# plan-scheduled merge items, UnifiedMergeRowCG), H200: per-item timeline (UNIFIED_TIMELINE=1) of both on the FA2 rows of
# the 96 trace-sampled captures -- compute span, tail after compute, merge wait / duration, pull cost. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
grep -q "Same record as the persistent loop" port07_unified/tests/unified_dispatch_ext.cu || { echo STALE; exit 1; }
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
build() {  # name, flags...
  local name=$1; shift
  rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
  env "$@" UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
  grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED $name; exit 1; }
  for w in "$@" UNIFIED_FA3_STOCK=1; do grep nvcc build_$name.log | grep -q -- "-D$w" || { echo FLAG_MISSING $name $w; exit 1; }; done
  echo "built $name"
}
build h200_np_tl UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_TIMELINE=1
build h200_pers_tl UNIFIED_FA2_NP=0 UNIFIED_MMA_KV16=2 UNIFIED_MMA_KV32=4 UNIFIED_FA2_MIN_BLOCKS=1 UNIFIED_FA2_MIN_BLOCKS16=3 UNIFIED_TIMELINE=1
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_FA2_CTAS_PER_SM=2 UNIFIED_FA2_CTAS_PER_SM16=3
unset UNIFIED_MERGE_MODE
B=/opt/tiger/w/trace_sampled_h200_20261025
for name in h200_np_tl h200_pers_tl; do
  export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
  for t in "two:mix32_two mix128_two" "four:mix32_four mix128_four"; do
    dirs=""; for s in ${t#*:}; do dirs="$dirs $B/$s/captures"; done
    timeout 1800 /dev/shm/w/venv/bin/python -u timeline_fa2_rows_20261025.py tl_$name.${t%%:*}.pt $dirs > tl_$name.${t%%:*}.log 2>&1
    echo "timeline $name ${t%%:*} $(grep -E 'TL_TOTAL|Error' tl_$name.${t%%:*}.log | tail -1)"
  done
done
echo TLNP_DONE
