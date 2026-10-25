#!/bin/bash
# Persistent overlap + LPT FA2 kernel (A6, H20): the 32-row class at v3's Q32 KV depth (MMA_KV 4, 2q x 2kv = 128-key
# tile) vs MMA_KV 2, and the persistent grid at 2 / 3 / 4 CTAs per SM (4-warp CTAs). FA2 rows vs v3's FA2 launch; FP32
# + 20 graph replays on the best. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
grep -q UNIFIED_FA2_CTAS_PER_SM unified_executor_20261024.py || { echo STALE; exit 1; }
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
build() {  # name, flags...
  local name=$1; shift
  rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
  env "$@" UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
  grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED $name; exit 1; }
  for w in "$@" UNIFIED_FA3_STOCK=1; do grep nvcc build_$name.log | grep -q -- "-D$w" || { echo FLAG_MISSING $name $w; exit 1; }; done
  echo "built $name: $(grep -A2 'Compiling entry function .*UnifiedFA2Kernel' build_$name.log | grep -E 'registers|spill' | head -2 | tr '\n' ' ')"
}
build pers_kv32_4 UNIFIED_FA2_NP=0 UNIFIED_MMA_KV16=2 UNIFIED_MMA_KV32=4
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry
unset UNIFIED_MERGE_MODE
B=/opt/tiger/w/trace_sampled_h20_20261024
for name in pers_u pers_kv32_4; do
  export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
  for c in 2 3 4; do
    for t in "two:mix32_two mix128_two" "four:mix32_four mix128_four"; do
      dirs=""; for s in ${t#*:}; do dirs="$dirs $B/$s/captures"; done
      UNIFIED_FA2_CTAS_PER_SM=$c timeout 1800 /dev/shm/w/venv/bin/python -u fa2_rows_compare_20261024.py $dirs > fa2rows_occ_${name}_c$c.${t%%:*}.log 2>&1
      echo "rows $name ctas/SM $c ${t%%:*} $(grep -E 'TOTAL|Error|STUCK' fa2rows_occ_${name}_c$c.${t%%:*}.log | tail -1 | sed 's/batchprefill=[0-9.]*ms //; s/batchattention=[0-9.]*ms //' | cut -c1-90)"
    done
  done
done
echo OCC_DONE
