#!/bin/bash
# FA2 parity with native v3 (A6): our FA2 rows vs v3's own FA2 row launch on identical rows.
# Variables: KV depth of the 32-row tile (v3's Q32 dispatch gets NUM_MMA_KV=4; ours had 2) and the split policy
# (fixed vs geometry = v3's row_split_size). Claude, 2026-10-24.
cd /dev/shm/w/ub
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8 VERBOSE=1 UNIFIED_FA2_CLASSES=2
export PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
grep -c 'strcmp(e, "geometry")' port07_unified/include/flashinfer/attention/fused_plan.cuh || { echo "STALE HEADER"; exit 1; }
build() {  # name, env...
  local name=$1; shift
  rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
  env "$@" TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name /dev/shm/w/venv/bin/python -u \
    build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
  echo "== build $name $(grep -c UNIFIED_BUILD_OK build_$name.log)"
  grep -A3 "entry function '_Z16UnifiedFA2Kernel" build_$name.log | grep -E 'spill|registers'
}
build p_base2 UNIFIED_FA2_DYNAMIC=1
build p_kv32_4 UNIFIED_FA2_DYNAMIC=1 UNIFIED_MMA_KV32=4
grep -c 'strcmp(e, "geometry")' merged/*/include/flashinfer/attention/fused_plan.cuh 2>/dev/null | head -2
B=/opt/tiger/w/trace_sampled_h200_20261023
for v in p_base2 p_kv32_4; do
  for pol in fixed geometry; do
    for sets in "two:mix32_two mix128_two" "four:mix32_four mix128_four"; do
      tag=${sets%%:*}; dirs=""; for s in ${sets#*:}; do dirs="$dirs $B/$s/captures"; done
      UNIFIED_SPLIT_POLICY=$pol UNIFIED_MERGE_ROWS=4 \
      UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$v/unified_dispatch_ext/unified_dispatch_ext.so \
        /dev/shm/w/venv/bin/python -u fa2_rows_compare_20261024.py $dirs > fa2par_${v}_${pol}_${tag}.log 2>&1
      echo "$v $pol $tag $(grep -E 'TOTAL|Error|error' fa2par_${v}_${pol}_${tag}.log | tail -2)"
    done
  done
done
echo PARITY_DONE
