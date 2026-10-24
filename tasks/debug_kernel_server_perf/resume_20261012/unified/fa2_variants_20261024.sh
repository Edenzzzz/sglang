#!/bin/bash
# FA2 executor diagnosis (A6): our FA2 path vs stock BatchAttention on identical decode/verify rows, under
# dynamic vs static LPT, with vs without the in-epilogue merge (timing bound), fixed vs flashinfer split rule.
cd /dev/shm/w/ub
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
B=/opt/tiger/w/trace_sampled_h200_20261023
for v in "base2 fixed 0" "base2 flashinfer 0" "static fixed 1" "nomerge fixed 0"; do
  set -- $v
  UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$1/unified_dispatch_ext/unified_dispatch_ext.so UNIFIED_SPLIT_POLICY=$2 UNIFIED_FA2_STATIC=$3 \
    /dev/shm/w/venv/bin/python -u fa2_rows_compare_20261024.py $B/mix128_two/captures $B/mix32_four/captures 2>&1 | grep TOTAL | sed "s/^/$1 $2 static=$3 /"
done
echo FA2_VARIANTS_DONE
