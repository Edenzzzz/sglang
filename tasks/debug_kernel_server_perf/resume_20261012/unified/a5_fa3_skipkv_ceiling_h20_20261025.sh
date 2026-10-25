#!/bin/bash
# A5.5 zero-KV-load ceiling on the flash-attention role (H20): stock_fa3_pdl rebuilt from flash-attention e9515d5 with
# FLASHINFER_FA3_SKIP_KV_LOAD=1 (paged K/V cp.async loads compiled out; the pipeline still commits) and, from the same
# tree, a control build without it. Stage 1 step time of A6 (NP executor, batched merge) with each library on the 92
# trace-sampled captures. The skip arm is a timing bound (output not validated): the most any KV-traffic reduction
# (multicast) could save on the step. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
W=/dev/shm/w/fa3skip; mkdir -p $W
[ -d $W/flash-attention ] || tar xzf /dev/shm/w/bundle/flash_attention_e9515d5_skipkv.tgz -C $W
grep -q FLASHINFER_FA3_SKIP_KV_LOAD $W/flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp || { echo NOT_PATCHED; exit 1; }
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8 PYTHONPATH=/dev/shm/w/up/python:$W/stock_fa3_unsplit_work_20261014 STOCK_FLASH_ATTENTION=$W/flash-attention STOCK_FA3_UNSPLIT_SPECIALIZE=1
for v in ctrl skip; do
  extra=""; [ $v = skip ] && extra="-DFLASHINFER_FA3_SKIP_KV_LOAD=1"
  rm -rf /opt/tiger/w/torch_ext_fa3_$v/stock_fa3_pdl
  STOCK_FA3_MERGED=/opt/tiger/w/stock_fa3_merged_$v STOCK_FA3_EXTRA="$extra" TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_fa3_$v \
    /dev/shm/w/venv/bin/python -u $W/stock_fa3_unsplit_work_20261014/load_stock_fa3_variant.py > build_fa3_$v.log 2>&1
  [ -f /opt/tiger/w/torch_ext_fa3_$v/stock_fa3_pdl/stock_fa3_pdl.so ] || { tail -20 build_fa3_$v.log; echo BUILD_FAILED $v; exit 1; }
  if [ $v = skip ]; then grep -q -- "-DFLASHINFER_FA3_SKIP_KV_LOAD=1" build_fa3_$v.log || { echo FLAG_MISSING; exit 1; }; fi
  echo "built fa3 $v"
done
export PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi CUDA_VISIBLE_DEVICES=0
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry UNIFIED_SO=/opt/tiger/w/torch_ext_unified_np_u/unified_dispatch_ext/unified_dispatch_ext.so
B=/opt/tiger/w/trace_sampled_h20_20261024; OUT=/opt/tiger/w/unified_h20_20261024
for v in ctrl skip; do
  bound=0; [ $v = skip ] && bound=1
  for d in mix32_two mix32_four mix128_two mix128_four; do
    NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_fa3_$v/stock_fa3_pdl/stock_fa3_pdl.so UNIFIED_TIMING_BOUND=$bound \
      /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json \
      --unified-only --append --label u_np_u_fa3$v 2>&1 | grep -E "FAIL|rror|Traceback"
  done
done
echo "control rebuild vs the serving library (same A6 build, u_np_u_tail_merge_unroll):"
/dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_np_u_fa3ctrl --baseline arm:u_np_u_tail_merge_unroll $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
echo "zero-KV-load bound vs control (speedup = most multicast could give):"
/dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_np_u_fa3skip --baseline arm:u_np_u_fa3ctrl $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
echo FA3SKIP_DONE
