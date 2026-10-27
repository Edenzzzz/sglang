#!/bin/bash
# Kernel ladder for a SECOND MODEL on the 4-H200 trial, from captures copied off the H20 (2026-10-28).
# The Qwen2.5-7B ladder (a6_h200x4_kernel_ladder_20261026.sh) established A6 at 4.0-8.7% over the naive role split
# (stock FA3 for the chunk + stock flashinfer FA2 for the rows) and 1.86-2.92x over sglang's single varlen call.
# The user's bar covers Llama-3.3-70B and GLM-4.5-Air too, and GLM is the model whose 48-packed-row verify rows the
# FA2 64-row tile class was built for, so it is where the tile dispatch should pay most. Re-capturing needs the 70B
# weights; instead the rank-0 captures taken on the H20 are replayed here, which is what the kernel arms consume.
#   a6_ladder_model_h200_20261028.sh <capture-root> <tag>
# Arms per capture cell: sglang's single FA3 call (default, already in kernel_results_v3.json), naive FA3 + stock
# FA2 (serial and two-stream, pip flashinfer, no port07 on the path), and A6 np64. Claude.
set -u
ROOT=$1; TAG=$2
cd /dev/shm/w/ub
LOG=/opt/tiger/w/ladder_${TAG}_20261028; mkdir -p $LOG
exec 9>/dev/shm/w/a2_kernel.lock; flock 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
export TORCH_CUDA_ARCH_LIST=9.0a PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_np64_h200x4/unified_dispatch_ext/unified_dispatch_ext.so
export UNIFIED_FA3_STOCK=1 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_MIN=65
export UNIFIED_FA2_CLASSES=3 UNIFIED_FA2_DYNAMIC=1 UNIFIED_TILES=64,32,16 NATIVE_PDL_PACKED_QUERY_LIMIT=64
OUT=/opt/tiger/w/ladder_${TAG}_out; mkdir -p $OUT
cells=""
for d in $ROOT/*/; do
  c=$(basename $d); ls $d/captures/*.pt >/dev/null 2>&1 || continue
  cells="$cells $c"; mkdir -p $OUT/$c
  [ -f $OUT/$c/kernel_results.json ] || cp $d/kernel_results_v3.json $OUT/$c/kernel_results.json 2>/dev/null || \
    { echo "no v3 results for $c, skipping"; continue; }
  UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $d/captures/*.pt > $LOG/check_$c.log 2>&1
  echo "check $c: FAIL $(grep -v GRAPH_FAILS $LOG/check_$c.log | grep -c FAIL), PASS $(grep -c ' PASS' $LOG/check_$c.log), err $(grep -cE 'Error|Traceback' $LOG/check_$c.log)"
  /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $d/captures $OUT/$c/kernel_results.json \
    --unified-only --append --label u_np64 > $LOG/bench_u_$c.log 2>&1
  grep -E "FAIL|rror|Traceback" $LOG/bench_u_$c.log | head -2
  PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python /dev/shm/w/venv/bin/python -u bench_naive_fa3_fa2_20261026.py \
    $d/captures $OUT/$c/kernel_results.json > $LOG/bench_naive_$c.log 2>&1
  grep -E "NAIVE_BENCH_DONE|rror|Traceback|not the stock" $LOG/bench_naive_$c.log | head -2
done
echo "cells:$cells"
for b in arm:naive_fa3_fa2_serial arm:naive_fa3_fa2_streams default; do
  echo "== $TAG u_np64 vs $b"
  /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_np64 --baseline $b $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
done
echo LADDER_${TAG}_DONE
