#!/bin/bash
# Kernel ladder on the 4-GPU H200 trial (d4b20cad59170ebd), 2026-10-26: the kernel claim against NAIVE FA3 + FA2 (user).
# 1. Re-capture the trace-sampled Qwen2.5-7B steps (real 4-mode MTP serving; 24 per context x mode cell).
# 2. Re-bench on them: sglang's single FA3 call (default), the FA3 role split, and our two-plan build (native v3).
# 3. Build the final serving dispatcher (A6 np64: FA2 classes 64 / 32 / 16, FA3 for rows > 64 packed queries, stock FA3
#    at the plan's split, ticketed in-launch merge), check FP32 + 20 graph replays on every capture, bench it.
# 4. Naive FA3 + FA2 (stock FA3 on the prefill rows + stock flashinfer FA2 from the pip release, flashinfer's own plan,
#    tile and merge; serial and two-stream) in a separate process whose PYTHONPATH has no port07 tree.
# 5. Time-weighted speedups of A6 over each baseline, and of the naive arms over sglang's call.
# Claude.
set -u
cd /dev/shm/w/ub
LOG=/opt/tiger/w/kernel_ladder_h200x4_20261026; mkdir -p $LOG
sed -e "s#ROOT=/opt/tiger/w/trace_sampled_h20_20261024#ROOT=/opt/tiger/w/trace_sampled_h200x4_20261026#" \
  trace_sampled_capture_h20_20261024.sh > trace_sampled_capture_h200x4_20261026.sh
B=/opt/tiger/w/trace_sampled_h200x4_20261026
if ! ls $B/mix128_four/captures/*.pt >/dev/null 2>&1; then
  bash trace_sampled_capture_h200x4_20261026.sh > $LOG/capture.log 2>&1
  grep -q TRACE_SAMPLED_CAPTURE_DONE $LOG/capture.log || { echo CAPTURE_FAILED; tail -20 $LOG/capture.log; exit 1; }
fi
echo "captures $(ls $B/*/captures/*.pt | wc -l)"
sed -e "s#/opt/tiger/w/trace_sampled_h200_20261025#$B#g" rebench_trace_sampled_v3_h200_20261025.sh > rebench_v3_h200x4_20261026.sh
bash rebench_v3_h200x4_20261026.sh > $LOG/rebench_v3.log 2>&1
grep -E "time-weighted|REBENCH" $LOG/rebench_v3.log | sed -E 's/ \|.*//'
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16 VERBOSE=1 UNIFIED_FA2_CLASSES=3 UNIFIED_FA2_DYNAMIC=1 UNIFIED_TILES=64,32,16
export PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
name=np64_h200x4
SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
if [ ! -f $SO ]; then
  UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name \
    /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > $LOG/build_$name.log 2>&1
  grep -q UNIFIED_BUILD_OK $LOG/build_$name.log || { grep " error" $LOG/build_$name.log | head; echo BUILD_FAILED; exit 1; }
fi
for w in UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 UNIFIED_FA2_CLASSES=3; do
  grep nvcc $LOG/build_$name.log | grep -q -- "-D$w" || { echo FLAG_MISSING $w; exit 1; }
done
echo "build $name ok $(md5sum < $SO | cut -c1-12)"
export CUDA_VISIBLE_DEVICES=0
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=65 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_SO=$SO
OUT=/opt/tiger/w/unified_h200x4_20261026; mkdir -p $OUT
for d in mix32_two mix32_four mix128_two mix128_four; do
  mkdir -p $OUT/$d; [ -f $OUT/$d/kernel_results.json ] || cp $B/$d/kernel_results_v3.json $OUT/$d/kernel_results.json
  UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > $LOG/check_$d.log 2>&1
  echo "check $d: FAIL $(grep -v GRAPH_FAILS $LOG/check_$d.log | grep -c FAIL), PASS $(grep -c ' PASS' $LOG/check_$d.log), graph fails $(grep GRAPH_FAILS $LOG/check_$d.log | awk '{s+=$2} END {print s+0}'), err $(grep -cE 'Error|Traceback' $LOG/check_$d.log)"
  /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json \
    --unified-only --append --label u_np64 > $LOG/bench_u_$d.log 2>&1
  grep -E "FAIL|rror|Traceback" $LOG/bench_u_$d.log | head -3
  # Naive FA3 + FA2: pip flashinfer only (no port07 on the path).
  PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python /dev/shm/w/venv/bin/python -u bench_naive_fa3_fa2_20261026.py \
    $B/$d/captures $OUT/$d/kernel_results.json > $LOG/bench_naive_$d.log 2>&1
  grep -E "NAIVE_BENCH_DONE|rror|Traceback|not the stock" $LOG/bench_naive_$d.log | head -3
done
for b in arm:naive_fa3_fa2_serial arm:naive_fa3_fa2_streams default arm:native_pdl_geometry_v3 arm:fa3_dual_split32; do
  echo "== u_np64 vs $b"
  /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_np64 --baseline $b $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
done
for a in naive_fa3_fa2_serial naive_fa3_fa2_streams; do
  echo "== $a vs default (sglang's single FA3 call)"
  /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py $a --baseline default $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
done
echo KERNEL_LADDER_DONE
