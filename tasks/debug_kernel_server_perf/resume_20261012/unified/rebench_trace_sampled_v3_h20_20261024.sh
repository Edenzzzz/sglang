#!/bin/bash
# H20 copy (2026-10-24). After trace_sampled_capture_h20_20261024.sh finishes: re-bench every trace-sampled capture with the v3 arm (the
# configuration the audited serving runs used), then report native v3 vs sglang's default FA3 (deployable) and vs the
# best FA3 variant per step (oracle), per stage/mode. Claude, 2026-10-23.
set -u
ROOT=/opt/tiger/w/trace_sampled_h20_20261024
until [ -f $ROOT/exit_status ]; do sleep 60; done
[ "$(cat $ROOT/exit_status)" = 0 ] || { echo "capture exit $(cat $ROOT/exit_status)"; exit 1; }
exec 9>/dev/shm/w/a2_kernel.lock; flock -n 9
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$ROOT/python:$ROOT:/dev/shm/w/fi HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export NATIVE_PDL_PACKED_QUERY_LIMIT=64
cp /dev/shm/w/perf/rss/bench_captured_mixed_20261023.py $ROOT/bench_captured_mixed_v3_20261023.py
cd $ROOT
for d in mix32_two mix32_four mix128_two mix128_four; do
  [ -d $d/captures ] || continue
  /dev/shm/w/venv/bin/python -u bench_captured_mixed_v3_20261023.py $d/captures $d/kernel_results_v3.json \
    --geometry-policy --policy-v3 > $d/kernel_bench_v3.log 2>&1 || { echo "bench failed $d"; tail -5 $d/kernel_bench_v3.log; continue; }
  for b in default best; do
    /dev/shm/w/venv/bin/python /dev/shm/w/perf/rss/analyze_trace_sampled_20261023.py native_pdl_geometry_v3 --baseline $b $d/kernel_results_v3.json
  done
done
echo REBENCH_V3_DONE
