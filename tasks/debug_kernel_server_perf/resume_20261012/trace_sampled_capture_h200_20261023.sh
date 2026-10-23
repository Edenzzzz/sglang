#!/bin/bash
# Trace-PROPORTIONAL kernel microbenchmark on the H200 (job 34c7bd503d461dc1), 2026-10-23. Claude, picking up from
# the Codex kernel_test session.
# Why: the stage-1 microbenchmark captured the FIRST eligible step per shape bucket, so its per-bucket speedups
# (1.5-1.8x headlines) over-weight buckets that are <1% of real mixed steps; weighted by the census the kernel is worth
# 0.975-1.13x of mixed-step attention time (trace_weighted_kernel_20261023.py). That estimate still uses one capture
# per bucket, and it contradicts stage 2 (four-mode 128-256K: kernel 1.03x vs serving +10.6% throughput). This
# replaces it with a systematic sample: CAPTURE_EVERY=N saves every Nth census step regardless of bucket (capped at
# 24), so the saved set has the trace's own shape distribution, for two-mode and four-mode, at 32-64K and 128-256K.
# Same H200 and frozen tree as the stage-2 serving runs, so stages 1 and 2 share hardware. The driver's own
# bench_captured_mixed then times native (geometry policy) against single- and dual-stream FA3 on every capture,
# with its FP32 reference check -- a lighter control set than the H20 SM-grid search, stated as such.
set -euo pipefail
SRC=/dev/shm/w/qwen25_v3_common_screen_h200_20261023
ROOT=/opt/tiger/w/trace_sampled_h200_20261023
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
mkdir -p "$ROOT"
echo $$ > "$ROOT/controller.pid"
trap 'printf "%s\n" "$?" > "$ROOT/exit_status"' EXIT
test -d "$ROOT/python" || cp -a "$SRC/python" "$ROOT/python"
cp "$SRC"/*.py "$SRC"/mn_bench.sh "$SRC"/tracelab_unique.jsonl.gz "$ROOT/"
cp /dev/shm/w/perf/tsc/*.py "$ROOT/"          # capture hook with CAPTURE_EVERY, capture driver, bench, patch
cp "$ROOT/mn_bench.sh" "$ROOT/mn_bench_capture.sh"
/dev/shm/w/venv/bin/python "$ROOT/apply_capture_native_fa3_20261023.py" "$ROOT/python/sglang/srt/layers/attention/flashattention_backend.py"
grep -q "capture_mixed_attention_20261023 import capture_mixed_attention" "$ROOT/python/sglang/srt/layers/attention/flashattention_backend.py"
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a
export PYTHONPATH=$ROOT/python:$ROOT:/dev/shm/w/fi HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export NATIVE_PDL_PACKED_QUERY_LIMIT=64 CAPTURE_PACKED_QUERY_LIMIT=64 CAPTURE_RANK=0 CAPTURE_GEOMETRY_POLICY=1
export TRACE_DATASET_PATH=$ROOT/tracelab_unique.jsonl.gz CAPTURE_MAX=24
cd "$ROOT"
sha256sum capture_mixed_attention_20261023.py native_fa3_fa2_pdl_20261023.py bench_captured_mixed_20261023.py > frozen.sha256
# Interval per (stage, mode) from the H20 census sizes, aiming at ~24 evenly spread captures:
# mix32 two ~480 steps -> 20, four ~292 -> 12; mix128 two ~411 -> 17, four ~203 -> 8.
for spec in "mix32 two 20" "mix32 four 12" "mix128 two 17" "mix128 four 8"; do
  set -- $spec
  echo "##### $1 $2 every $3"
  CAPTURE_EVERY=$3 /dev/shm/w/venv/bin/python -u capture_serving_workloads_20261023.py "$ROOT" --stages $1 --modes $2
done
sha256sum -c frozen.sha256
echo TRACE_SAMPLED_CAPTURE_DONE
