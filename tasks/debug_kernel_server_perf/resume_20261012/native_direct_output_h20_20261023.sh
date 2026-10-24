#!/bin/bash
set -euo pipefail
ROOT=/dev/shm/w/native_direct_output_20261023
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export NATIVE_Q32_SOURCE_ROOT=/dev/shm/w/native_q32_check_20261023/native_fa2_q32_v2_work_20261015
export PYTHONPATH=/dev/shm/w/up/python:$ROOT:$NATIVE_Q32_SOURCE_ROOT
export TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_native_fp32_direct_20261023
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_fa3_pdl_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_q32_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=$TORCH_EXTENSIONS_DIR/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
cd "$ROOT"
/dev/shm/w/venv/bin/python -u load_native_fp32_partials_20261023.py "$NATIVE_Q32_SOURCE_ROOT" "$ROOT/generated" --direct-output > build.log 2>&1
/dev/shm/w/venv/bin/python -u -m unittest test_native_direct_output_20261023 -v > tests.log 2>&1
grep -q 'Ran 4 tests' tests.log
! grep -q skipped tests.log
sha256sum "$NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY" > library.sha256
for attempt in $(seq 1 90); do
  test ! -f /dev/shm/w/qwen3_captured_20261023/raw_upload.pid && break
  test -f /dev/shm/w/qwen3_captured_20261023/raw_upload.exit && break
  sleep 10
done
if test -f /dev/shm/w/qwen3_captured_20261023/raw_upload.pid; then
  test "$(cat /dev/shm/w/qwen3_captured_20261023/raw_upload.exit)" = 0
fi
/dev/shm/w/venv/bin/python -u bench_direct_output_captured_20261023.py "$ROOT/results.json" > bench.log 2>&1
echo NATIVE_DIRECT_OUTPUT_MATRIX_DONE
