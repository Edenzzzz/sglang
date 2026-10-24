#!/bin/bash
set -euo pipefail
ROOT=/dev/shm/w/native_q64_rows_h200_20261023
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/dev/shm/w/up/python:$ROOT:/dev/shm/w/native_kernel_sources_20261023/native_fa2_q32_v2_work_20261015
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
cd "$ROOT"
/dev/shm/w/venv/bin/python -u -m unittest test_native_pdl_partition_20261023 test_native_q64_rows_20261023 -v > tests.log 2>&1
grep -q 'Ran 12 tests' tests.log
! grep -q skipped tests.log
echo NATIVE_Q64_ROWS_H200_GATES_PASS
