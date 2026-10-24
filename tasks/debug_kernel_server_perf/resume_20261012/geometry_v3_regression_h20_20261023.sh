#!/bin/bash
set -euo pipefail
ROOT=/dev/shm/w/geometry_v3_regression_20261023
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/dev/shm/w/up/python:$ROOT
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_fa3_pdl_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_q32_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
cd "$ROOT"
/dev/shm/w/venv/bin/python -u -m unittest test_native_pdl_partition_20261023 -v > host_tests.log 2>&1
/dev/shm/w/venv/bin/python -u validate_geometry_v3_20261023.py
