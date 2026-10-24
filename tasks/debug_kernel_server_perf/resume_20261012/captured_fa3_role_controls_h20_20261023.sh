#!/bin/bash
set -euo pipefail
ROOT=/dev/shm/w/captured_fa3_role_controls_20261023
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export PYTHONPATH=/dev/shm/w/up/python:$ROOT
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_fa3_pdl_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_q32_20261023/native_fa2_q32/native_fa2_q32.so
test "$(cat /dev/shm/w/captured_long_policy_20261023/chain_exit_status)" = 0
cd "$ROOT"
/dev/shm/w/venv/bin/python -u bench_fa3_role_controls_20261023.py "$ROOT/results.json" \
  /dev/shm/w/captured_pair_tuning_20261023/joined /dev/shm/w/captured_long_policy_20261023
