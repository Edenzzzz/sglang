#!/bin/bash
set -euo pipefail
ROOT=/dev/shm/w/captured_pair_tuning_20261023
export ROOT
echo $$ > "$ROOT/chain.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/chain_exit_status"' EXIT
export CAPTURE_STAGE_ARGS="--stages mix32 --modes four"
bash "$ROOT/captured_mixed_h20_20261023.sh" > "$ROOT/capture_runner.log" 2>&1
mv "$ROOT/controller.pid" "$ROOT/capture_controller.pid"
mv "$ROOT/exit_status" "$ROOT/capture_exit_status"
bash "$ROOT/native_pdl_combined_h20_gate_20261023.sh" > "$ROOT/combined_runner.log" 2>&1
mv "$ROOT/controller.pid" "$ROOT/combined_controller.pid"
mv "$ROOT/exit_status" "$ROOT/combined_exit_status"
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export PYTHONPATH=/dev/shm/w/up/python:$ROOT:/dev/shm/w/native_q32_check_20261023/native_fa2_q32_v2_work_20261015
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_fa3_pdl_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_q32_20261023/native_fa2_q32/native_fa2_q32.so
mkdir "$ROOT/joined"
ln -s /dev/shm/w/captured_mixed_v2_20261023/mix32_two "$ROOT/joined/mix32_two"
ln -s "$ROOT/mix32_four" "$ROOT/joined/mix32_four"
cd "$ROOT"
/dev/shm/w/venv/bin/python -u tune_captured_mixed_20261023.py "$ROOT/joined" "$ROOT/tuned" > "$ROOT/tuning.log" 2>&1
echo CAPTURED_PAIR_TUNING_DONE
