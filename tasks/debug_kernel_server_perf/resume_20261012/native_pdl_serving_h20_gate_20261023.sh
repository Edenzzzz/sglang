#!/bin/bash
set -euo pipefail
ROOT=${ROOT:-/dev/shm/w/native_pdl_serving_20261023}
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test ! -d "$ROOT/python"
cp -a /dev/shm/w/up/python "$ROOT/python"
/dev/shm/w/venv/bin/python "$ROOT/apply_native_pdl_fa3_backend_20261023.py" \
  "$ROOT/python/sglang/srt/layers/attention/flashattention_backend.py"
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export PYTHONPATH=$ROOT/python:$ROOT:/dev/shm/w/fi
export HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_fa3_pdl_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_q32_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_SERVING_ROOT=$ROOT
cd "$ROOT"
/dev/shm/w/venv/bin/python -u native_pdl_serving_gate_20261023.py
