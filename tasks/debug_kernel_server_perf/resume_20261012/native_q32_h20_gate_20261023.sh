#!/bin/bash
set -euo pipefail
ROOT=/dev/shm/w/native_q32_check_20261023
WORK=$ROOT/native_fa2_q32_v2_work_20261015
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export PYTHONPATH=/dev/shm/w/up/python:$WORK
export TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_native_q32_20261023
export NATIVE_Q32_MERGED=/opt/tiger/w/native_q32_merged_20261023
mkdir -p "$TORCH_EXTENSIONS_DIR"
cd "$WORK"
/dev/shm/w/venv/bin/python - <<'PY'
import torch

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print("GPU", torch.cuda.get_device_name(0), "SMs", torch.cuda.get_device_properties(0).multi_processor_count, flush=True)
print("TORCH", torch.__version__, "CUDA", torch.version.cuda, flush=True)
PY
/dev/shm/w/venv/bin/python -u test_native_q32.py > "$ROOT/tests.log" 2>&1
cat "$ROOT/tests.log"
grep -q NATIVE_Q32_EXECUTED_GEOMETRY "$ROOT/tests.log"
echo NATIVE_Q32_H20_GATE_PASS
