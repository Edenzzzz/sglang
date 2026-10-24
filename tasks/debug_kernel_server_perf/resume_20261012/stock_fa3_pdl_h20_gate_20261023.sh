#!/bin/bash
set -euo pipefail
ROOT=/dev/shm/w/stock_fa3_pdl_check_20261023
WORK=$ROOT/stock_fa3_unsplit_work_20261014
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export PYTHONPATH=/dev/shm/w/up/python:$WORK
export TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_stock_fa3_pdl_20261023
export STOCK_FA3_MERGED=/opt/tiger/w/stock_fa3_pdl_merged_20261023
export STOCK_FLASH_ATTENTION=$ROOT/flash-attention STOCK_FA3_UNSPLIT_SPECIALIZE=1
mkdir -p "$TORCH_EXTENSIONS_DIR"
cd "$WORK"
/dev/shm/w/venv/bin/python - <<'PY'
import torch

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print("GPU", torch.cuda.get_device_name(0), "SMs", torch.cuda.get_device_properties(0).multi_processor_count, flush=True)
PY
/dev/shm/w/venv/bin/python -u test_stock_fa3.py > "$ROOT/tests.log" 2>&1
cat "$ROOT/tests.log"
grep -q 'Ran 3 tests' "$ROOT/tests.log"
! grep -q skipped "$ROOT/tests.log"
echo STOCK_FA3_PDL_H20_GATE_PASS
