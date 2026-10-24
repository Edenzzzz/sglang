#!/bin/bash
set -euo pipefail
ROOT=/dev/shm/w/native_kernel_sources_20261023
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
NATIVE=$ROOT/native_fa2_q32_v2_work_20261015
STOCK=$ROOT/stock_fa3_unsplit_work_20261014
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export PYTHONPATH=/dev/shm/w/up/python:$ROOT:$NATIVE:$STOCK
export NATIVE_Q32_MERGED=/opt/tiger/w/native_q32_merged_h200_20261023
export STOCK_FA3_MERGED=/opt/tiger/w/stock_fa3_merged_h200_20261023
export STOCK_FLASH_ATTENTION=$ROOT/flash-attention STOCK_FA3_UNSPLIT_SPECIALIZE=1
export NATIVE_EXTENSIONS=/opt/tiger/w/torch_ext_native_h200_20261023
export STOCK_EXTENSIONS=/opt/tiger/w/torch_ext_stock_h200_20261023
export NATIVE_PDL_FA2_LIBRARY=$NATIVE_EXTENSIONS/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=$STOCK_EXTENSIONS/stock_fa3_pdl/stock_fa3_pdl.so
mkdir -p "$NATIVE_EXTENSIONS" "$STOCK_EXTENSIONS"
cd "$ROOT"
/dev/shm/w/venv/bin/python -u - <<'PY'
import os
import torch

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.get_device_name(0) == 'NVIDIA H200'
print('GPU', torch.cuda.get_device_name(0), 'SMS', torch.cuda.get_device_properties(0).multi_processor_count, flush=True)
os.environ['TORCH_EXTENSIONS_DIR'] = os.environ['NATIVE_EXTENSIONS']
from load_native_q32 import build as build_native
build_native()
os.environ['TORCH_EXTENSIONS_DIR'] = os.environ['STOCK_EXTENSIONS']
from load_stock_fa3 import build as build_stock
build_stock()
PY
TORCH_EXTENSIONS_DIR=$NATIVE_EXTENSIONS /dev/shm/w/venv/bin/python -u "$NATIVE/test_native_q32.py" > native_tests.log 2>&1
TORCH_EXTENSIONS_DIR=$STOCK_EXTENSIONS /dev/shm/w/venv/bin/python -u "$STOCK/test_stock_fa3.py" > stock_tests.log 2>&1
/dev/shm/w/venv/bin/python -u test_native_pdl_combined_20261023.py > combined_tests.log 2>&1
grep -q 'Ran 5 tests' native_tests.log
grep -q 'Ran 3 tests' stock_tests.log
grep -q 'Ran 5 tests' combined_tests.log
! grep -q skipped native_tests.log stock_tests.log combined_tests.log
/dev/shm/w/venv/bin/python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import torch

paths = {name: Path(os.environ[name]) for name in ('NATIVE_PDL_FA2_LIBRARY', 'NATIVE_PDL_FA3_LIBRARY')}
report = dict(gpu=torch.cuda.get_device_name(0), sms=torch.cuda.get_device_properties(0).multi_processor_count,
              torch=torch.__version__, libraries={name: dict(path=str(path), sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()) for name, path in paths.items()})
Path('environment.json').write_text(json.dumps(report, indent=2) + '\n')
print('H200_NATIVE_KERNEL_GATES_PASS', json.dumps(report), flush=True)
PY
