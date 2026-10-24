#!/bin/bash
set -euo pipefail
ROOT=/dev/shm/w/combined_h200_retry_20261023
SOURCE=/dev/shm/w/native_kernel_sources_20261023
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/dev/shm/w/up/python:$ROOT:$SOURCE:$SOURCE/native_fa2_q32_v2_work_20261015
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
cd "$ROOT"
/dev/shm/w/venv/bin/python -u test_native_pdl_combined_20261023.py > combined_tests.log 2>&1
cat combined_tests.log
grep -q 'Ran 5 tests' combined_tests.log
! grep -q skipped combined_tests.log
/dev/shm/w/venv/bin/python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import torch

source = Path('/dev/shm/w/native_kernel_sources_20261023')
for name, count in (('native_tests.log', 5), ('stock_tests.log', 3)):
    log = (source / name).read_text()
    assert f'Ran {count} tests' in log and '\nOK\n' in log and 'FAILED' not in log and 'skipped' not in log
assert torch.cuda.get_device_name(0) == 'NVIDIA H200'
paths = {name: Path(os.environ[name]) for name in ('NATIVE_PDL_FA2_LIBRARY', 'NATIVE_PDL_FA3_LIBRARY')}
report = dict(gpu=torch.cuda.get_device_name(0), sms=torch.cuda.get_device_properties(0).multi_processor_count,
              torch=torch.__version__, native_tests=5, producer_tests=3, combined_tests=5,
              libraries={name: dict(path=str(path), sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()) for name, path in paths.items()})
Path('environment.json').write_text(json.dumps(report, indent=2) + '\n')
print('H200_ALL_13_NATIVE_GATES_PASS', json.dumps(report), flush=True)
PY
