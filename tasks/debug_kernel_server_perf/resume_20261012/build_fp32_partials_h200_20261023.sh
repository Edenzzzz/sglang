#!/bin/bash
set -euo pipefail
ROOT=${ROOT:-/dev/shm/w/native_fp32_partials_20261023}
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export NATIVE_Q32_SOURCE_ROOT=${NATIVE_Q32_SOURCE_ROOT:-/dev/shm/w/native_kernel_sources_20261023/native_fa2_q32_v2_work_20261015}
export PYTHONPATH=/dev/shm/w/up/python:$ROOT:$NATIVE_Q32_SOURCE_ROOT
export TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_native_fp32_20261023
export NATIVE_PDL_FA2_LIBRARY=${NATIVE_PDL_FA2_LIBRARY:-/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so}
export NATIVE_PDL_FA3_LIBRARY=${NATIVE_PDL_FA3_LIBRARY:-/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so}
export NATIVE_PDL_FA2_FP32_LIBRARY=$TORCH_EXTENSIONS_DIR/native_fa2_q32_fp32/native_fa2_q32_fp32.so
cd "$ROOT"
/dev/shm/w/venv/bin/python -u load_native_fp32_partials_20261023.py "$NATIVE_Q32_SOURCE_ROOT" "$ROOT/generated" > build.log 2>&1
/dev/shm/w/venv/bin/python -u -m unittest test_native_fp32_partials_20261023 -v > tests.log 2>&1
grep -q 'Ran 5 tests' tests.log
! grep -q skipped tests.log
sha256sum "$NATIVE_PDL_FA2_FP32_LIBRARY" > library.sha256
echo NATIVE_FP32_PARTIALS_GATES_PASS
