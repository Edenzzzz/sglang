#!/bin/bash
set -euo pipefail
: "${ROOT:?Set an isolated gate root}"
: "${NATIVE_Q32_SOURCE_ROOT:?Set the native source root}"
: "${NATIVE_PDL_FA2_LIBRARY:?Set the native FA2 library}"
: "${NATIVE_PDL_FA3_LIBRARY:?Set the native FA3 library}"
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/dev/shm/w/up/python:$ROOT:$NATIVE_Q32_SOURCE_ROOT
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export NATIVE_WORKSPACE_GPU_TEST=1
cd "$ROOT"
/dev/shm/w/venv/bin/python -u -m unittest test_native_workspace_growth_20261023 test_native_pdl_partition_20261023 -v > tests.log 2>&1
grep -q 'Ran 14 tests' tests.log
! grep -q skipped tests.log
echo NATIVE_WORKSPACE_GROWTH_GATE_PASS
