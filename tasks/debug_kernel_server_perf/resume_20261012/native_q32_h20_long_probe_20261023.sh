#!/bin/bash
set -euo pipefail
ROOT=/dev/shm/w/native_q32_check_20261023
WORK=$ROOT/native_fa2_q32_v2_work_20261015
PROBE_TAG=${PROBE_TAG:-probe}
PROBE_OUTPUT=${PROBE_OUTPUT:-long_mix.json}
echo $$ > "$ROOT/${PROBE_TAG}_controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/${PROBE_TAG}_exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
grep -q NATIVE_Q32_H20_GATE_PASS "$ROOT/runner.log"
test ! -e "$ROOT/$PROBE_OUTPUT"
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export PYTHONPATH=/dev/shm/w/up/python:$WORK
export TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_native_q32_20261023
export NATIVE_Q32_MERGED=/opt/tiger/w/native_q32_merged_20261023
EXTRA=()
[ "${FA3_CONTROLS:-0}" != 1 ] || EXTRA+=(--fa3-controls)
/dev/shm/w/venv/bin/python -u "$ROOT/native_fa2_long_mix_20261023.py" "$ROOT/$PROBE_OUTPUT" "${EXTRA[@]}"
