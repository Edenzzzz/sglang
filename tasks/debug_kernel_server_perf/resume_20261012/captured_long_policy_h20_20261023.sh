#!/bin/bash
set -euo pipefail
ROOT=/dev/shm/w/captured_long_policy_20261023
export ROOT
echo $$ > "$ROOT/chain.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/chain_exit_status"' EXIT
bash "$ROOT/native_pdl_combined_h20_gate_20261023.sh" > "$ROOT/combined_runner.log" 2>&1
mv "$ROOT/controller.pid" "$ROOT/combined_controller.pid"
mv "$ROOT/exit_status" "$ROOT/combined_exit_status"
export CAPTURE_GEOMETRY_POLICY=1
export CAPTURE_STAGE_ARGS="--stages mix128 fanout384"
bash "$ROOT/captured_mixed_h20_20261023.sh" > "$ROOT/capture_runner.log" 2>&1
echo CAPTURED_LONG_POLICY_MATRIX_DONE
