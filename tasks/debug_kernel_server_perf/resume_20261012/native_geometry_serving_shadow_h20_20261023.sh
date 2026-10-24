#!/bin/bash
set -euo pipefail
export ROOT=/dev/shm/w/native_geometry_serving_shadow_20261023
export SGLANG_NATIVE_PDL_SPLIT_POLICY=geometry_v1
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_TOKEN_MODE SGLANG_SIMULATE_ACC_METHOD
bash "$ROOT/native_pdl_model_shadow_h20_20261023.sh"
