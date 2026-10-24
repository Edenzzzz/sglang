#!/bin/bash
set -euo pipefail
export ROOT=/dev/shm/w/fp32_partials_captured_20261023
export NATIVE_Q32_SOURCE_ROOT=/dev/shm/w/native_q32_check_20261023/native_fa2_q32_v2_work_20261015
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_q32_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_fa3_pdl_20261023/stock_fa3_pdl/stock_fa3_pdl.so
source "$ROOT/build_fp32_partials_h200_20261023.sh"
/dev/shm/w/venv/bin/python -u bench_fp32_partials_captured_20261023.py "$ROOT/results.json" > "$ROOT/bench.log" 2>&1
