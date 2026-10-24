#!/bin/bash
set -euo pipefail
ROOT=${ROOT:-/dev/shm/w/fp32_adjudication_h200_20261023}
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test ! -d "$ROOT/python"
cp -a /dev/shm/w/up/python "$ROOT/python"
/dev/shm/w/venv/bin/python "$ROOT/apply_native_pdl_fa3_backend_20261023.py" \
  "$ROOT/python/sglang/srt/layers/attention/flashattention_backend.py"
/dev/shm/w/venv/bin/python "$ROOT/apply_fp32_adjudication_20261023.py" \
  "$ROOT/python/sglang/srt/layers/attention/flashattention_backend.py"
/dev/shm/w/venv/bin/python "$ROOT/apply_native_pdl_model_shadow_20261023.py" \
  "$ROOT/python/sglang/srt/speculative/eagle_worker_v2.py"
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export PYTHONPATH=$ROOT/python:$ROOT:/dev/shm/w/fi
export HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export SGLANG_NATIVE_PDL=1 SGLANG_NATIVE_PDL_VERIFY=1
export SGLANG_NATIVE_PDL_SPLIT_POLICY=${SGLANG_NATIVE_PDL_SPLIT_POLICY:-geometry_v2}
export SGLANG_MIXED_SPEC_MIN_CTX=8192 SGLANG_MIXED_SPEC_MIN_QUEUE=4
export NATIVE_PDL_MODEL_SHADOW=$ROOT/shadow.jsonl NATIVE_PDL_FP32_ADJUDICATE=1
export FP32_ADJUDICATION_ROOT=$ROOT
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_TOKEN_MODE SGLANG_SIMULATE_ACC_METHOD
unset SGLANG_NATIVE_PDL_ORIGINAL_ROW_TILE
export SRV_ARGS="--attention-backend fa3 --chunked-prefill-size 512 --disable-cuda-graph"
cd "$ROOT"
/dev/shm/w/venv/bin/python -m unittest test_fp32_adjudication_20261023 -v > "$ROOT/reference_tests.log" 2>&1
/dev/shm/w/venv/bin/python -u run_fp32_adjudication_20261023.py
