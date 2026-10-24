#!/bin/bash
set -euo pipefail
export ROOT=/dev/shm/w/qwen3_captured_20261023
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
test ! -d "$ROOT/python"
cp -a /dev/shm/w/up/python "$ROOT/python"
/dev/shm/w/venv/bin/python "$ROOT/apply_capture_native_fa3_20261023.py" "$ROOT/python/sglang/srt/layers/attention/flashattention_backend.py"
cp "$ROOT/mn_bench.sh" "$ROOT/mn_bench_capture.sh"
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export PYTHONPATH=$ROOT/python:$ROOT:/dev/shm/w/fi
export HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_fa3_pdl_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_q32_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
cd "$ROOT"
sha256sum native_fa3_fa2_pdl_20261023.py /dev/shm/w/qwen3_8b/config.json /dev/shm/w/qwen3_8b_eagle3/config.json > frozen_before_capture.sha256
/dev/shm/w/venv/bin/python -u capture_serving_workloads_20261023.py "$ROOT" --stages mix32 --model-dir qwen3_8b --draft-dir qwen3_8b_eagle3 --agent-seed 20261023
/dev/shm/w/venv/bin/python -u bench_fa3_role_controls_20261023.py "$ROOT/role_controls.json" "$ROOT"
/dev/shm/w/venv/bin/python -u bench_fa3_row_packing_20261023.py "$ROOT/validation.json" --prior "$ROOT/role_controls.json" --roots "$ROOT"
echo QWEN3_CAPTURED_NATIVE_MATRIX_DONE
