#!/bin/bash
set -euo pipefail
ROOT=${ROOT:-/dev/shm/w/geometry_v3_holdout_20261023}
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
test ! -d "$ROOT/python"
cp -a /dev/shm/w/up/python "$ROOT/python"
/dev/shm/w/venv/bin/python "$ROOT/apply_capture_native_fa3_20261023.py" "$ROOT/python/sglang/srt/layers/attention/flashattention_backend.py"
cp "$ROOT/mn_bench.sh" "$ROOT/mn_bench_capture.sh"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$ROOT/python:$ROOT:/dev/shm/w/fi
export HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_fa3_pdl_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_q32_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
cd "$ROOT"
export TRACE_DATASET_PATH=$ROOT/tracelab_unique.jsonl.gz
/dev/shm/w/venv/bin/python deduplicate_trace_rounds_20261023.py /dev/shm/w/workloads/tracelab/syfi_full_min.jsonl.gz "$TRACE_DATASET_PATH" "$ROOT/trace_deduplication.json"
date -u +%FT%TZ > frozen_before_capture.timestamp
sha256sum native_fa3_fa2_pdl_20261023.py capture_serving_workloads_20261023.py > frozen_before_capture.sha256
/dev/shm/w/venv/bin/python -u -m unittest test_native_pdl_partition_20261023 -v > host_tests.log 2>&1
/dev/shm/w/venv/bin/python -u capture_serving_workloads_20261023.py "$ROOT" --stages mix64 --agent-seed 20261024
sha256sum -c frozen_before_capture.sha256
/dev/shm/w/venv/bin/python -u bench_fa3_role_controls_20261023.py "$ROOT/role_controls.json" "$ROOT"
/dev/shm/w/venv/bin/python -u bench_fa3_row_packing_20261023.py "$ROOT/row_controls.json" --prior "$ROOT/role_controls.json" --roots "$ROOT"
/dev/shm/w/venv/bin/python -u bench_captured_sm_controls_20261023.py "$ROOT/sm_controls.json" --prior "$ROOT/row_controls.json" --roots "$ROOT"
/dev/shm/w/venv/bin/python -u validate_geometry_v3_20261023.py --output "$ROOT/validation.json" --sm-report "$ROOT/sm_controls.json" --prior "$ROOT/row_controls.json" --roots "$ROOT"
sha256sum -c frozen_before_capture.sha256
echo GEOMETRY_V3_NEW_CONTEXT_MATRIX_DONE
