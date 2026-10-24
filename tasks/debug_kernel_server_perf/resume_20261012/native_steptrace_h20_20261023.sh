#!/bin/bash
set -euo pipefail
ROOT=/dev/shm/w/native_steptrace_h20_20261023
export ROOT
test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES=0 SERVING_TRACE_FEATURES=1
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_fa3_pdl_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_q32_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export TRACE_DATASET_PATH=$ROOT/tracelab_unique.jsonl.gz
/dev/shm/w/venv/bin/python "$ROOT/deduplicate_trace_rounds_20261023.py" /dev/shm/w/workloads/tracelab/syfi_full_min.jsonl.gz "$TRACE_DATASET_PATH" "$ROOT/trace_deduplication.json"
export SERVING_PAIR_ARGS='--modes four --step-trace --direct-output --split-policy geometry_v3 --concurrency 8 --sessions 16 --chat-rate 8 --chat-requests 960'
exec bash "$ROOT/native_serving_pair_h200_20261023.sh"
