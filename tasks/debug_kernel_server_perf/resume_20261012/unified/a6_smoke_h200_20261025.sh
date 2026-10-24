#!/bin/bash
# A6 serving smoke test (H200): one short four-mode pair, A6 as the native arm; checks activation and the audit.
set -u
SRC=/dev/shm/w/qwen25_v3_common_screen_h200_20261023; R=/opt/tiger/w/a6_smoke_h200_20261025
rm -rf $R; mkdir -p $R; cp -a $SRC/python $R/python; cp $SRC/*.py $SRC/*.sh $SRC/tracelab_unique.jsonl.gz $R/
cp /dev/shm/w/ub/native_fa3_fa2_pdl_20261023.py /dev/shm/w/ub/run_native_serving_pair_20261023.py /dev/shm/w/ub/native_serving_protocol_20261023.py /dev/shm/w/ub/audit_native_serving_pair_20261023.py /dev/shm/w/ub/unified_executor_20261024.py $R/
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_np2s_stock/unified_dispatch_ext/unified_dispatch_ext.so UNIFIED_FA3_STOCK=1 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_MIN=33
export PYTHONPATH=$R/python:$R:/dev/shm/w/fi:/dev/shm/w/drivers/serving TRACE_DATASET_PATH=$R/tracelab_unique.jsonl.gz
cd $R
/dev/shm/w/venv/bin/python -u run_native_serving_pair_20261023.py $R --stage mix32 --modes four --concurrency 2 --sessions 2 --chat-rate 4 --chat-requests 64 --repeats 1 --split-policy geometry_v3 --direct-output --packed-query-limit 64 --ours --native-env SGLANG_A6_UNIFIED=1 > smoke.log 2>&1
echo "pair rc=$?"; tail -3 smoke.log
echo "a6 lines: $(grep -c 'a6_unified=1' $R/*_native/server.log) plans: $(grep -c A6_UNIFIED_PLAN $R/*_native/server.log) tracebacks: $(grep -c Traceback $R/*_native/server.log) errors: $(grep -ciE 'cuda error|illegal' $R/*_native/server.log)"
grep -m2 "A6_UNIFIED\|a6_unified" $R/*_native/server.log | cut -c1-250
echo A6_SMOKE_DONE
