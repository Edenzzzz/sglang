#!/bin/bash
# S4 on the final kernel (H200, 2026-10-25): step traces of a matched four-mode schedule (both arms run the 4-mode
# mixed step; no EDF), stock sglang's single FA3 call vs A6 (SGLANG_A6_UNIFIED=1), two opposite-order repeats; then the
# step model is fitted on repeat 1 and scored on repeat 2 for each arm (evaluate_native_step_model_20261023.py, MLP with
# causal acceptance as the primary model). TraceLab deduplicated as on 2026-10-23. Claude.
set -euo pipefail
SRC=/dev/shm/w/qwen25_v3_common_screen_h200_20261023
R=/opt/tiger/w/a6_s4_steptrace_h200_20261025
# (runs after the load run)
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
mkdir -p $R
test -d $R/python || cp -a $SRC/python $R/python
cp $SRC/*.py $SRC/*.sh $R/
cp /dev/shm/w/ub/native_fa3_fa2_pdl_20261023.py /dev/shm/w/ub/run_native_serving_pair_20261023.py /dev/shm/w/ub/native_serving_protocol_20261023.py \
   /dev/shm/w/ub/audit_native_serving_pair_20261023.py /dev/shm/w/ub/unified_executor_20261024.py \
   /dev/shm/w/ub/evaluate_native_step_model_20261023.py /dev/shm/w/ub/fit_4mode_step.py /dev/shm/w/ub/apply_steptrace_features.py \
   /dev/shm/w/ub/steptrace_features.py $R/
/dev/shm/w/venv/bin/python $R/apply_steptrace_features.py $R/python/sglang/srt
grep -q "STEPTRACE_FEATURES source=request_cpu_v2" $R/python/sglang/srt/managers/scheduler.py
/dev/shm/w/venv/bin/python $R/deduplicate_trace_rounds_20261023.py /dev/shm/w/workloads/tracelab/syfi_full_min.jsonl.gz $R/tracelab_unique.jsonl.gz $R/trace_deduplication.json
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_np2s_stock/unified_dispatch_ext/unified_dispatch_ext.so UNIFIED_FA3_STOCK=1 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_MIN=33
export PYTHONPATH=$R/python:$R:/dev/shm/w/fi:/dev/shm/w/drivers/serving TRACE_DATASET_PATH=$R/tracelab_unique.jsonl.gz
cd $R
/dev/shm/w/venv/bin/python -u run_native_serving_pair_20261023.py $R --stage mix32 --modes four --step-trace --direct-output \
  --split-policy geometry_v3 --packed-query-limit 64 --concurrency 8 --sessions 16 --chat-rate 8 --chat-requests 960 --repeats 2 \
  --native-env SGLANG_A6_UNIFIED=1
/dev/shm/w/venv/bin/python audit_native_serving_pair_20261023.py $R $R/audit.json || echo "AUDIT_FOUR_MODE_ONLY (the audit expects both modes; checked by the composition count below)"
n=$(cat $R/*_native/server.log | grep -c "a6_unified=1" || true); echo "a6 composition: $n"; [ "$n" -gt 0 ] || { echo A6_NOT_ACTIVE; exit 1; }
/dev/shm/w/venv/bin/python evaluate_native_step_model_20261023.py $R $R/step_model_eval.json
/dev/shm/w/venv/bin/python - $R/step_model_eval.json <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for x in r["results"]:
    print({k: x[k] for k in x if k in ("arm", "model", "train_repeat", "test_repeat")}, {k: v for k, v in x.get("scores", {}).get("all", {}).items() if k in ("mape", "median_ape", "p90_ape", "samples")})
PY
echo A6_S4_DONE
