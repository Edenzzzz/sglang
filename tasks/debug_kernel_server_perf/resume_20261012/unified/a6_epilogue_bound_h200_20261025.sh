#!/bin/bash
# B1 re-bound on the current kernel (A6), H200: one stage-2 pair (mix32 four-mode, agent C=8 + chat 8 req/s; stock FA3 vs
# ours on A6), torch profiler over 60 steps of each arm 150 s into its run (GPU activity), kernel time bucketed by
# parse_kernel_tail.py. The small-op buckets (sampling / verify, index / copy, elementwise / norm) bound what fusing the
# draft / verify epilogue could save. Claude, 2026-10-25.
set -u
SRC=/opt/tiger/w/a6_plan_ab_h200_20261025/plan_new
P=/opt/tiger/w/a6_epilogue_bound_h200_20261025; mkdir -p $P
R=$P/root; mkdir -p $R
test -d $R/python || cp -a $SRC/python $R/python
cp $SRC/*.py $SRC/*.sh $SRC/tracelab_unique.jsonl.gz $R/
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_h200_np_u/unified_dispatch_ext/unified_dispatch_ext.so
export UNIFIED_FA3_STOCK=1 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_MIN=33
export PYTHONPATH=$R/python:$R:/dev/shm/w/fi:/dev/shm/w/drivers/serving TRACE_DATASET_PATH=$R/tracelab_unique.jsonl.gz
FIX="--ours-env SGLANG_TTFT_SLO_MS=500 --ours-env SGLANG_EDF_LONG_PROMPT_TOKENS=8192 --ours-env SGLANG_EDF_LONG_TTFT_SLO_MS=10000 --ours-env SGLANG_EDF_LATE_DEMOTE=1 --ours-env SGLANG_EDF_LATE_DEMOTE_TOK_S=1e12"
cd $R
( # profile each arm once its server is up and 150 s into the load
  for arm in fa3 native; do
    d=$R/mix32_four_r1_$arm
    for i in $(seq 1 600); do [ -f $d/protocol.json ] && grep -q "fired up" $d/server.log 2>/dev/null && break; sleep 5; done
    port=$(/dev/shm/w/venv/bin/python -c "import json,sys; c=json.load(open('$d/protocol.json'))['command']; print(c[c.index('--port')+1])")
    sleep 150
    mkdir -p $P/$arm
    curl -s -m 60 -X POST http://127.0.0.1:$port/start_profile -H 'Content-Type: application/json' \
      -d "{\"output_dir\": \"$P/$arm\", \"num_steps\": 60, \"activities\": [\"GPU\"]}" > $P/start_$arm.json
    echo "profile $arm port $port: $(head -c 150 $P/start_$arm.json)"
  done ) &
/dev/shm/w/venv/bin/python -u run_native_serving_pair_20261023.py "$R" --stage mix32 --modes four --repeats 1 --split-policy geometry_v3 \
  --direct-output --packed-query-limit 64 --ours $FIX --native-env SGLANG_A6_UNIFIED=1 2>&1 | grep -E "NATIVE_SERVING_ARM_DONE|Error|Traceback" | cut -c1-300
wait
for arm in fa3 native; do
  echo "== $arm traces: $(ls $P/$arm | wc -l)"
  /dev/shm/w/venv/bin/python /dev/shm/w/drivers/serving/parse_kernel_tail.py $P/$arm/*.json* 2>&1 | tail -14
done
echo EPI_BOUND_DONE
