#!/bin/bash
# Stage 2 for Llama-3.3-70B (H20 job 2f608c1b1a67cf09 pod 4, TP4, trained EAGLE3 head), 2026-10-25: our full single-node
# stack (EDF with per-tenant deadlines, demote-on-miss, 4-mode MTP step on the A6 kernel) vs
# stock sglang on FA3. Both arms run the head at a fixed acceptance of 3.0 tokens per verify (the trained head accepts
# ~1.3 on these prompts; Qwen / GLM heads reach 3.0-3.6 in four-mode). mix32 (32-64K agents; 128K native context), load
# near capacity (agent conc 4, chat 2 req/s), two
# opposite-order repeats, audited; the A6 arm must log a6_unified=1. Claude.
set -euo pipefail
W=/dev/shm/w
SRC=$W/s2src_glm; BASE=/opt/tiger/w/a6_stage2_llama_h20_20261025
mkdir -p $SRC $BASE
[ -f $SRC/run_native_serving_pair_20261023.py ] || tar xzf $W/s2src.tgz -C $SRC
cp $W/ub/*.py $SRC/
grep -q common_env $SRC/run_native_serving_pair_20261023.py
test -d $SRC/python || cp -a $W/python $SRC/python
grep -q SGLANG_EDF_LATE_DEMOTE $SRC/python/sglang/srt/managers/scheduler.py
grep -q EDF_TP_CLOCK $SRC/python/sglang/srt/managers/scheduler_components/request_receiver.py  # TP ranks share one EDF clock
grep -q SGLANG_NATIVE_PDL $SRC/python/sglang/srt/layers/attention/flashattention_backend.py
grep -q prepare_unified $SRC/native_fa3_fa2_pdl_20261023.py
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test -f $W/llama70b/config.json && test -f $W/llama70b_eagle3_sgl/config.json
export CUDA_VISIBLE_DEVICES=0,1,2,3 TORCH_CUDA_ARCH_LIST=9.0a HF_HOME=$W/hf FI_ROOT=$W/fi
A=/opt/tiger/w/a6lib
export NATIVE_PDL_FA2_LIBRARY=$A/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=$A/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=$A/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=$A/native_fa2_q32_fp32_direct.so
export UNIFIED_SO=$A/unified_dispatch_ext.so UNIFIED_FA3_STOCK=1 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_MIN=65
COMMON="--split-policy geometry_v3 --direct-output --packed-query-limit 64 --repeats 2 --ours --model-dir llama70b --draft-dir llama70b_eagle3_sgl --tp-size 4 --no-yarn --common-env SGLANG_SIMULATE_ACC_LEN=3.0 --common-env SGLANG_SIMULATE_ACC_METHOD=match-expected"
FIX="--ours-env SGLANG_TTFT_SLO_MS=500 --ours-env SGLANG_EDF_LONG_PROMPT_TOKENS=8192 --ours-env SGLANG_EDF_LONG_TTFT_SLO_MS=10000 --ours-env SGLANG_EDF_LATE_DEMOTE=1 --ours-env SGLANG_EDF_LATE_DEMOTE_TOK_S=1e12"
R=$BASE/${LLAMA_ROOT:-cap_a6}
mkdir -p $R
test -d $R/python || cp -a $SRC/python $R/python
cp $SRC/*.py $SRC/*.sh $SRC/tracelab_unique.jsonl.gz $R/
export PYTHONPATH=$R/python:$R:$W/fi:$W/drivers/serving TRACE_DATASET_PATH=$R/tracelab_unique.jsonl.gz
cd $R
/dev/shm/w/venv/bin/python -u run_native_serving_pair_20261023.py "$R" --stage mix32 --modes ${LLAMA_MODES:-four two} ${LLAMA_LOAD:---concurrency 4 --sessions 8 --chat-rate 2 --chat-requests 480} $COMMON $FIX --native-env SGLANG_A6_UNIFIED=1
/dev/shm/w/venv/bin/python audit_native_serving_pair_20261023.py "$R" "$R/audit.json" || true
n=$(cat $R/*_native/server.log 2>/dev/null | grep -c "a6_unified=1" || true)
echo "a6 composition: $n"; [ "$n" -gt 0 ] || { echo A6_NOT_ACTIVE; exit 1; }
/dev/shm/w/venv/bin/python $W/perf/bench_per_tenant.py mix32_*/requests || true
echo A6_STAGE2_LLAMA_DONE
