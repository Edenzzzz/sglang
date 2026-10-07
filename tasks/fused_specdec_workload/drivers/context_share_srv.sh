#!/bin/bash
# Batches 8 and 32 of the context-share measurement, through a server (chunked prefill), since
# bench_one_batch's one-shot prefill OOMs there. No mixed chunk, so measured steps are pure decode.
set -u
W=/dev/shm/w
. $W/venv/bin/activate
export PYTHONPATH=$W/python:$W/fi HF_HOME=$W/hf FI_ROOT=$W/fi TORCH_CUDA_ARCH_LIST=9.0a
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
O=$W/ctxshare
python -m sglang.launch_server --model $W/qwen7b --attention-backend flashinfer --port 30091 \
  --host 127.0.0.1 --context-length 131072 --mem-fraction-static 0.85 \
  --json-model-override-args '{"rope_scaling":{"factor":4.0,"original_max_position_embeddings":32768,"type":"yarn"}}' \
  > $O/server.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:30091/health > /dev/null; do sleep 5; kill -0 $S || exit 1; done
python $W/tasks/fused_specdec_workload/context_share_srv.py http://127.0.0.1:30091 $O/result.jsonl \
  8:4096 8:16384 8:32768 8:65536 8:120000 32:4096 32:16384 32:32768
kill $S; sleep 5
python $W/tasks/fused_specdec_workload/context_share.py $O/result.jsonl
echo CTXSHAREDONE
