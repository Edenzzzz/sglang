#!/bin/bash
# Attention share of a decode step vs context length (debug_kernel_server_perf_4_mode.md item
# "Context length where attention is >50% of e2e"). One bench_one_batch launch covers every
# (batch, input length) pair; then the kernel-only timing of the same shapes.
set -u
W=/dev/shm/w
. $W/venv/bin/activate
export PYTHONPATH=$W/python:$W/fi HF_HOME=$W/hf FI_ROOT=$W/fi TORCH_CUDA_ARCH_LIST=9.0a
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
O=$W/ctxshare; rm -rf $O; mkdir -p $O; cd $O
for B in 1 8 32; do
  case $B in 32) LENS="4096 16384 32768";; *) LENS="4096 16384 32768 65536 120000";; esac
  python -m sglang.bench_one_batch --model-path $W/qwen7b --attention-backend flashinfer \
    --context-length 131072 --mem-fraction-static 0.85 \
    --json-model-override-args '{"rope_scaling":{"factor":4.0,"original_max_position_embeddings":32768,"type":"yarn"}}' \
    --batch-size $B --input-len $LENS --output-len 32 --result-filename $O/result.jsonl > $O/obb_b$B.log 2>&1
  echo "bench_one_batch B=$B rc=$? rows=$(wc -l < $O/result.jsonl)"
done
python $W/tasks/fused_specdec_workload/context_share.py $O/result.jsonl
echo CTXSHAREDONE
