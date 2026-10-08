#!/bin/bash
# Replay the long-context, decode-heavy Claude slice from TraceLab on one H800.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
TRACE_PATH="${TRACE_PATH:-$REPO_ROOT/workloads/tracelab/syfi_coding_trace.jsonl.gz}"
MODEL="${MODEL:-/opt/tiger/models/Qwen2.5-7B-Instruct}"

if [ ! -f "$TRACE_PATH" ]; then
    echo "TraceLab dataset not found: $TRACE_PATH" >&2
    echo "Download syfi_coding_trace.jsonl.gz from the TraceLab v0.0.1 release." >&2
    exit 1
fi

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export PATH="$REPO_ROOT/.venv/bin:$PATH"

# LPT-only cap override for controlled screens. Cap tuning trades persistent
# kernel efficiency against the mixed-step decode-interference window; no one
# cap won every TraceLab load, and this never changes DMC/PP or adds a minimum.
METHODS_SPEC="${METHODS_SPEC:-DMC_CG LPT PP_DECODE_CG}" \
CASES_SPEC="${CASES_SPEC:-1:1:inf}" \
NUM_PROMPTS="${NUM_PROMPTS:-400}" \
WARMUP_REQUESTS="${WARMUP_REQUESTS:-10}" \
LPT_CHUNKED_PREFILL_SIZE="${LPT_CHUNKED_PREFILL_SIZE:-7168}" \
RESULT_WORKLOAD_NAME=tracelab_claude \
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/h800_qwen25_tracelab_claude_faithful_64k100k_n400_mr40}" \
MODEL="$MODEL" \
PORT="${PORT:-30005}" \
SERVER_EXTRA_ARGS="--max-running-requests=${MAX_RUNNING_REQUESTS:-40} --context-length=131072 --json-model-override-args={\"rope_scaling\":{\"factor\":4.0,\"original_max_position_embeddings\":32768,\"type\":\"yarn\"}} ${SERVER_EXTRA_ARGS_EXTRA:-}" \
BENCH_EXTRA_ARGS="--tokenize-prompt --use-trace-timestamps --mooncake-slowdown-factor ${TRACELAB_TIME_SCALE:-0.01} --dataset-name tracelab-claude --dataset-path $TRACE_PATH --tracelab-num-groups ${TRACELAB_NUM_GROUPS:-10} --tracelab-min-context-len ${TRACELAB_MIN_CONTEXT_LEN:-65536} --tracelab-max-context-len ${TRACELAB_MAX_CONTEXT_LEN:-100000} --tracelab-min-output-len ${TRACELAB_MIN_OUTPUT_LEN:-1} ${BENCH_EXTRA_ARGS_EXTRA:-}" \
bash "$REPO_ROOT/tasks/debug_kernel_server_perf/run_gain_search.sh"
