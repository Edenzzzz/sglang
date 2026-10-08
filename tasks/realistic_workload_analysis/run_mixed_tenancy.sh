#!/bin/bash
# Mixed-tenancy serving: one node serving a coding-agent tenant (long-context
# sessions with intermittent tool-result appends and long "thinking" decodes)
# alongside a chat tenant (fresh short prompts, long decodes) — the common
# production case of one cluster backing both a chat product and a coding agent.
#
# This is the composition the persistent overlap kernel is built for: a large
# long-context decode population meeting *fresh, short-prefix* prefill chunks.
# Pure long-context replay never produces it, because every chunk there attends
# a long session prefix and the prefill FLOPs dominate.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
export PATH="$REPO_ROOT/.venv/bin:$PATH"
export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,0.0.0.0"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

MODEL="${MODEL:-/opt/tiger/models/Qwen2.5-7B-Instruct}"
TRACE_PATH="${TRACE_PATH:-$REPO_ROOT/workloads/tracelab/syfi_coding_trace.jsonl.gz}"
CHAT_DATASET="${CHAT_DATASET:-$REPO_ROOT/workloads/ShareGPT_V3_unfiltered_cleaned_split.json}"
PORT="30013"  # fixed: some shells export PORT
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/h800_mixed_tenancy}"
METHODS="${METHODS:-DMC_CG LPT}"

# Coding tenant: causal session replay, moderate-long contexts so both tenants
# fit the KV pool simultaneously.
CODEX_GROUPS="${CODEX_GROUPS:-12}"
CODEX_N="${CODEX_N:-40}"
CODEX_MIN_CTX="${CODEX_MIN_CTX:-32768}"
CODEX_MAX_CTX="${CODEX_MAX_CTX:-49152}"
# Chat tenant: fresh prompts (no shared prefix), sustained arrivals with
# moderate outputs so they inject prefills continuously without crowding the
# long-context sessions out of the decode pool.
CHAT_N="${CHAT_N:-3500}"
CHAT_RATE="${CHAT_RATE:-12}"
CHAT_IN="${CHAT_IN:-512}"
CHAT_OUT="${CHAT_OUT:-96}"
MAX_RUNNING="${MAX_RUNNING:-48}"

mkdir -p "$RESULTS_DIR"
[ -f "$TRACE_PATH" ] || { echo "missing trace: $TRACE_PATH" >&2; exit 1; }

cleanup() { pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null || true; sleep 3; }
trap cleanup EXIT

wait_server() {
    local log=$1 t=0
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        sleep 5; t=$((t+5))
        if [ $t -gt 900 ]; then echo "server timeout; tail:" >&2; tail -20 "$log" >&2; return 1; fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server died; tail:" >&2; tail -20 "$log" >&2; return 1; fi
    done
}

for method in $METHODS; do
    [ -f "$RESULTS_DIR/${method}_chat.json" ] && [ -f "$RESULTS_DIR/${method}_codex.json" ] && continue
    cleanup
    log="$RESULTS_DIR/server_${method}.log"
    env_vars=()
    if [ "$method" = LPT ]; then
        env_vars=(SGLANG_FLASHINFER_USE_PERSISTENT=1)
    else
        # USE_PERSISTENT defaults to 1: a true BatchPrefill baseline must
        # disable it explicitly.
        env_vars=(SGLANG_FLASHINFER_USE_PERSISTENT=0 SGLANG_DISABLE_MIXED_CG=1)
    fi
    env "${env_vars[@]}" python3 -m sglang.launch_server \
        --model "$MODEL" --attention-backend flashinfer --enable-mixed-chunk \
        --mem-fraction-static 0.85 --port "$PORT" --host 127.0.0.1 \
        --max-running-requests "$MAX_RUNNING" --context-length 131072 \
        --json-model-override-args '{"rope_scaling":{"factor":4.0,"original_max_position_embeddings":32768,"type":"yarn"}}' \
        --watchdog-timeout 600 >"$log" 2>&1 &
    SERVER_PID=$!
    wait_server "$log"

    curl -sf "http://127.0.0.1:$PORT/flush_cache" >/dev/null || true

    # Both tenants hit the same server concurrently.
    python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port "$PORT" \
        --model "$MODEL" --dataset-name tracelab-claude --dataset-path "$TRACE_PATH" \
        --tokenize-prompt --use-trace-timestamps --mooncake-slowdown-factor 0.01 \
        --tracelab-num-groups "$CODEX_GROUPS" --tracelab-min-context-len "$CODEX_MIN_CTX" \
        --tracelab-max-context-len "$CODEX_MAX_CTX" --tracelab-min-output-len 512 \
        --request-rate inf --num-prompts "$CODEX_N" --warmup-requests 4 \
        --output-file "$RESULTS_DIR/${method}_codex.json" --output-details --disable-tqdm \
        >"$RESULTS_DIR/bench_${method}_codex.log" 2>&1 &
    CODEX_PID=$!
    python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port "$PORT" \
        --model "$MODEL" --dataset-name random --dataset-path "$CHAT_DATASET" \
        --random-input "$CHAT_IN" --random-output "$CHAT_OUT" --random-range-ratio 0.5 \
        --request-rate "$CHAT_RATE" --num-prompts "$CHAT_N" --warmup-requests 4 \
        --output-file "$RESULTS_DIR/${method}_chat.json" --output-details --disable-tqdm \
        >"$RESULTS_DIR/bench_${method}_chat.log" 2>&1 &
    CHAT_PID=$!
    wait $CODEX_PID; wait $CHAT_PID
    echo "=== $method done ==="
    grep "forward mix" "$log" | tail -1 || true
    cleanup
done
