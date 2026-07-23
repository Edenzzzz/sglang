#!/bin/bash
# Search synthetic serving workloads for regimes where fused LPT+mixed-CG turns
# its kernel advantage into end-to-end throughput or tail-latency gains.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-30000}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-900}"
DATASET_PATH="${DATASET_PATH:-workloads/ShareGPT_V3_unfiltered_cleaned_split.json}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/gain_search_results}"
METHODS_SPEC="${METHODS_SPEC:-DMC LPT_CG}"
EDF_ITL_MS="${EDF_ITL_MS:-}"
EDF_TTFT_MS="${EDF_TTFT_MS:-2000}"
EDF_KV_WATERMARK="${EDF_KV_WATERMARK:-}"
# Conservative LPT default for memory-sensitive and long-output workloads.
# The WL1/3 short-output points use 5; the WL2 midpoint uses 7.5.
LPT_PLAN_OVERHEAD_MS="${LPT_PLAN_OVERHEAD_MS:-10}"
LPT_MODEL="${LPT_MODEL:-$REPO_ROOT/tasks/perf_model/lpt_model.pt}"
DMC_MODEL="${DMC_MODEL:-$REPO_ROOT/tasks/perf_model/dmc_model.pt}"
read -r -a SERVER_EXTRA_FLAGS <<<"${SERVER_EXTRA_ARGS:-}"
mkdir -p "$RESULTS_DIR"

# launch_server performs its warmup through the advertised host. Keep local
# control traffic away from an inherited HTTP proxy.
export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,0.0.0.0"

# input:output:request-rate. These emphasize moderate prefill chunks mixed with
# enough long-lived decodes to approach the favorable kernel cases (3-8), while
# including nearby points to locate where the advantage disappears.
if [ -n "${CASES_SPEC:-}" ]; then
    read -r -a CASES <<<"$CASES_SPEC"
else
    CASES=(
        "2048:512:8" "2048:512:16"
        "4096:512:8" "4096:512:16"
        "4096:1024:8" "4096:1024:12"
        "6144:1024:8" "6144:1024:12"
        "8192:1024:8"
        "8192:2048:4" "8192:2048:8"
    )
fi

server_pid=""
cleanup() {
    if [ -n "$server_pid" ]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null || true
}
trap cleanup EXIT

wait_server() {
    local logfile=$1
    local attempts=$((SERVER_READY_TIMEOUT_S / 2))
    for _ in $(seq 1 "$attempts"); do
        if curl -sf "http://localhost:$PORT/health" >/dev/null; then
            return 0
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -40 "$logfile"
            return 1
        fi
        sleep 2
    done
    tail -40 "$logfile"
    return 1
}

start_server() {
    local method=$1 logfile="$RESULTS_DIR/server_${1}.log"
    local -a method_env=() flags=()
    cleanup
    server_pid=""
    if [ "$method" = LPT_CG ]; then
        flags+=(--enable-mixed-chunk --enable-piecewise-cuda-graph)
        method_env+=(
            SGLANG_FLASHINFER_USE_PERSISTENT=1
            SGLANG_EDF_PLAN_OVERHEAD_MS="$LPT_PLAN_OVERHEAD_MS"
        )
        [ -z "$EDF_ITL_MS" ] || method_env+=(SGLANG_EDF_LATENCY_MODEL="$LPT_MODEL")
    else
        case "$method" in
            DMC_PP)
                # Prefill-prioritized baseline with pure prefill/decode graphs.
                flags+=(--enable-piecewise-cuda-graph)
                ;;
            PP_NOCG)
                # Graph-free prefill-prioritized baseline.  This is the
                # correctness reference when private FlashInfer decode graphs
                # corrupt peak-load outputs.
                flags+=(--disable-cuda-graph)
                ;;
            DMC_NOCG)
                flags+=(--enable-mixed-chunk --disable-cuda-graph)
                ;;
            *)
                flags+=(--enable-piecewise-cuda-graph --enable-mixed-chunk)
                ;;
        esac
        [ "$method" = DMC_SYNC ] && flags+=(--disable-overlap-schedule)
        # A true DMC baseline must not capture mixed graphs through BatchAttention.
        method_env+=(SGLANG_DISABLE_MIXED_CG=1)
        [ -z "$EDF_ITL_MS" ] || method_env+=(SGLANG_EDF_LATENCY_MODEL="$DMC_MODEL")
    fi
    if [ -n "$EDF_ITL_MS" ]; then
        method_env+=(
            SGLANG_EDF_SCHEDULING=1
            SGLANG_TTFT_SLO_MS="$EDF_TTFT_MS"
            SGLANG_ITL_SLO_MS="$EDF_ITL_MS"
        )
        [ -z "$EDF_KV_WATERMARK" ] || method_env+=(
            SGLANG_EDF_KV_ADMISSION_WATERMARK="$EDF_KV_WATERMARK"
        )
        flags+=(
            --enable-priority-scheduling
            --schedule-low-priority-values-first
            --priority-scheduling-preemption-threshold 999999
        )
    fi
    env "${method_env[@]}" python3 -m sglang.launch_server \
        --model "$MODEL" --attention-backend flashinfer \
        --mem-fraction-static 0.85 --port "$PORT" --host 127.0.0.1 \
        --watchdog-timeout 600 "${flags[@]}" "${SERVER_EXTRA_FLAGS[@]}" \
        >"$logfile" 2>&1 &
    server_pid=$!
    wait_server "$logfile"
}

for method in $METHODS_SPEC; do
    start_server "$method"
    for case_spec in "${CASES[@]}"; do
        IFS=: read -r input output rate <<<"$case_spec"
        outfile="$RESULTS_DIR/${method}_in${input}_out${output}_rate${rate}.json"
        [ -f "$outfile" ] && continue
        curl -sf "http://localhost:$PORT/flush_cache" >/dev/null || true
        python3 -m sglang.bench_serving --backend sglang --port "$PORT" \
            --dataset-name random --dataset-path "$DATASET_PATH" \
            --random-input "$input" --random-output "$output" \
            --request-rate "$rate" --num-prompts "$NUM_PROMPTS" \
            --warmup-requests "$WARMUP_REQUESTS" \
            --output-file "$outfile" --output-details --disable-tqdm
    done
done

python3 "$SCRIPT_DIR/summarize_gain_search.py" "$RESULTS_DIR"
