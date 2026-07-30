#!/bin/bash
# EDF benchmark sweep: PP / DMC / LPT × ShareGPT/Code-Feedback/SWE-Gym × multiple rates
# All runs with --enable-priority-scheduling + EDF env vars + SLO-aware chunk control
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
export PATH="$REPO_ROOT/.venv/bin:$PATH"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-30000}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
mkdir -p "$RESULTS_DIR"
RATES="${RATES:-2 4 8 12 16 24 inf}"
METHODS="${METHODS:-PP_DECODE_CG DMC_CG LPT}"
WORKLOAD_FILTER="${WORKLOAD_FILTER:-}"
NUM_PROMPTS_OVERRIDE="${NUM_PROMPTS_OVERRIDE:-400}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-10}"
LPT_PERSISTENT="${LPT_PERSISTENT:-1}"
LPT_CHUNKED_PREFILL_SIZE="${LPT_CHUNKED_PREFILL_SIZE:-7168}"
SERVER_PID=""

LPT_MODEL="${LPT_MODEL:-$REPO_ROOT/tasks/perf_model/lpt_model.pt}"
DMC_MODEL="${DMC_MODEL:-$REPO_ROOT/tasks/perf_model/dmc_model.pt}"

kill_all() {
    if [ -n "$SERVER_PID" ]; then
        # launch_server creates scheduler/detokenizer children; terminate its
        # process group so an orphaned child cannot retain the listening port.
        kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
    local stale_pid
    while read -r stale_pid; do
        [ -n "$stale_pid" ] && kill -9 "$stale_pid" 2>/dev/null || true
    done < <(pgrep -f "[s]glang.launch_server.*--port $PORT" 2>/dev/null || true)
    while read -r stale_pid; do
        [ -n "$stale_pid" ] && kill -9 "$stale_pid" 2>/dev/null || true
    done < <(pgrep -f "[s]glang.bench_serving.*--port $PORT" 2>/dev/null || true)
    sleep 5
}
trap kill_all EXIT INT TERM

wait_server() {
    local logfile=$1
    for i in $(seq 1 450); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
            echo "  Server ready after $((i*2))s"
            return 0
        fi
        if grep -Eq "address already in use|ERROR:.*bind|Address already in use" "$logfile" 2>/dev/null; then
            echo "  ERROR: Server could not bind to port $PORT"
            tail -20 "$logfile"
            return 1
        fi
        if [ -n "$SERVER_PID" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "  ERROR: Server process exited before becoming healthy"
            tail -20 "$logfile"
            return 1
        fi
        sleep 2
    done
    echo "  ERROR: Server failed!"
    tail -20 "$logfile"
    return 1
}

run_bench() {
    local method=$1 wl=$2 rate=$3 dataset=$4 num=$5 ttft=$6 itl=$7 extra=$8
    local outfile="$RESULTS_DIR/${method}_${wl}_rate${rate}.json"

    if [ -f "$outfile" ]; then
        echo "  SKIP: $outfile exists"
        return 0
    fi

    if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "  Server unhealthy, aborting"
        return 1
    fi

    echo "  Run $method/$wl/rate=$rate (TTFT=$ttft, ITL=$itl)"
    curl -s http://localhost:$PORT/flush_cache > /dev/null 2>&1 || true
    sleep 2

    timeout 3600 python3 -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port $PORT \
        --dataset-name $dataset $extra \
        --num-prompts $num \
        --warmup-requests "$WARMUP_REQUESTS" \
        --request-rate $rate \
        --output-file "$outfile" \
        --output-details --disable-tqdm 2>&1 | tail -20

    [ -f "$outfile" ] && echo "  OK" || echo "  FAILED"
}

start_server() {
    local method=$1 ttft=$2 itl=$3 logfile=$4
    local extra_flags=""
    local model_env=""
    local prefix_env=""
    local chunked_prefill_size=8192

    case "$method" in
        PP_DECODE_CG)
            # Unchanged BatchPrefill plus regular decode CUDA graphs.
            extra_flags=""
            model_env=""  # PP doesn't use chunk control (no mixed batches)
            ;;
        DMC_CG)
            # Unchanged mixed BatchPrefill plus regular decode CUDA graphs.
            # USE_PERSISTENT defaults to 1 and must be disabled explicitly for
            # a true BatchPrefill baseline.
            extra_flags="--enable-mixed-chunk"
            model_env="SGLANG_EDF_LATENCY_MODEL=$DMC_MODEL"
            prefix_env="SGLANG_FLASHINFER_USE_PERSISTENT=0"
            ;;
        LPT)
            # Persistent mixed attention plus regular decode CUDA graphs.
            extra_flags="--enable-mixed-chunk"
            model_env="SGLANG_EDF_LATENCY_MODEL=$LPT_MODEL"
            prefix_env="SGLANG_FLASHINFER_USE_PERSISTENT=$LPT_PERSISTENT SGLANG_EDF_PLAN_OVERHEAD_MS=auto"
            chunked_prefill_size="$LPT_CHUNKED_PREFILL_SIZE"
            ;;
    esac

    env $prefix_env \
        SGLANG_EDF_SCHEDULING=1 SGLANG_TTFT_SLO_MS=$ttft SGLANG_ITL_SLO_MS=$itl \
        $model_env \
        nohup setsid python -m sglang.launch_server --model $MODEL $extra_flags \
        --attention-backend flashinfer \
        --chunked-prefill-size "$chunked_prefill_size" --mem-fraction-static 0.85 \
        --port $PORT --host 127.0.0.1 \
        --enable-priority-scheduling --schedule-low-priority-values-first \
        --priority-scheduling-preemption-threshold 999999 \
        --watchdog-timeout 600 > "$logfile" 2>&1 &
    SERVER_PID=$!
    disown
}

# Workload definitions: name:dataset:num:ttft_ms:itl_ms:extra_args
WORKLOADS=(
    "sharegpt:sharegpt:200:2000:100:--dataset-path workloads/ShareGPT_V3_unfiltered_cleaned_split.json"
    "code_feedback:code-feedback:200:3000:150:"
    "swe:swe-gym:100:5000:150:"
)

for METHOD in $METHODS; do
    for wl_info in "${WORKLOADS[@]}"; do
        IFS=: read -r wl dataset num ttft itl extra <<< "$wl_info"
        if [ -n "$WORKLOAD_FILTER" ] && [ "$wl" != "$WORKLOAD_FILTER" ]; then
            continue
        fi
        if [ -n "$NUM_PROMPTS_OVERRIDE" ]; then
            num=$NUM_PROMPTS_OVERRIDE
        fi
        echo ""
        echo "===== $METHOD / $wl ====="
        kill_all

        LOGFILE="/tmp/edf_${METHOD}_${wl}.log"
        rm -f "$LOGFILE"
        start_server "$METHOD" "$ttft" "$itl" "$LOGFILE"

        if ! wait_server "$LOGFILE"; then
            kill_all
            continue
        fi

        for rate in $RATES; do
            if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
                echo "  Server died, restarting"
                kill_all
                start_server "$METHOD" "$ttft" "$itl" "$LOGFILE"
                wait_server "$LOGFILE" || break
            fi
            run_bench "$METHOD" "$wl" "$rate" "$dataset" "$num" "$ttft" "$itl" "$extra"
        done
    done
done

kill_all
echo ""
echo "=== Done ==="
ls "$RESULTS_DIR"/*.json 2>/dev/null | wc -l
echo "total JSON results"
