#!/bin/bash
# Run all 5 server configs for both workloads
# gpu_keepalive MUST NOT be running during benchmarks (it's compute-intensive)
# The sglang server itself keeps GPU allocated while running.
export PATH="/mlx_devbox/users/wenxuan.tan/playground/sglang/.venv/bin:$PATH"

RESULTS_FILE="/tmp/benchmark_results_v3.txt"
echo "=== Server Benchmark Results (keepalive OFF) ===" > $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE

# Kill any keepalive at the start
pkill -f "gpu_keepalive" 2>/dev/null
sleep 1

wait_for_server() {
    for i in $(seq 1 180); do
        curl -s http://localhost:30000/v1/models 2>/dev/null | grep -q "Qwen" && return 0
        sleep 2
    done
    echo "TIMEOUT" && return 1
}

run_config() {
    local NAME="$1" PERSISTENT="$2" POD="$3" NO_FLIPPED="$4" MIXED_CHUNK="$5"

    echo "" | tee -a $RESULTS_FILE
    echo "========== $NAME ==========" | tee -a $RESULTS_FILE

    # Kill previous server and any keepalive
    pkill -f "sglang" 2>/dev/null
    pkill -f "gpu_keepalive" 2>/dev/null
    sleep 5

    # Verify keepalive is off
    if pgrep -f "gpu_keepalive" > /dev/null 2>&1; then
        echo "ERROR: keepalive still running!" | tee -a $RESULTS_FILE
        return 1
    fi

    local CMD="python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B --attention-backend flashinfer --watchdog-timeout 999999 --mem-fraction-static 0.85 --enable-piecewise-cuda-graph --port 30000"
    [ "$MIXED_CHUNK" = "1" ] && CMD="$CMD --enable-mixed-chunk"

    SGLANG_FLASHINFER_USE_PERSISTENT=$PERSISTENT \
    SGLANG_FLASHINFER_USE_POD=$POD \
    SGLANG_FLASHINFER_NO_FLIPPED_SCHEDULE=$NO_FLIPPED \
    $CMD > /tmp/sglang_server_current.log 2>&1 &
    local SERVER_PID=$!

    if ! wait_for_server; then
        echo "FAILED to start $NAME" | tee -a $RESULTS_FILE
        kill $SERVER_PID 2>/dev/null
        return 1
    fi
    echo "$NAME server ready (PID=$SERVER_PID)"

    # Warmup for POD JIT (but NOT with keepalive!)
    if [ "$POD" = "1" ]; then
        echo "Warming up POD JIT..."
        python3 -m sglang.bench_serving --backend sglang --dataset-name random \
            --random-input 4096 --random-output 1024 --request-rate 4 --num-prompt 20 --port 30000 > /dev/null 2>&1
    fi

    # WL1: 4096 in, 1024 out, 400 prompts
    echo "--- $NAME: WL1 (4096/1024/400) ---" | tee -a $RESULTS_FILE
    python3 -m sglang.bench_serving --backend sglang --dataset-name random \
        --random-input 4096 --random-output 1024 --request-rate 8 --num-prompt 400 --port 30000 2>&1 \
        | tail -20 | tee -a $RESULTS_FILE

    # WL2: 6500 in, 7500 out, 400 prompts
    echo "--- $NAME: WL2 (6500/7500/400) ---" | tee -a $RESULTS_FILE
    python3 -m sglang.bench_serving --backend sglang --dataset-name random \
        --random-input 6500 --random-output 7500 --request-rate 8 --num-prompt 400 --port 30000 2>&1 \
        | tail -20 | tee -a $RESULTS_FILE

    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo "--- $NAME done ---"
}

run_config "Prefill-prioritized"      0 0 0 0
run_config "Default-mixed-chunk"      0 0 0 1
run_config "Persistent (static)"      1 0 1 1
run_config "Persistent v3 (dynamic)"  1 0 0 1
run_config "POD"                      0 1 0 1

# Restart keepalive after all benchmarks
bash /mlx_devbox/users/wenxuan.tan/playground/VibeCUDA/scripts/gpu_keepalive.sh start 2>/dev/null

echo "" | tee -a $RESULTS_FILE
echo "=== ALL DONE ===" | tee -a $RESULTS_FILE
echo "Full results in $RESULTS_FILE"
