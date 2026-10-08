#!/bin/bash
# Ratio sweep: vary input/output lengths across 3 key configs to find v3 sweet spot
# gpu_keepalive MUST NOT be running during benchmarks (it's compute-intensive)
# Uses 200 prompts per workload for faster iteration (vs 400 in full benchmark)

RESULTS_FILE="/tmp/ratio_sweep_results_v2.txt"
CSV_FILE="/tmp/ratio_sweep_results_v2.csv"
echo "=== Ratio Sweep Results v2 (keepalive OFF) ===" > $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "config,input_len,output_len,num_prompts,total_tok_s,out_tok_s,ttft_median_ms,itl_median_ms,ttft_p99_ms,itl_p99_ms" > $CSV_FILE

# Kill any keepalive at the start to free GPU memory for 0.9
pkill -f "gpu_keepalive" 2>/dev/null
sleep 5

wait_for_server() {
    for i in $(seq 1 180); do
        curl -s http://localhost:30000/v1/models 2>/dev/null | grep -q "Qwen" && return 0
        sleep 2
    done
    echo "TIMEOUT" && return 1
}

# Workload matrix: (input_len, output_len) pairs
# Sweep along two axes + diagonal
WORKLOADS=(
    "1024,1024"    # balanced short
    "4096,1024"    # prefill-heavy (existing WL1)
    "2048,2048"    # balanced medium
    "1024,4096"    # decode-heavy
    "4096,4096"    # balanced long
    "6500,7500"    # balanced very long (existing WL2)
)

run_bench() {
    local CONFIG_NAME="$1" INPUT="$2" OUTPUT="$3"
    local TAG="${CONFIG_NAME}: in=${INPUT} out=${OUTPUT}"
    echo "--- $TAG ---" | tee -a $RESULTS_FILE

    local RAW
    RAW=$(python3 -m sglang.bench_serving --backend sglang --dataset-name random \
        --random-input "$INPUT" --random-output "$OUTPUT" --request-rate 8 --num-prompt 100 --port 30000 2>&1)

    echo "$RAW" | tail -18 | tee -a $RESULTS_FILE

    # Parse key metrics for CSV (handle whitespace-padded output)
    local TOTAL_TOK=$(echo "$RAW" | grep "Total token throughput" | grep -oP '[\d.]+\s*$' | tr -d ' ' || echo "0")
    local OUT_TOK=$(echo "$RAW" | grep "Output token throughput" | grep -oP '[\d.]+\s*$' | tr -d ' ' || echo "0")
    local TTFT_MED=$(echo "$RAW" | grep "Median TTFT" | grep -oP '[\d.]+\s*$' | tr -d ' ' || echo "0")
    local ITL_MED=$(echo "$RAW" | grep "Median ITL" | grep -oP '[\d.]+\s*$' | tr -d ' ' || echo "0")
    local TTFT_P99=$(echo "$RAW" | grep "P99 TTFT" | grep -oP '[\d.]+\s*$' | tr -d ' ' || echo "0")
    local ITL_P99=$(echo "$RAW" | grep "P99 ITL" | grep -oP '[\d.]+\s*$' | tr -d ' ' || echo "0")

    echo "${CONFIG_NAME},${INPUT},${OUTPUT},100,${TOTAL_TOK},${OUT_TOK},${TTFT_MED},${ITL_MED},${TTFT_P99},${ITL_P99}" >> $CSV_FILE
}

run_config() {
    local NAME="$1" PERSISTENT="$2" POD="$3" NO_FLIPPED="$4" MIXED_CHUNK="$5"

    echo "" | tee -a $RESULTS_FILE
    echo "========== $NAME ==========" | tee -a $RESULTS_FILE

    # Kill previous server
    pkill -f "sglang.launch_server" 2>/dev/null
    pkill -9 -f "compile_worker" 2>/dev/null
    sleep 3

    local CMD="python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B --attention-backend flashinfer --watchdog-timeout 999999 --mem-fraction-static 0.85 --enable-piecewise-cuda-graph --port 30000"
    [ "$MIXED_CHUNK" = "1" ] && CMD="$CMD --enable-mixed-chunk"

    # Stop keepalive well before server starts so GPU memory is fully freed for 0.9
    pkill -f "gpu_keepalive" 2>/dev/null
    sleep 5
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

    # Warmup for POD JIT
    if [ "$POD" = "1" ]; then
        echo "Warming up POD JIT (25 requests)..."
        python3 -m sglang.bench_serving --backend sglang --dataset-name random \
            --random-input 4096 --random-output 1024 --request-rate 4 --num-prompt 25 --port 30000 > /dev/null 2>&1
    fi

    # Run all workloads for this config
    for WL in "${WORKLOADS[@]}"; do
        IFS=',' read -r INPUT OUTPUT <<< "$WL"
        run_bench "$NAME" "$INPUT" "$OUTPUT"
    done

    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo "--- $NAME done ---"
}

# Run 3 key configs (skip prefill-prioritized and persistent-static for sweep)
run_config "Default-mixed-chunk"      0 0 0 1
run_config "Persistent-v3-dynamic"    1 0 0 1
run_config "POD"                      0 1 0 1

# Restart keepalive after all benchmarks
bash /mlx_devbox/users/wenxuan.tan/playground/VibeCUDA/scripts/gpu_keepalive.sh start 2>/dev/null

echo "" | tee -a $RESULTS_FILE
echo "=== ALL DONE ===" | tee -a $RESULTS_FILE
echo "Full results: $RESULTS_FILE"
echo "CSV results: $CSV_FILE"
