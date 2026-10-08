#!/bin/bash
# Workload scan: find optimal input/output length for v3/POD vs default
# Includes P95/P99 TTFT and ITL, all 4 key configs, agent-realistic workloads
# B200 run — to be rerun on H100
export PATH="/mlx_devbox/users/wenxuan.tan/playground/sglang/.venv/bin:$PATH"
cd /mlx_devbox/users/wenxuan.tan/playground/sglang

RESULTS_FILE="/tmp/workload_scan_results.txt"
CSV_FILE="/tmp/workload_scan_results.csv"
echo "=== Workload Scan (B200, keepalive OFF) ===" > $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "config,input,output,num_prompts,total_tok_s,out_tok_s,ttft_med,ttft_p95,ttft_p99,itl_med,itl_p95,itl_p99" > $CSV_FILE

# Stop keepalive
bash /mlx_devbox/users/wenxuan.tan/playground/VibeCUDA/scripts/gpu_keepalive.sh stop 2>/dev/null
sleep 3

wait_for_server() {
    for i in $(seq 1 180); do
        curl -s http://localhost:30000/v1/models 2>/dev/null | grep -q "Qwen" && return 0
        sleep 2
    done
    echo "TIMEOUT" && return 1
}

# Workload matrix — covers agent-realistic lengths + ratio sweep
WORKLOADS=(
    "512,512"      # very short
    "1024,1024"    # short balanced
    "2048,1024"    # prefill-leaning short
    "4096,1024"    # prefill-heavy (WL1, Nebius-SWE turn 4-5)
    "4096,4096"    # balanced long
    "6500,4000"    # Nebius-SWE mid-turn realistic
    "9000,4000"    # Nebius-SWE late-turn realistic
    "6500,7500"    # balanced very long (WL2)
    "1024,4096"    # decode-heavy
    "2048,2048"    # balanced medium
)

run_bench() {
    local CONFIG_NAME="$1" INPUT="$2" OUTPUT="$3"
    local TAG="${CONFIG_NAME}: in=${INPUT} out=${OUTPUT}"
    echo "--- $TAG ---" | tee -a $RESULTS_FILE

    local RAW
    RAW=$(python3 -m sglang.bench_serving --backend sglang --dataset-name random \
        --random-input "$INPUT" --random-output "$OUTPUT" --request-rate 8 --num-prompt 200 --port 30000 2>&1)

    echo "$RAW" | tail -22 | tee -a $RESULTS_FILE

    # Parse metrics
    local TOTAL_TOK=$(echo "$RAW" | grep "Total token throughput" | grep -oP '[\d.]+\s*$' | tr -d ' ')
    local OUT_TOK=$(echo "$RAW" | grep "Output token throughput" | grep -oP '[\d.]+\s*$' | tr -d ' ')
    local TTFT_MED=$(echo "$RAW" | grep "Median TTFT" | grep -oP '[\d.]+\s*$' | tr -d ' ')
    local TTFT_P95=$(echo "$RAW" | grep "P95 TTFT" | grep -oP '[\d.]+\s*$' | tr -d ' ')
    local TTFT_P99=$(echo "$RAW" | grep "P99 TTFT" | grep -oP '[\d.]+\s*$' | tr -d ' ')
    local ITL_MED=$(echo "$RAW" | grep "Median ITL" | grep -oP '[\d.]+\s*$' | tr -d ' ')
    local ITL_P95=$(echo "$RAW" | grep "P95 ITL" | grep -oP '[\d.]+\s*$' | tr -d ' ')
    local ITL_P99=$(echo "$RAW" | grep "P99 ITL" | grep -oP '[\d.]+\s*$' | tr -d ' ')

    echo "${CONFIG_NAME},${INPUT},${OUTPUT},200,${TOTAL_TOK},${OUT_TOK},${TTFT_MED},${TTFT_P95},${TTFT_P99},${ITL_MED},${ITL_P95},${ITL_P99}" >> $CSV_FILE
}

run_config() {
    local NAME="$1" PERSISTENT="$2" POD="$3" NO_FLIPPED="$4" MIXED_CHUNK="$5"

    echo "" | tee -a $RESULTS_FILE
    echo "========== $NAME ==========" | tee -a $RESULTS_FILE

    pkill -f "sglang.launch_server" 2>/dev/null
    sleep 5

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

    # POD warmup
    if [ "$POD" = "1" ]; then
        echo "Warming up POD JIT..."
        python3 -m sglang.bench_serving --backend sglang --dataset-name random \
            --random-input 4096 --random-output 1024 --request-rate 4 --num-prompt 25 --port 30000 > /dev/null 2>&1
    fi

    for WL in "${WORKLOADS[@]}"; do
        IFS=',' read -r INPUT OUTPUT <<< "$WL"
        run_bench "$NAME" "$INPUT" "$OUTPUT"
    done

    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo "--- $NAME done ---"
}

# Run 4 key configs
run_config "Prefill-prioritized"       0 0 0 0
run_config "Default-mixed-chunk"       0 0 0 1
run_config "Persistent-v3-fixed"       1 0 0 1
run_config "POD"                       0 1 0 1

# Restart keepalive
bash /mlx_devbox/users/wenxuan.tan/playground/VibeCUDA/scripts/gpu_keepalive.sh start 2>/dev/null

echo "" | tee -a $RESULTS_FILE
echo "=== ALL DONE ===" | tee -a $RESULTS_FILE
echo "Full results: $RESULTS_FILE"
echo "CSV: $CSV_FILE"
