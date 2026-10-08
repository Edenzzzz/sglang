#!/bin/bash
# HOL blocking scan: find where prefill-prioritized loses to mixed-chunk
# due to head-of-line blocking from long prefills.
# Primary signal: P99 ITL (HOL blocking shows up here first)
export PATH="/mlx_devbox/users/wenxuan.tan/playground/sglang/.venv/bin:$PATH"
cd /mlx_devbox/users/wenxuan.tan/playground/sglang

RESULTS_FILE="/tmp/hol_blocking_scan_results.txt"
CSV_FILE="/tmp/hol_blocking_scan_results.csv"
echo "=== HOL Blocking Scan (B200) ===" > $RESULTS_FILE
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

# Large prefill lengths — prefill-prio should lose at some point due to HOL blocking
# Keep output fixed at 1024 to isolate prefill length effect
WORKLOADS=(
    "4096,1024"     # baseline (prefill-prio wins here)
    "8192,1024"     # 2x baseline
    "12000,1024"    # ~3x
    "16000,1024"    # 4x — should start to see HOL
    "24000,1024"    # 6x — likely HOL blocking visible
    "32000,1024"    # near Qwen3-8B limit (32K context)
)

run_bench() {
    local CONFIG_NAME="$1" INPUT="$2" OUTPUT="$3"
    local TAG="${CONFIG_NAME}: in=${INPUT} out=${OUTPUT}"
    echo "--- $TAG ---" | tee -a $RESULTS_FILE

    local RAW
    RAW=$(python3 -m sglang.bench_serving --backend sglang --dataset-name random \
        --random-input "$INPUT" --random-output "$OUTPUT" --request-rate 8 --num-prompt 100 --port 30000 2>&1)

    echo "$RAW" | tail -22 | tee -a $RESULTS_FILE

    local TOTAL_TOK=$(echo "$RAW" | grep "Total token throughput" | grep -oP '[\d.]+\s*$' | tr -d ' ')
    local OUT_TOK=$(echo "$RAW" | grep "Output token throughput" | grep -oP '[\d.]+\s*$' | tr -d ' ')
    local TTFT_MED=$(echo "$RAW" | grep "Median TTFT" | grep -oP '[\d.]+\s*$' | tr -d ' ')
    local TTFT_P95=$(echo "$RAW" | grep "P95 TTFT" | grep -oP '[\d.]+\s*$' | tr -d ' ')
    local TTFT_P99=$(echo "$RAW" | grep "P99 TTFT" | grep -oP '[\d.]+\s*$' | tr -d ' ')
    local ITL_MED=$(echo "$RAW" | grep "Median ITL" | grep -oP '[\d.]+\s*$' | tr -d ' ')
    local ITL_P95=$(echo "$RAW" | grep "P95 ITL" | grep -oP '[\d.]+\s*$' | tr -d ' ')
    local ITL_P99=$(echo "$RAW" | grep "P99 ITL" | grep -oP '[\d.]+\s*$' | tr -d ' ')

    echo "${CONFIG_NAME},${INPUT},${OUTPUT},100,${TOTAL_TOK},${OUT_TOK},${TTFT_MED},${TTFT_P95},${TTFT_P99},${ITL_MED},${ITL_P95},${ITL_P99}" >> $CSV_FILE
}

run_config() {
    local NAME="$1" PERSISTENT="$2" MIXED_CHUNK="$3"

    echo "" | tee -a $RESULTS_FILE
    echo "========== $NAME ==========" | tee -a $RESULTS_FILE

    pkill -f "sglang.launch_server" 2>/dev/null
    sleep 5

    local CMD="python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B --attention-backend flashinfer --watchdog-timeout 999999 --mem-fraction-static 0.85 --enable-piecewise-cuda-graph --port 30000"
    [ "$MIXED_CHUNK" = "1" ] && CMD="$CMD --enable-mixed-chunk"

    SGLANG_FLASHINFER_USE_PERSISTENT=$PERSISTENT \
    $CMD > /tmp/sglang_server_hol.log 2>&1 &
    local SERVER_PID=$!

    if ! wait_for_server; then
        echo "FAILED to start $NAME" | tee -a $RESULTS_FILE
        kill $SERVER_PID 2>/dev/null
        return 1
    fi
    echo "$NAME server ready (PID=$SERVER_PID)"

    for WL in "${WORKLOADS[@]}"; do
        IFS=',' read -r INPUT OUTPUT <<< "$WL"
        run_bench "$NAME" "$INPUT" "$OUTPUT"
    done

    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo "--- $NAME done ---"
}

# Compare prefill-prioritized vs persistent-v3 mixed-chunk
run_config "Prefill-prioritized"  0 0
run_config "Persistent-v3"        1 1

# Restart keepalive
bash /mlx_devbox/users/wenxuan.tan/playground/VibeCUDA/scripts/gpu_keepalive.sh start 2>/dev/null

echo "" | tee -a $RESULTS_FILE
echo "=== ALL DONE ===" | tee -a $RESULTS_FILE
echo "Full results: $RESULTS_FILE"
echo "CSV: $CSV_FILE"
