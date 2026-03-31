#!/bin/bash
# Run one server config benchmark across WL1-6
# Usage: run_one_config.sh <name> <persistent> <pod> <no_flipped> <mixed_chunk>
export PATH="/mlx_devbox/users/wenxuan.tan/playground/sglang/.venv/bin:$PATH"
cd /mlx_devbox/users/wenxuan.tan/playground/sglang

NAME="$1" PERSISTENT="$2" POD="$3" NO_FLIPPED="$4" MIXED_CHUNK="$5"
RESULTS="/tmp/benchmark_results_v3.txt"
BENCH_DIR="tasks/debug_kernel_server_perf/profile_logs/bench_results"
mkdir -p "$BENCH_DIR"

echo "" >> $RESULTS
echo "========== $NAME ==========" >> $RESULTS
echo "========== $NAME =========="

# Kill any existing server and stop keepalive
pkill -f "sglang.launch_server" 2>/dev/null
bash /mlx_devbox/users/wenxuan.tan/playground/VibeCUDA/scripts/gpu_keepalive.sh stop 2>/dev/null
sleep 3

CMD="python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B --attention-backend flashinfer --watchdog-timeout 999999 --mem-fraction-static 0.85 --enable-piecewise-cuda-graph --port 30000"
[ "$MIXED_CHUNK" = "1" ] && CMD="$CMD --enable-mixed-chunk"

SGLANG_FLASHINFER_USE_PERSISTENT=$PERSISTENT \
SGLANG_FLASHINFER_USE_POD=$POD \
SGLANG_FLASHINFER_NO_FLIPPED_SCHEDULE=$NO_FLIPPED \
$CMD > /tmp/sglang_server_current.log 2>&1 &
SERVER_PID=$!

# Wait for server
for i in $(seq 1 180); do
    curl -s http://localhost:30000/v1/models 2>/dev/null | grep -q "Qwen" && break
    sleep 2
done
echo "$NAME server ready (PID=$SERVER_PID)"

# Warmup (JIT + CUDA graphs)
echo "Warming up..."
python3 -m sglang.bench_serving --backend sglang --dataset-name random \
    --random-input 4096 --random-output 1024 --request-rate 4 --num-prompt 25 --port 30000 > /dev/null 2>&1

run_wl() {
    local WL="$1" INPUT="$2" OUTPUT="$3" RATE="$4"
    echo "--- $NAME: WL${WL} (${INPUT}/${OUTPUT}/${RATE}) ---" | tee -a $RESULTS
    python3 -m sglang.bench_serving --backend sglang --dataset-name random \
        --random-input $INPUT --random-output $OUTPUT --request-rate $RATE --num-prompt 400 --port 30000 2>&1 \
        | tee "$BENCH_DIR/${NAME}_WL${WL}.txt" | tail -22 | tee -a $RESULTS
}

run_wl 1 4096  1024 8
run_wl 2 8192  2048 4
run_wl 3 12000 1024 4
run_wl 4 16384 1024 3
run_wl 5 4096  4096 6
run_wl 6 6000  6000 3

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "--- $NAME done ---" | tee -a $RESULTS

# Restart keepalive to hold GPU
bash /mlx_devbox/users/wenxuan.tan/playground/VibeCUDA/scripts/gpu_keepalive.sh start 2>/dev/null
