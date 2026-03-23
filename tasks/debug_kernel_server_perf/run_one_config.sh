#!/bin/bash
# Run one server config benchmark
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

# Kill any existing server and stop keepalive (it corrupts benchmark results)
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

# Warmup for POD JIT
if [ "$POD" = "1" ]; then
    echo "Warming up POD JIT..."
    python3 -m sglang.bench_serving --backend sglang --dataset-name random \
        --random-input 4096 --random-output 1024 --request-rate 4 --num-prompt 25 --port 30000 > /dev/null 2>&1
fi

# WL1 — save full output + summary to results
echo "--- $NAME: WL1 (4096/1024/400) ---" | tee -a $RESULTS
python3 -m sglang.bench_serving --backend sglang --dataset-name random \
    --random-input 4096 --random-output 1024 --request-rate 8 --num-prompt 400 --port 30000 2>&1 \
    | tee "$BENCH_DIR/${NAME}_WL1.txt" | tail -22 | tee -a $RESULTS

# WL2 — save full output + summary to results
echo "--- $NAME: WL2 (6500/7500/400) ---" | tee -a $RESULTS
python3 -m sglang.bench_serving --backend sglang --dataset-name random \
    --random-input 6500 --random-output 7500 --request-rate 8 --num-prompt 400 --port 30000 2>&1 \
    | tee "$BENCH_DIR/${NAME}_WL2.txt" | tail -22 | tee -a $RESULTS

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "--- $NAME done ---" | tee -a $RESULTS

# Restart keepalive to hold GPU
bash /mlx_devbox/users/wenxuan.tan/playground/VibeCUDA/scripts/gpu_keepalive.sh start 2>/dev/null
