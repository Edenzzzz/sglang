#!/bin/bash
# Profile the sglang SERVER process with nsys (not the client)
export PATH="/mlx_devbox/users/wenxuan.tan/playground/sglang/.venv/bin:$PATH"
cd /mlx_devbox/users/wenxuan.tan/playground/sglang

OUTDIR="tasks/debug_kernel_server_perf/profile_logs"
CONFIG_NAME="${1:-default_mixed_chunk}"
PERSISTENT="${2:-0}"

echo "=== Config: $CONFIG_NAME, PERSISTENT=$PERSISTENT ==="
nvidia-smi --query-gpu=name --format=csv,noheader

# Launch server under nsys
SGLANG_FLASHINFER_USE_PERSISTENT=$PERSISTENT \
nsys profile -o "$OUTDIR/${CONFIG_NAME}_server" --force-overwrite true \
  --trace cuda,nvtx --cuda-memory-usage=true \
  python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B \
  --attention-backend flashinfer --watchdog-timeout 999999 \
  --mem-fraction-static 0.85 --enable-piecewise-cuda-graph \
  --enable-mixed-chunk --port 30000 > /tmp/nsys_server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server
for i in $(seq 1 180); do
    curl -s http://localhost:30000/v1/models 2>/dev/null | grep -q "Qwen" && echo "Server ready after ${i}x2s" && break
    sleep 2
done

# Run benchmark (client, no profiling)
echo "=== Running benchmark ==="
python3 -m sglang.bench_serving --backend sglang --dataset-name random \
  --random-input 4096 --random-output 1024 --request-rate 8 --num-prompt 50 --port 30000 2>&1 | tail -18

# Kill server to flush nsys profile
echo "=== Stopping server ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "Profile saved to $OUTDIR/${CONFIG_NAME}_server.nsys-rep"
ls -la "$OUTDIR/${CONFIG_NAME}_server.nsys-rep" 2>/dev/null
