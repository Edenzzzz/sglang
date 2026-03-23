#!/bin/bash
# Run nsys profiling on mlx worker for v3 vs default comparison
export PATH="/mlx_devbox/users/wenxuan.tan/playground/sglang/.venv/bin:$PATH"
cd /mlx_devbox/users/wenxuan.tan/playground/sglang

OUTDIR="/mlx_devbox/users/wenxuan.tan/playground/sglang/tasks/debug_kernel_server_perf/profile_logs"
mkdir -p "$OUTDIR"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Launch server with default-mixed-chunk, profile a short benchmark
echo "=== Profiling Default-mixed-chunk ==="
python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B \
  --attention-backend flashinfer --watchdog-timeout 999999 \
  --mem-fraction-static 0.85 --enable-piecewise-cuda-graph \
  --enable-mixed-chunk --port 30000 > /tmp/nsys_default_server.log 2>&1 &
SERVER_PID=$!

# Wait for server
for i in $(seq 1 120); do
    curl -s http://localhost:30000/v1/models 2>/dev/null | grep -q "Qwen" && break
    sleep 2
done
echo "Server ready"

# Profile 50 requests
nsys profile -o "$OUTDIR/default_mixed_chunk" --force-overwrite true \
  python3 -m sglang.bench_serving --backend sglang --dataset-name random \
  --random-input 4096 --random-output 1024 --request-rate 8 --num-prompt 50 --port 30000 2>&1 | tail -20

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
sleep 5

# Launch server with persistent v3
echo "=== Profiling Persistent v3 ==="
SGLANG_FLASHINFER_USE_PERSISTENT=1 \
python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B \
  --attention-backend flashinfer --watchdog-timeout 999999 \
  --mem-fraction-static 0.85 --enable-piecewise-cuda-graph \
  --enable-mixed-chunk --port 30000 > /tmp/nsys_v3_server.log 2>&1 &
SERVER_PID=$!

for i in $(seq 1 120); do
    curl -s http://localhost:30000/v1/models 2>/dev/null | grep -q "Qwen" && break
    sleep 2
done
echo "Server ready"

nsys profile -o "$OUTDIR/persistent_v3" --force-overwrite true \
  python3 -m sglang.bench_serving --backend sglang --dataset-name random \
  --random-input 4096 --random-output 1024 --request-rate 8 --num-prompt 50 --port 30000 2>&1 | tail -20

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "=== Done ==="
echo "Profiles saved to $OUTDIR"
ls -la "$OUTDIR"/*.nsys-rep 2>/dev/null
