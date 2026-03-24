#!/bin/bash
# Analyze nsys profiles to compare mixed vs decode step frequency and duration
export PATH="/mlx_devbox/users/wenxuan.tan/playground/sglang/.venv/bin:$PATH"
cd /mlx_devbox/users/wenxuan.tan/playground/sglang/tasks/debug_kernel_server_perf/profile_logs/nsys

echo "=== Default-mixed-chunk kernel counts ==="
# Count persistent kernel calls (should be 0 for default)
nsys stats --report cuda_gpu_kern_sum default_mixed_chunk_server.nsys-rep --format csv 2>/dev/null \
  | grep -iE "Persistent|BatchPrefill|BatchDecode|batch_attention" | head -10

echo ""
echo "=== Persistent v3 kernel counts ==="
nsys stats --report cuda_gpu_kern_sum persistent_v3_server.nsys-rep --format csv 2>/dev/null \
  | grep -iE "Persistent|BatchPrefill|BatchDecode|batch_attention" | head -10

echo ""
echo "=== Default CUDA API summary (top 10) ==="
nsys stats --report cuda_api_sum default_mixed_chunk_server.nsys-rep --format csv 2>/dev/null \
  | head -12

echo ""
echo "=== v3 CUDA API summary (top 10) ==="
nsys stats --report cuda_api_sum persistent_v3_server.nsys-rep --format csv 2>/dev/null \
  | head -12
