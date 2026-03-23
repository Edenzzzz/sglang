#!/bin/bash
export PATH="/mlx_devbox/users/wenxuan.tan/playground/sglang/.venv/bin:$PATH"
cd /mlx_devbox/users/wenxuan.tan/playground/sglang/tasks/profile_logs

echo "=== Default-mixed-chunk kernel summary ==="
nsys stats --report cuda_gpu_kern_sum default_mixed_chunk.nsys-rep 2>&1 | head -40

echo ""
echo "=== Persistent v3 kernel summary ==="
nsys stats --report cuda_gpu_kern_sum persistent_v3.nsys-rep 2>&1 | head -40

echo ""
echo "=== Default GPU trace summary ==="
nsys stats --report cuda_gpu_trace default_mixed_chunk.nsys-rep 2>&1 | head -20

echo ""
echo "=== V3 GPU trace summary ==="
nsys stats --report cuda_gpu_trace persistent_v3.nsys-rep 2>&1 | head -20
