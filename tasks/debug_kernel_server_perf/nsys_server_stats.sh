#!/bin/bash
export PATH="/mlx_devbox/users/wenxuan.tan/playground/sglang/.venv/bin:$PATH"
cd /mlx_devbox/users/wenxuan.tan/playground/sglang/tasks/debug_kernel_server_perf/profile_logs

for f in default_mixed_chunk_server persistent_v3_server; do
    if [ ! -f "${f}.nsys-rep" ]; then
        echo "=== $f: not found ==="
        continue
    fi
    echo "=== $f: kernel summary (top 15 by total time) ==="
    nsys stats --report cuda_gpu_kern_sum "${f}.nsys-rep" --force-export true 2>&1 | head -25
    echo ""
    echo "=== $f: CUDA API summary ==="
    nsys stats --report cuda_api_sum "${f}.nsys-rep" 2>&1 | head -20
    echo ""
done
