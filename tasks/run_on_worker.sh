#!/bin/bash
# Run benchmark on mlx worker to avoid container memory limits
. /mlx_devbox/users/wenxuan.tan/playground/sglang/.venv/bin/activate
cd /mlx_devbox/users/wenxuan.tan/playground/flashinfer

echo "=== Environment ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import flashinfer; print('flashinfer:', flashinfer.__version__); import torch; print('GPU:', torch.cuda.get_device_name(0))"

echo "=== Running benchmark ==="
python3 benchmarks/bench_mixed_attention.py 2>&1
