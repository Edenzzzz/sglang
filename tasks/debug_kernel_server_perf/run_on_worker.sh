#!/bin/bash
# Run benchmark on mlx worker to avoid container memory limits
export PATH="/mlx_devbox/users/wenxuan.tan/playground/sglang/.venv/bin:$PATH"
cd /mlx_devbox/users/wenxuan.tan/playground/flashinfer

echo "=== Environment ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import flashinfer; print('flashinfer:', flashinfer.__version__); import torch; print('GPU:', torch.cuda.get_device_name(0))"

echo "=== Clearing JIT cache ==="
python3 -m flashinfer clear-cache 2>/dev/null
rm -rf ~/.cache/flashinfer 2>/dev/null

echo "=== Running benchmark ==="
python3 benchmarks/bench_mixed_attention.py 2>&1
