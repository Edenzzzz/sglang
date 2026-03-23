#!/bin/bash
# Launch the 5-config benchmark via nohup so it survives SSH disconnection
nohup bash /mlx_devbox/users/wenxuan.tan/playground/sglang/tasks/run_all_benchmarks.sh > /tmp/benchmark_v3_log.txt 2>&1 &
echo "Benchmark PID: $!"
echo "Output: /tmp/benchmark_v3_log.txt"
echo "Results: /tmp/benchmark_results_v3.txt"
echo "Check progress: tail -5 /tmp/benchmark_results_v3.txt"
