#!/bin/bash
pkill -9 -f "sglang.launch_server" 2>/dev/null
pkill -9 -f "bench_serving" 2>/dev/null
pkill -9 -f "compile_worker" 2>/dev/null
sleep 2
echo "All killed"
