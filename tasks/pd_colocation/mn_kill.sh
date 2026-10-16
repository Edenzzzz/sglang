#!/bin/bash
# Tear down every multi-node process on this pod. A script file, not an inline exec string: an inline
# pattern list appears on the exec shell's own command line, so the loop killed its own shell and every
# per-pod call ran to its timeout (~13 min of overhead per arm, 2026-10-16).
for p in $(ps -eo pid,cmd | grep -E 'sglang.launch_server|sglang::|sglang_router|router.py|bench_serving|benchmark.serving|mn_bench|mooncake_master' \
           | grep -v -E 'grep|mn_kill' | awk '{print $1}'); do
  [ "$p" != "$$" ] && kill -9 $p 2>/dev/null
done
sleep 2
rm -f ~/.cache/torch_extensions/py312_cu128/fused_fa3/lock
echo killed
