#!/bin/bash
# After the Qwen3-8B half-load table (pid $1): the step decomposition on the idle H200 (2026-10-21). Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
while kill -0 "$1" 2>/dev/null; do sleep 30; done
python3 - <<'PY'
import sys, time; sys.path.insert(0, "/tmp")
from px import run
run("cd /dev/shm/w/perf && CUDA_VISIBLE_DEVICES=0 setsid nohup /dev/shm/w/venv/bin/python fusion_decompose_20261021.py > fusion_decompose.out 2>&1 < /dev/null &")
for _ in range(60):
    out = run("pgrep -f fusion_decompose >/dev/null && echo RUN || echo DONE")[1]
    if "DONE" in out: break
    time.sleep(30)
print(run("grep -v INFO /dev/shm/w/perf/fusion_decompose.out")[1][-6000:])
PY
echo H200_DECOMPOSE_DONE
