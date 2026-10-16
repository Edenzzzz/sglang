#!/bin/bash
# Rerun of the 8-GPU disagg arm at C=16 / chat 4 req/s (2026-10-16): in mn_matrix_8gpu_20261016 its prefill
# workers died on import -- mooncake links libcudart.so.12 and the cu130 venv had none; nvidia-cuda-runtime-cu12
# installed on all pods 17:0x, after that arm's node script had already resolved an empty CUDART. Waits for
# the matrix to finish (same pods). Claude, 2026-10-16.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
until grep -q MATRIX8_DONE /tmp/h20/mn_matrix_8gpu.log 2>/dev/null; do sleep 60; done
export SID=bb854a10b8954223 NPODS=4 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export AGENT_CONC=16 CHAT_RATE=4 SESSIONS=48 CHAT_N=1200 RUN_TAG=_8g_c16_r4_rerun
KERNEL=lpt PREEMPT=0 python3 -u mn_run.py disagg_2p2d_mtps
echo DISAGG_RERUN_DONE
