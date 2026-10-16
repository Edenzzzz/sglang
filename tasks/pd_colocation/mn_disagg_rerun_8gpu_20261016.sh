#!/bin/bash
# Rerun of the 8-GPU disagg arm at C=16 / chat 4 req/s (2026-10-16): in mn_matrix_8gpu_20261016 its prefill
# workers died on import -- mooncake links libcudart.so.12 and the cu130 venv had none; nvidia-cuda-runtime-cu12
# installed on all pods 17:0x, after that arm's node script had already resolved an empty CUDART. Waits for
# the matrix to finish (same pods). Claude, 2026-10-16.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=bb854a10b8954223 NPODS=4 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
# 18:3x: also the C=32 point -- the C=32 disagg arm failed on a port collision between the two GPUs'
# workers on one prefill pod (mn_node.sh now claims ports under a lock).
for point in "16 4" "32 8"; do
  read -r C R <<< "$point"
  export AGENT_CONC=$C CHAT_RATE=$R SESSIONS=$((C * 3)) CHAT_N=$((R * 300)) RUN_TAG=_8g_c${C}_r${R}_rerun
  KERNEL=lpt PREEMPT=0 python3 -u mn_run.py disagg_2p2d_mtps
done
echo DISAGG_RERUN_DONE
