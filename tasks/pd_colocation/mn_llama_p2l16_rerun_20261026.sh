#!/bin/bash
# After the stock PD 3 + 1 rerun (PID $1): rerun stock PD 2 + 2 at C = 16 (pod 0 claimed no ports in r1: only 3 / 4
# workers came up and the arm aborted after 40 min). Claude, 2026-10-26.
while kill -0 "$1" 2>/dev/null; do sleep 60; done
cd /opt/tiger/sglang/tasks/pd_colocation
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=4 GPUS_PER_POD=4 TP=4 MODEL_DIR=llama70b DRAFT_DIR=llama70b_eagle3_sgl
export CHAT_TEMPLATE=1 FO_TEXT=1 YARN_FACTOR=0 CTX_LEN=131072 ATTN_BACKEND=fa3 POOL_KV_TOKENS=1170000 FO_TASKS=8 FO_PREFIX=32768
export AGENT_CONC=16 CHAT_RATE=4 CHAT_N=3600 PATH=$HOME/.merlin-cli/bin:$PATH
REP=_s3llama_L16_r1b ARMS=disagg_p2_mtps_hc bash mn_fanout_8gpu_20261018.sh
H=$SID.worker_0.seedjob.wenxuan-tan@merlin-ssh-proxy-cn.byted.org
timeout 300 ssh -o StrictHostKeyChecking=no -p 22222 $H "/dev/shm/w/venv/bin/python /dev/shm/w/perf/fanout_per_tenant.py /dev/shm/w/mn/disagg_p2_mtps_hc_8g_fanout_s3llama_L16_r1b" 2>&1 | grep -v Warning
echo LLAMA_P2L16_DONE
