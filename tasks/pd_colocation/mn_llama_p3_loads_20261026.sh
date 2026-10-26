#!/bin/bash
# Stock PD 3 + 1 at the remaining Llama stage-3 loads (it ranks second at C = 8 by agent goodput, 94 vs 1 + 3's 87, once
# the router-policy bug was fixed), after the 2 + 2 C = 16 rerun (PID $1). Claude, 2026-10-26.
while kill -0 "$1" 2>/dev/null; do sleep 60; done
cd /opt/tiger/sglang/tasks/pd_colocation
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=4 GPUS_PER_POD=4 TP=4 MODEL_DIR=llama70b DRAFT_DIR=llama70b_eagle3_sgl
export CHAT_TEMPLATE=1 FO_TEXT=1 YARN_FACTOR=0 CTX_LEN=131072 ATTN_BACKEND=fa3 POOL_KV_TOKENS=1170000 FO_TASKS=8 FO_PREFIX=32768
export PATH=$HOME/.merlin-cli/bin:$PATH
H=$SID.worker_0.seedjob.wenxuan-tan@merlin-ssh-proxy-cn.byted.org
run() { export AGENT_CONC=$1 CHAT_RATE=$2 CHAT_N=$(( $2 * 900 )); REP=$3 ARMS=disagg_p3_mtps_hc bash mn_fanout_8gpu_20261018.sh
  timeout 300 ssh -o StrictHostKeyChecking=no -p 22222 $H "/dev/shm/w/venv/bin/python /dev/shm/w/perf/fanout_per_tenant.py /dev/shm/w/mn/disagg_p3_mtps_hc_8g_fanout$3" 2>&1 | grep -v Warning; }
run 4 1 _s3llama_L4_r1
run 12 3 _s3llama_L12_r1
run 16 4 _s3llama_L16_r1
FO_SPIKE=300:4:300 run 8 2 _s3llama_spike_r1
echo LLAMA_P3_LOADS_DONE
