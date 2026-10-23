#!/bin/bash
# Rerun of ours at the phase-A load (32 programs / chat 8, 8 pods x 2 GPUs) -- the phase-A arm aborted before starting
# because the cn-seed WebShell SSO session expired mid-sweep ("mooncake master did not start: webshell relay redirected
# to SSO"); not a failure of the arm. Waits for $1 (phase B); mn_run's cluster lock serialises it against other arms.
# Claude, 2026-10-23.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=8 GPUS_PER_POD=2 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1 FO_TEXT=1
export RA_LOG=/tmp/h20/mn_32g_calibrate.log POOL_KV_TOKENS=6696000 ATTN_BACKEND=fa3 KERNEL=lpt
export AGENT_CONC=32 FO_TASKS=16 CHAT_RATE=8 CHAT_N=7200
EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_cap16_L32_r2 ARMS=coloc_slo_edf_hc bash mn_fanout_8gpu_20261018.sh
python3 - <<'PY'
import sys; sys.path.insert(0, "/opt/tiger/sglang/tasks/pd_colocation")
from px_multi import Cluster
c = Cluster("2f608c1b1a67cf09", "cn-seed", nodes=["worker-0"])
d = "coloc_slo_edf_hc_8g_fanout_cap16_L32_r2"
print(d, c.run(0, f"/dev/shm/w/venv/bin/python /dev/shm/w/perf/fanout_per_tenant.py /dev/shm/w/mn/{d} 2>&1 | tail -1", timeout=240)[1].strip()[-230:])
PY
echo MN_CAP16_OURS_L32_DONE
