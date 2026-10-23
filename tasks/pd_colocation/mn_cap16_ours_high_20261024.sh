#!/bin/bash
# Find OURS' capacity at 16 GPUs (8 pods x 2, job 2f608c1b1a67cf09), 2026-10-24, after $1. Phase B ended with ours still
# above 90% chat attainment at the highest load tested (64 programs + chat 16: 94.2%), so the capacity ratio vs the
# best-tuned disagg (which holds >= 90% only at 16 programs) is so far a lower bound. Ours at 96 / 128 programs with
# chat 24 / 32 req/s, same protocol (FA3, kernel off, pool on, cache-first SLO router). Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=8 GPUS_PER_POD=2 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1 FO_TEXT=1
export RA_LOG=/tmp/h20/mn_32g_calibrate.log POOL_KV_TOKENS=6696000 ATTN_BACKEND=fa3 KERNEL=lpt
for L in "96 24" "128 32"; do
  set -- $L; export AGENT_CONC=$1 FO_TASKS=$(( $1 / 2 )) CHAT_RATE=$2 CHAT_N=$(( $2 * 900 ))
  EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_cap16_L$1_r1 ARMS=coloc_slo_edf_hc bash mn_fanout_8gpu_20261018.sh
  python3 - "coloc_slo_edf_hc_8g_fanout_cap16_L$1_r1" <<'PY'
import sys; sys.path.insert(0, "/opt/tiger/sglang/tasks/pd_colocation")
from px_multi import Cluster
c = Cluster("2f608c1b1a67cf09", "cn-seed", nodes=["worker-0"])
print(sys.argv[1], c.run(0, f"/dev/shm/w/venv/bin/python /dev/shm/w/perf/fanout_per_tenant.py /dev/shm/w/mn/{sys.argv[1]} 2>&1 | tail -1", timeout=240)[1].strip()[-230:], flush=True)
PY
done
echo MN_CAP16_OURS_HIGH_DONE
