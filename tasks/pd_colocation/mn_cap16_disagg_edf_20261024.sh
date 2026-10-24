#!/bin/bash
# Stronger disagg baseline at 16 GPUs (8 pods x 2 H20, job 2f608c1b1a67cf09), 2026-10-24. In the capacity sweep
# (mn_capacity_16g_*), best-tuned disagg held >= 90% chat attainment only at 16 programs vs ours to 64-96. Disagg's
# prefill pods ran FCFS (clean upstream), so chat prefills queued behind 64K-token agent prefills. This gives the
# best split (6 prefill + 2 decode pods) the same deadline-ordered admission ours uses -- our tree with
# apply_edf_prefill_role_20261024.py, EDF=1, uncached-token deadlines, order only (no chunk preemption on prefill
# pods) -- and reruns the load grid. Same protocol and scoring as the capacity sweep. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-now}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=8 GPUS_PER_POD=2 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1 FO_TEXT=1
export RA_LOG=/tmp/h20/mn_32g_calibrate.log POOL_KV_TOKENS=6696000 ATTN_BACKEND=fa3 KERNEL=lpt
export DISAGG_SGL_PY=/dev/shm/w/up/python
for L in "16 4" "32 8" "48 12" "64 16"; do
  set -- $L; export AGENT_CONC=$1 FO_TASKS=$(( $1 / 2 )) CHAT_RATE=$2 CHAT_N=$(( $2 * 900 ))
  REP=_cap16_L$1_r1 ARMS=disagg_p6_edf_hc bash mn_fanout_8gpu_20261018.sh
  python3 - "disagg_p6_edf_hc_8g_fanout_cap16_L$1_r1" <<'PY'
import sys; sys.path.insert(0, "/opt/tiger/sglang/tasks/pd_colocation")
from px_multi import Cluster
c = Cluster("2f608c1b1a67cf09", "cn-seed", nodes=["worker-0"])
d = f"/dev/shm/w/mn/{sys.argv[1]}"
print(sys.argv[1], c.run(0, f"/dev/shm/w/venv/bin/python /dev/shm/w/perf/fanout_per_tenant.py {d} 2>&1 | tail -1", timeout=240)[1].strip()[-230:], flush=True)
print(sys.argv[1], "prefill-role EDF log lines:", c.run(0, f"cat {d}/*.log 2>/dev/null | grep -c EDF_PREFILL_ROLE_QUEUE", timeout=120)[1].strip()[-40:], flush=True)
PY
done
echo MN_CAP16_DISAGG_EDF_DONE
