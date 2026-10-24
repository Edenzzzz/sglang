#!/bin/bash
# Conditional disagg with ONE prefill pod (16 GPUs, 2026-10-24). Phase A tried 2 / 4 / 6 prefill pods of 8 and 2 won,
# but its prefill workers took only 0.5% of requests, so fewer may be better: cond_p1 at 64 / 96 / 128 programs so the
# baseline is tuned on both sides of the best split. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-now}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=8 GPUS_PER_POD=2 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1 FO_TEXT=1
export RA_LOG=/tmp/h20/mn_32g_calibrate.log POOL_KV_TOKENS=6696000 ATTN_BACKEND=fa3 KERNEL=lpt
export EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5"
score() { python3 - "$1" <<'PY'
import sys; sys.path.insert(0, "/opt/tiger/sglang/tasks/pd_colocation")
from px_multi import Cluster
c = Cluster("2f608c1b1a67cf09", "cn-seed", nodes=["worker-0"])
d = f"/dev/shm/w/mn/{sys.argv[1]}"
print(sys.argv[1], c.run(0, f"/dev/shm/w/venv/bin/python /dev/shm/w/perf/fanout_per_tenant.py {d} 2>&1 | tail -1", timeout=240)[1].strip()[-230:], flush=True)
print(sys.argv[1], c.run(0, f"/dev/shm/w/venv/bin/python /dev/shm/w/tasks/pd_colocation/cond_disagg_check.py {d} 2>&1 | tail -1", timeout=240)[1].strip()[-300:], flush=True)
PY
}
load() { export AGENT_CONC=$1 FO_TASKS=$(( $1 / 2 )) CHAT_RATE=$2 CHAT_N=$(( $2 * 900 )); }
BEST=1
for L in "64 16" "96 24" "128 32"; do
  set -- $L; load $1 $2
  REP=_cap16_L$1_r1 ARMS=cond_p${BEST}_edf_hc bash mn_fanout_8gpu_20261018.sh
  score cond_p${BEST}_edf_hc_8g_fanout_cap16_L$1_r1
done
echo MN_CAP16_COND_P1_DONE
