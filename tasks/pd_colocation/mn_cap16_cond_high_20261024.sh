#!/bin/bash
# Conditional disagg at ours' capacity knee (16 GPUs, 2026-10-24): after mn_cap16_cond_disagg_20261024.sh ($1), the
# best phase-A split at 96 / 128 programs (chat 24 / 32 req/s), where ours held 89.2% / 72.6% chat attainment.
# Phase A at 64 programs had conditional disagg (2 prefill pods) at 94.6% vs ours 94.2%, so the comparison that
# decides the multi-node claim is at and past the knee. Claude.
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
BEST=$(python3 - <<'PY'
import re
rows = []
for l in open("/tmp/h20/mn_cap16_cond.log"):
    m = re.match(r"cond_p(\d+)_edf_hc_8g_fanout_cap16_L64_r1 .*0\.5s/20ms=\s*([0-9.]+)%", l)
    if m: rows.append((float(m.group(2)), int(m.group(1))))
print(max(rows)[1] if rows else "")
PY
)
echo "best conditional-disagg split (prefill pods of 8): $BEST"
[ -n "$BEST" ] || { echo "no phase-A scores parsed"; exit 1; }
for L in "96 24" "128 32"; do
  set -- $L; load $1 $2
  REP=_cap16_L$1_r1 ARMS=cond_p${BEST}_edf_hc bash mn_fanout_8gpu_20261018.sh
  score cond_p${BEST}_edf_hc_8g_fanout_cap16_L$1_r1
done
echo MN_CAP16_COND_HIGH_DONE
