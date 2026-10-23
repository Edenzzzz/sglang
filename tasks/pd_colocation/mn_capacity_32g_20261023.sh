#!/bin/bash
# 32-GPU capacity at SLO (8 pods x 4 H20, job 2f608c1b1a67cf09), 2026-10-23: the second point of the 16 -> 32 scale
# sweep, same protocol as mn_capacity_16g_20261023.sh + phase B, load scaled 2x with GPU count. Starts after $1 (the
# 16-GPU phase B and its reruns). Phase A: disagg at every P:D split (1..7 prefill pods of 8, i.e. 4..28 of 32 GPUs),
# KV pool on, at the middle load (64 programs, chat 16 req/s), plus ours. Phase B: ours and the two best splits -- picked
# from phase A's own scores -- at 32 / 64 / 96 / 128 programs with chat 8 / 16 / 24 / 32 req/s. Capacity = highest load
# held at >= 90% per-tenant attainment (chat 0.5 s + 20 ms, agent 10 s + 80 ms). Every arm on FA3; ours with the kernel
# off (schedule vs schedule). Qwen2.5-7B, no spec. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=8 GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1 FO_TEXT=1
export RA_LOG=/tmp/h20/mn_32g_calibrate.log POOL_KV_TOKENS=13392000 ATTN_BACKEND=fa3 KERNEL=lpt
RA_SLO="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5"
load() { export AGENT_CONC=$1 FO_TASKS=$(( $1 / 2 )) CHAT_RATE=$2 CHAT_N=$(( $2 * 900 )); }
score() { python3 - "$1" <<'PY'
import sys; sys.path.insert(0, "/opt/tiger/sglang/tasks/pd_colocation")
from px_multi import Cluster
c = Cluster("2f608c1b1a67cf09", "cn-seed", nodes=["worker-0"])
print(sys.argv[1], c.run(0, f"/dev/shm/w/venv/bin/python /dev/shm/w/perf/fanout_per_tenant.py /dev/shm/w/mn/{sys.argv[1]} 2>&1 | tail -1", timeout=240)[1].strip()[-230:], flush=True)
PY
}
echo "##### PHASE A (32 GPUs): P:D sweep at 64 programs / chat 16"
load 64 16
for P in 1 2 3 4 5 6 7; do
  REP=_cap32_L64_r1 ARMS=disagg_p${P}_hc bash mn_fanout_8gpu_20261018.sh
  score disagg_p${P}_hc_8g_fanout_cap32_L64_r1
done
EXTRA_RA="$RA_SLO" REP=_cap32_L64_r1 ARMS=coloc_slo_edf_hc bash mn_fanout_8gpu_20261018.sh
score coloc_slo_edf_hc_8g_fanout_cap32_L64_r1
echo PHASE_A32_DONE
BEST=$(python3 - <<'PY'
import re
rows = []
for l in open("/tmp/h20/mn_cap32.log"):
    m = re.match(r"(disagg_p(\d+)_hc_8g_fanout_cap32_L64_r1) .*agent 10s/80ms=\s*([0-9.]+)tok/s", l)
    if m: rows.append((float(m.group(3)), int(m.group(2))))
rows.sort(reverse=True)
print(" ".join(str(p) for _, p in rows[:2]))
PY
)
echo "best disagg splits (prefill pods of 8): $BEST"
[ -n "$BEST" ] || { echo "no phase-A disagg scores parsed"; exit 1; }
for L in "32 8" "96 24" "128 32"; do
  set -- $L; load $1 $2
  EXTRA_RA="$RA_SLO" REP=_cap32_L$1_r1 ARMS=coloc_slo_edf_hc bash mn_fanout_8gpu_20261018.sh
  score coloc_slo_edf_hc_8g_fanout_cap32_L$1_r1
  for P in $BEST; do
    REP=_cap32_L$1_r1 ARMS=disagg_p${P}_hc bash mn_fanout_8gpu_20261018.sh
    score disagg_p${P}_hc_8g_fanout_cap32_L$1_r1
  done
done
echo MN_CAP32_DONE
