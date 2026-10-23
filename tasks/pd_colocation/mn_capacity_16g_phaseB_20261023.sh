#!/bin/bash
# Phase B of the 16-GPU capacity-at-SLO comparison (8 pods x 2 GPUs, job 2f608c1b1a67cf09, 2026-10-23), after phase A
# (pid $1, mn_capacity_16g_20261023.sh). Picks the two disagg P:D splits with the highest agent SLO goodput at the
# phase-A load FROM PHASE A'S OWN SCORE LINES (no hand choice), then runs ours and those two at every load:
# agent programs 16 / 32 / 48 / 64 with chat 4 / 8 / 12 / 16 req/s (32 / 8 reused from phase A). Capacity = highest load
# at which a system holds >= 90% per-tenant attainment (chat 0.5 s + 20 ms, agent 10 s + 80 ms).
# Then reruns the two INVALID arms of 2026-10-22 on the same 4 x 4 layout they ran on: kernel-on + MTP and the 0.75x
# optimistic predictor, to see whether their stream cuts / silent chat client reproduce. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
grep -q PHASE_A_DONE /tmp/h20/mn_cap16.log || { echo "phase A did not finish"; exit 1; }
BEST=$(python3 - <<'PY'
import re
rows = []
for l in open("/tmp/h20/mn_cap16.log"):
    m = re.match(r"(disagg_p(\d+)_hc_8g_fanout_cap16_L32_r1) .*agent 10s/80ms=\s*([0-9.]+)tok/s", l)
    if m: rows.append((float(m.group(3)), int(m.group(2))))
rows.sort(reverse=True)
print(" ".join(str(p) for _, p in rows[:2]))
PY
)
echo "best disagg splits (prefill pods of 8): $BEST"
[ -n "$BEST" ] || { echo "no phase-A disagg scores parsed"; exit 1; }
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=8 GPUS_PER_POD=2 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1 FO_TEXT=1
export RA_LOG=/tmp/h20/mn_32g_calibrate.log POOL_KV_TOKENS=6696000 ATTN_BACKEND=fa3 KERNEL=lpt
RA_SLO="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5"
score() { python3 - "$1" <<'PY'
import sys; sys.path.insert(0, "/opt/tiger/sglang/tasks/pd_colocation")
from px_multi import Cluster
c = Cluster("2f608c1b1a67cf09", "cn-seed", nodes=["worker-0"])
print(sys.argv[1], c.run(0, f"/dev/shm/w/venv/bin/python /dev/shm/w/perf/fanout_per_tenant.py /dev/shm/w/mn/{sys.argv[1]} 2>&1 | tail -1", timeout=240)[1].strip()[-230:], flush=True)
PY
}
for L in "16 4" "48 12" "64 16"; do
  set -- $L; export AGENT_CONC=$1 FO_TASKS=$(( $1 / 2 )) CHAT_RATE=$2 CHAT_N=$(( $2 * 900 ))
  EXTRA_RA="$RA_SLO" REP=_cap16_L$1_r1 ARMS=coloc_slo_edf_hc bash mn_fanout_8gpu_20261018.sh
  score coloc_slo_edf_hc_8g_fanout_cap16_L$1_r1
  for P in $BEST; do
    REP=_cap16_L$1_r1 ARMS=disagg_p${P}_hc bash mn_fanout_8gpu_20261018.sh
    score disagg_p${P}_hc_8g_fanout_cap16_L$1_r1
  done
done
echo PHASE_B_DONE
# reruns of the two invalid 2026-10-22 arms, on their original 4 x 4 layout and load
export NPODS=4 GPUS_PER_POD=4 FO_TASKS=32 AGENT_CONC=64 CHAT_RATE=16 CHAT_N=14400
unset KERNEL
env SPLIT_ROWS_FA3=1 SPLIT_POLICY=1 SPLIT_FASTPATH=1 SPLIT_OVERLAP=1 VERIFY_FA3=1 EXTRA_RA="$RA_SLO" \
  REP=_kernelon_16g_mtp_r2 ARMS=coloc_slo_edf_mtp_hc bash mn_fanout_8gpu_20261018.sh
score coloc_slo_edf_mtp_hc_8g_fanout_kernelon_16g_mtp_r2
BASE=$(grep -oE "ROUTER_ARGS=.*" /tmp/h20/mn_32g_calibrate.log | tail -1 | cut -d= -f2-)
OPT=$(python3 -c "
import re,sys; b=sys.argv[1]
b=re.sub(r'--prefill-curve ([0-9:.,]+)', lambda m:'--prefill-curve '+','.join(f'{t}:{float(v)*0.75:.1f}' for t,v in (p.split(':') for p in m.group(1).split(','))), b)
print(re.sub(r'--prefill-tok-s ([0-9.]+)', lambda m:f'--prefill-tok-s {float(m.group(1))/0.75:.0f}', b))" "$BASE")
RA_OVERRIDE="$OPT" ATTN_BACKEND=fa3 KERNEL=lpt EXTRA_RA="$RA_SLO" REP=_pred_optimistic_16g_r2 ARMS=coloc_slo_edf_hc \
  bash mn_fanout_8gpu_20261018.sh
score coloc_slo_edf_hc_8g_fanout_pred_optimistic_16g_r2
echo MN_CAP16_PHASEB_DONE
