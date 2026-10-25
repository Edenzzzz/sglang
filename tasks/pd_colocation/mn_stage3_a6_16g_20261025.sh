#!/bin/bash
# Stage 3, full stack, 16 H20 GPUs (job 2f608c1b1a67cf09, 4 pods x 4 GPUs, Qwen2.5-7B TP1 = 16 workers), 2026-10-25.
#   ours      coloc_slo_edf_mtp_hc, KERNEL=a6: SLO-checked cache-first routing + mooncake pool + EDF with chunk preemption
#             + the 4-mode MTP step on the unified FA3 + FA2 dispatcher (A6, FA3 backend), EAGLE3 head
#   stock PD  disagg_p<P>_mtps_hc: sglang PD disaggregation on clean upstream, stock MTP on every role, FA3 backend, pool;
#             P of 4 pods prefill, swept P = 1 / 2 / 3 (phase A), the best two kept for phase B
#   cond      cond_p1_edf_mtp_hc: Dynamo's conditional disaggregation (remote prefill when the effective uncached length
#             > 2048 tokens AND > 0.7 of the prompt) on the same A6 workers as ours: only placement differs
# Workload: fan-out agents on shared 64K task prefixes (Zipf 1.2, popularity rotating every 180 s, 4 rounds, 5 s tool
# gaps) + ShareGPT chat with the chat template, 900 s. Load: agent programs C with chat C/4 req/s. Then a mixture spike:
# a second fan-out wave of 32 programs from t = 300 s for 300 s on top of 64 + 16.
# The router's step model is calibrated on an A6 + MTP worker first. Score: per-tenant SLO attainment (chat 0.5 s TTFT +
# 20 ms TPOT, agent 10 s + 80 ms) and SLO goodput (fanout_per_tenant.py). Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=4 GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1 FO_TEXT=1
export ATTN_BACKEND=fa3 POOL_KV_TOKENS=6696000 FO_TASKS=32
CAL=/tmp/h20/mn_a6_calibrate_16g.log; mkdir -p /tmp/h20
if ! grep -q ROUTER_ARGS= $CAL 2>/dev/null; then
  KERNEL=a6 EDF=1 SPEC=1 PREEMPT=1 CHUNK=2048 EDF_UNCACHED=1 python3 -u mn_run.py calibrate > $CAL 2>&1
  grep -q ROUTER_ARGS= $CAL || { echo CALIBRATE_FAILED; tail -20 $CAL; exit 1; }
fi
echo "router: $(grep -oE 'ROUTER_ARGS=.*' $CAL | tail -1 | cut -c1-200)"
export RA_LOG=$CAL
RA_SLO="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5"
score() { python3 - "$1" <<'PY'
import sys; sys.path.insert(0, "/opt/tiger/sglang/tasks/pd_colocation")
from px_multi import Cluster
c = Cluster("2f608c1b1a67cf09", "cn-seed", nodes=["worker-0"])
print("SCORE", sys.argv[1], c.run(0, f"/dev/shm/w/venv/bin/python /dev/shm/w/perf/fanout_per_tenant.py /dev/shm/w/mn/{sys.argv[1]} 2>&1 | tail -1", timeout=240)[1].strip()[-260:], flush=True)
PY
}
load() { export AGENT_CONC=$1 CHAT_RATE=$2 CHAT_N=$(( $2 * 900 )); }
ours() { KERNEL=a6 EXTRA_RA="$RA_SLO" REP=$1 ARMS=coloc_slo_edf_mtp_hc bash mn_fanout_8gpu_20261018.sh; score coloc_slo_edf_mtp_hc_8g_fanout$1; }
pd() { REP=$2 ARMS=disagg_p$1_mtps_hc bash mn_fanout_8gpu_20261018.sh; score disagg_p$1_mtps_hc_8g_fanout$2; }
cond() { KERNEL=a6 COND_THRESHOLD=2048 COND_FRAC=0.7 REP=$1 ARMS=cond_p1_edf_mtp_hc bash mn_fanout_8gpu_20261018.sh; score cond_p1_edf_mtp_hc_8g_fanout$1; }
# Phase A: stock PD P:D split at 64 + 16
load 64 16
for P in 1 2 3; do pd $P _s3_L64_r1; done
BEST=$(grep -E "^SCORE disagg_p[0-9]_mtps_hc_8g_fanout_s3_L64_r1" ${STAGE3_LOG:?} | python3 -c "
import re,sys
rows=[]
for l in sys.stdin:
    m=re.match(r'SCORE disagg_p(\d)_.*agent 10s/80ms=\s*([0-9.]+)tok/s', l)
    if m: rows.append((float(m.group(2)), m.group(1)))
rows.sort(reverse=True); print(' '.join(p for _, p in rows[:2]))")
[ -n "$BEST" ] || BEST="1 2"
echo "best stock PD splits (prefill pods of 4): $BEST"
# Phase B: ours / cond / best PD splits across load
for L in "64 16" "32 8" "96 24" "128 32"; do
  set -- $L; load $1 $2
  ours _s3_L$1_r1
  cond _s3_L$1_r1
  if [ $1 != 64 ]; then for P in $BEST; do pd $P _s3_L$1_r1; done; fi
done
# Mixture spike on 64 + 16
load 64 16; export FO_SPIKE=300:32:300
ours _s3_spike_r1; cond _s3_spike_r1; for P in $BEST; do pd $P _s3_spike_r1; done
unset FO_SPIKE
echo STAGE3_DONE
