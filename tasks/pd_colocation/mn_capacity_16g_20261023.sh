#!/bin/bash
# Paper-grade multi-node comparison: CAPACITY AT SLO, not a ratio at one overload point (16 H20s as 8 pods x 2 GPUs,
# job 2f608c1b1a67cf09, 2026-10-23). The earlier 16-GPU table compared at a load sized to OUR capacity, where disagg
# was saturated (chat attainment 3.8-23%), so its goodput ratios (7-44x) divided by a near-zero baseline. Three
# fairness fixes, each one a reviewer would ask for:
#   1. load SWEPT: agent programs 16 / 32 / 48 / 64 with chat 4 / 8 / 12 / 16 req/s; the metric is the highest load at
#      which each system holds >= 90% per-tenant attainment (chat 0.5 s + 20 ms, agent 10 s + 80 ms) -- DistServe-style
#      goodput -- plus the full attainment-vs-load curve
#   2. P:D swept at WORKER granularity: 8 pods x 2 GPUs gives 1..7 prefill pods of 8 (2..14 of 16 GPUs), not the
#      three pod-level splits 4 x 4 allows
#   3. disagg gets the cluster KV pool too (`_hc`: HiCache + mooncake store on its prefill pods), as ours does
# Phase A: every P:D split at the middle load (32 programs, chat 8 req/s) -> keep the best two. Phase B: ours and the two
# best disagg splits across all four loads. Every arm on sglang's FA3 backend; ours is the colocated schedule + router
# with the kernel off (KERNEL=lpt), so the comparison is schedule vs schedule. Qwen2.5-7B, no spec. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=8 GPUS_PER_POD=2 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1 FO_TEXT=1
export RA_LOG=/tmp/h20/mn_32g_calibrate.log POOL_KV_TOKENS=6696000 ATTN_BACKEND=fa3 KERNEL=lpt
RA_SLO="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5"
load() { export AGENT_CONC=$1 FO_TASKS=$(( $1 / 2 )) CHAT_RATE=$2 CHAT_N=$(( $2 * 900 )); }
score() { python3 - "$1" <<'PY'
import sys; sys.path.insert(0, "/opt/tiger/sglang/tasks/pd_colocation")
from px_multi import Cluster
c = Cluster("2f608c1b1a67cf09", "cn-seed", nodes=["worker-0"])
d = f"/dev/shm/w/mn/{sys.argv[1]}"
print(sys.argv[1], c.run(0, f"/dev/shm/w/venv/bin/python /dev/shm/w/perf/fanout_per_tenant.py {d} 2>&1 | tail -1", timeout=240)[1].strip()[-230:], flush=True)
PY
}
echo "##### PHASE A: P:D sweep at 32 programs / chat 8"
load 32 8
for P in 1 2 3 4 5 6 7; do
  REP=_cap16_L32_r1 ARMS=disagg_p${P}_hc bash mn_fanout_8gpu_20261018.sh
  score disagg_p${P}_hc_8g_fanout_cap16_L32_r1
done
EXTRA_RA="$RA_SLO" REP=_cap16_L32_r1 ARMS=coloc_slo_edf_hc bash mn_fanout_8gpu_20261018.sh
score coloc_slo_edf_hc_8g_fanout_cap16_L32_r1
echo PHASE_A_DONE
