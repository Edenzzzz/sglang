#!/bin/bash
# What does the FUSED KERNEL add at cluster scale? (16 H20s = 4 pods x 4, job 2f608c1b1a67cf09, 2026-10-22, after $1.)
# The confound-free multi-node result deliberately ran ours with KERNEL=lpt on upstream's FA3 backend -- no split, no
# fused dispatch -- to attribute the win to the schedule and router alone, and it cleared the bar there (962.7 agent
# tok/s, chat SLO attainment 96.5% vs disagg 3+1's 23.3%). That leaves the kernel's own cluster-scale contribution
# unmeasured: on one GPU it is worth ~+2.5% for our schedule and its real value is step predictability.
# So: the SAME workload and schedule (32 tasks / 64 agent programs, chat 16 req/s, 900 s, cache-first SLO router +
# worker EDF + pool) with our full fused stack ON -- split rows on FA3, per-step split policy, per-layer fastpath,
# verify on FA3 -- and the same again with MTP. Compared against the KERNEL=lpt arms of mn_fa3_16g_20261022.sh, this
# isolates the kernel's multi-node contribution with everything else held fixed. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=4 GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1 FO_TEXT=1
export RA_LOG=/tmp/h20/mn_32g_calibrate.log POOL_KV_TOKENS=6696000
export FO_TASKS=32 AGENT_CONC=64 CHAT_RATE=16 CHAT_N=14400
KON="SPLIT_ROWS_FA3=1 SPLIT_POLICY=1 SPLIT_FASTPATH=1 SPLIT_OVERLAP=1"
env $KON EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_kernelon_16g_nospec_r1 \
  ARMS=coloc_slo_edf_hc bash mn_fanout_8gpu_20261018.sh
env $KON VERIFY_FA3=1 EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_kernelon_16g_mtp_r1 \
  ARMS=coloc_slo_edf_mtp_hc bash mn_fanout_8gpu_20261018.sh
python3 - <<'PY'
import sys; sys.path.insert(0, "/opt/tiger/sglang/tasks/pd_colocation")
from px_multi import Cluster
c = Cluster("2f608c1b1a67cf09", "cn-seed", nodes=["worker-0"])
L = "/dev/shm/w/mn"
pairs = [("kernel OFF, no spec", "coloc_slo_edf_hc_8g_fanout_fa3only_16g_nospec_r1"),
         ("kernel ON,  no spec", "coloc_slo_edf_hc_8g_fanout_kernelon_16g_nospec_r1"),
         ("kernel OFF, MTP", "coloc_slo_edf_mtp_hc_8g_fanout_fa3only_16g_mtp_r1"),
         ("kernel ON,  MTP", "coloc_slo_edf_mtp_hc_8g_fanout_kernelon_16g_mtp_r1")]
for name, d in pairs:
    out = c.run(0, f"test -d {L}/{d} && /dev/shm/w/venv/bin/python /dev/shm/w/perf/fanout_per_tenant.py {L}/{d} 2>&1 | tail -2 || echo MISSING {d}", timeout=240)[1]
    print(f"## {name}\n{out}", flush=True)
PY
echo MN_KERNEL_ON_DONE
