#!/bin/bash
# Stage 3 for Llama-3.3-70B, full stack, 16 H20 GPUs (job 2f608c1b1a67cf09, 4 pods x 4 GPUs, TP4 = 4 workers, one per
# pod; trained EAGLE3 head llama70b_eagle3_sgl, real acceptance on the natural-text fan-out), 2026-10-26. Same arms and
# protocol as the Qwen2.5-7B driver (mn_stage3_a6_16g_20261025.sh); a P:D split is whole pods (P of 4 workers prefill).
# Prerequisite: llama70b + llama70b_eagle3_sgl on every pod's /dev/shm/w (copy only when no benchmark runs).
# Load is scaled to Llama's single-node capacity (stage 2: ~2 programs + 0.5 chat/s per pod near capacity), so
# programs C = 4 / 8 / 12 / 16 with chat C/4 req/s, 32K fan-out prefixes on 8 tasks (64K x 32 tasks would not fit a
# pod's KV). The per-tenant SLO scorer prints the paper's matrix (PAPER line) before the original line.
# Original Qwen header follows.
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
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=4 GPUS_PER_POD=4 TP=4 MODEL_DIR=llama70b DRAFT_DIR=llama70b_eagle3_sgl
export CHAT_TEMPLATE=1 FO_TEXT=1 YARN_FACTOR=0 CTX_LEN=131072
# Pool sized to the same host bytes as Qwen's 6.7M tokens: Llama-70B KV is 320 KB/token (80 layers x 8 heads x 128 x 2 x 2 B).
export ATTN_BACKEND=fa3 POOL_KV_TOKENS=1170000 FO_TASKS=${FO_TASKS:-8} FO_PREFIX=${FO_PREFIX:-32768}
for w in 0 1 2 3; do
  ssh -o StrictHostKeyChecking=no -p 22222 $SID.worker_$w.seedjob.wenxuan-tan@merlin-ssh-proxy-cn.byted.org \
    "test -f /dev/shm/w/llama70b/config.json && test -f /dev/shm/w/llama70b_eagle3_sgl/config.json" 2>/dev/null \
    || { echo "LLAMA_WEIGHTS_MISSING worker-$w"; exit 1; }
done
CAL=/tmp/h20/mn_a6_calibrate_16g_llama.log; mkdir -p /tmp/h20
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
# R: repeat tag. Repeats > 1 reuse phase A's splits (BEST_SPLITS) and skip it.
R=${R:-r1}
# Phase A: stock PD P:D split at 64 + 16
load 8 2
if [ -n "${BEST_SPLITS:-}" ]; then BEST=$BEST_SPLITS; else
for P in 1 2 3; do pd $P _s3llama_L8_r1; done
BEST=$(grep -E "^SCORE disagg_p[0-9]_mtps_hc_8g_fanout_s3llama_L8_r1" ${STAGE3_LOG:?} | python3 -c "
import re,sys
rows=[]
for l in sys.stdin:
    m=re.match(r'SCORE disagg_p(\d)_.*agent 10s/80ms=\s*([0-9.]+)tok/s', l)
    if m: rows.append((float(m.group(2)), m.group(1)))
rows.sort(reverse=True); print(' '.join(p for _, p in rows[:2]))")
fi
[ -n "$BEST" ] || BEST="1 2"
echo "best stock PD splits (prefill pods of 4): $BEST"
# Phase B: ours / cond / best PD splits across load
for L in "8 2" "4 1" "12 3" "16 4"; do
  set -- $L; load $1 $2
  ours _s3llama_L$1_$R
  cond _s3llama_L$1_$R
  if [ $1 != 8 ] || [ $R != r1 ]; then for P in $BEST; do pd $P _s3llama_L$1_$R; done; fi
done
# Mixture spike on 8 + 2: a second wave of 4 programs from t = 300 s for 300 s
load 8 2; export FO_SPIKE=300:4:300
ours _s3llama_spike_$R; cond _s3llama_spike_$R; for P in $BEST; do pd $P _s3llama_spike_$R; done
unset FO_SPIKE
echo STAGE3_LLAMA_DONE
