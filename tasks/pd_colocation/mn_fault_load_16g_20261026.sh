#!/bin/bash
# Fault tolerance at a chosen load (FAULT_LOAD), full stack, 16 H20 GPUs (job 2f608c1b1a67cf09, 4 pods x 4 GPUs, Qwen2.5-7B TP1 = 16 workers),
# 2026-10-26. Same arms, workload and SLOs as stage 3 (mn_stage3_a6_16g_20261025.sh) at 32 agent programs + chat 8/s,
# where every system but stock PD holds its SLOs. At t = 300 s every worker on one pod is killed (FAULT_AT / FAULT_POD
# in mn_run.py): 4 of 16 GPUs.
#   ours      coloc_slo_edf_mtp_hc (A6 + 4-mode MTP + EDF + SLO-checked cache-first routing + pool), pod 1 killed
#   cond      conditional disaggregation on the same workers (prefill workers = pod 0), pod 1 (decode) killed
#   stock PD  2 + 2 pods, health-check ejection on (DISAGG_HEALTH=1); pod 1 (prefill) and, separately, pod 2 (decode)
#             killed
# Scored per tenant overall (fanout_per_tenant.py, PAPER line) and in windows before / after the fault
# (tasks/perf_model/spike_windows.py --spike 300:900). Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=4 GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1 FO_TEXT=1
export ATTN_BACKEND=fa3 POOL_KV_TOKENS=6696000 FO_TASKS=32
export RA_LOG=/tmp/h20/mn_a6_calibrate_16g.log FAULT_AT=300 DISAGG_HEALTH=1
grep -q ROUTER_ARGS= $RA_LOG || { echo NO_CALIBRATION; exit 1; }
RA_SLO="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5"
# FAULT_LOAD="C:rate" (default 32:8): agent programs and chat req/s; FAULT_ARMS picks arms (default all four).
IFS=: read -r C RATE <<< "${FAULT_LOAD:-32:8}"
export AGENT_CONC=$C CHAT_RATE=$RATE CHAT_N=$(( RATE * 900 ))
L=$([ "$C" = 32 ] && echo "" || echo "_L$C")
R=${R:-r1}
ARMS_ON=${FAULT_ARMS:-"ours cond pd_p1 pd_p2"}
case " $ARMS_ON " in *" ours "*)
FAULT_POD=1 KERNEL=a6 EXTRA_RA="$RA_SLO" REP=_fault_p1${L}_$R ARMS=coloc_slo_edf_mtp_hc bash mn_fanout_8gpu_20261018.sh ;; esac
case " $ARMS_ON " in *" cond "*)
FAULT_POD=1 KERNEL=a6 COND_THRESHOLD=2048 COND_FRAC=0.7 REP=_fault_p1${L}_$R ARMS=cond_p1_edf_mtp_hc bash mn_fanout_8gpu_20261018.sh ;; esac
case " $ARMS_ON " in *" pd_p1 "*)
FAULT_POD=1 REP=_fault_p1${L}_$R ARMS=disagg_p2_mtps_hc bash mn_fanout_8gpu_20261018.sh ;; esac
case " $ARMS_ON " in *" pd_p2 "*)
FAULT_POD=2 REP=_fault_p2${L}_$R ARMS=disagg_p2_mtps_hc bash mn_fanout_8gpu_20261018.sh ;; esac
echo MN_FAULT_FULLSTACK_DONE
