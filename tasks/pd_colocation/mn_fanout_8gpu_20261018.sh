#!/bin/bash
# PD point 2b on 8 H20s (job bb854a10b8954223, 4 pods x 2 GPUs): fan-out agents on shared long task prefixes with
# rotating popularity (drivers/serving/fanout_agents.py via mn_bench.sh WORKLOAD=fanout), + ShareGPT chat. 16 tasks x
# 64K-token prefixes (~1M-token hot set, above one pool's KV), 32 programs kept active, 4 rounds, 5 s tool gaps,
# Zipf 1.2, hot set rotating every 180 s, 900 s. Arms, MTP on every role:
#   disagg_2p2d_mtps           PD disaggregation (2 prefill + 2 decode pods), sglang router, stock MTP on every role,
#                              stock kernel and the clean upstream tree (SGL_PY=up_clean, default chunk), as in
#                              mn_scaling_20261018.sh: on our patched tree its prefill workers assert at warmup
#                              (disaggregation/prefill.py `batch.spec_info is result.next_draft_input`)
#   coloc_cache_edf_mtp        colocation, prefix affinity only (no pool)
#   coloc_slo_edf_mtp_hc       ours: SLO-checked hit-first routing + mooncake pool + transfer term + EDF + 4-mode MTP
# Claude, 2026-10-18.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
RA=$(grep -oE "ROUTER_ARGS=.*" ${RA_LOG:-/tmp/h20/mn_slo_calibrate.log} | tail -1 | cut -d= -f2-)
export SID=bb854a10b8954223 MN_CP=cn-seed NPODS=4 GPUS_PER_POD=2
export KERNEL=${KERNEL:-mtpsplit} PREEMPT=1 CHUNK=${CHUNK:-2048} HC_PREFETCH=timeout EDF_UNCACHED=1
export WORKLOAD=fanout FO_TASKS=${FO_TASKS:-16} FO_PREFIX=${FO_PREFIX:-65536} AGENT_CONC=${AGENT_CONC:-32} FO_ROUNDS=4
export FO_GAP=5 FO_ZIPF=1.2 FO_ROTATE=${FO_ROTATE:-180} FO_DURATION=900 CHAT_SHAREGPT=1 CHAT_RATE=${CHAT_RATE:-8} CHAT_N=${CHAT_N:-7200}
for arm in ${ARMS:-disagg_2p2d_mtps coloc_cache_edf_mtp coloc_slo_edf_mtp_hc}; do
  case $arm in
    coloc_slo_*) RUN_TAG=_8g_fanout${REP:-} ROUTER_ARGS="$RA --pool-kv-tokens 3348000 --prefetch-policy timeout ${EXTRA_RA:-}" \
                            python3 -u mn_run.py $arm ;;
    disagg_*) (unset CHUNK; RUN_TAG=_8g_fanout${REP:-} SGL_PY=/dev/shm/w/up_clean/python KERNEL=lpt PREEMPT=0 \
                 python3 -u mn_run.py $arm) ;;
    *) RUN_TAG=_8g_fanout${REP:-} python3 -u mn_run.py $arm ;;
  esac
done
echo FANOUT8_DONE
