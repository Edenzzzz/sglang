#!/bin/bash
# PD item 7, transfer-aware feasibility (2026-10-17): with the cluster-wide mooncake L3 KV pool ON (_hc), does
# predicting the best-effort prefix pull a queued request gets change the veto's routing, and does it help?
# Agent C=16, chat 8 req/s, final single-GPU stack on every worker. Arms (all pool on):
#   coloc_cache_edf_mtp_hc   cache affinity only
#   coloc_slo_edf_mtp_hc     veto, router ignores the pool (no transfer term)
#   coloc_slo_edf_mtp_hc     veto + transfer term (--pool-kv-tokens = 8 workers x 32 GB / 57 344 B per token)
# Router logs ttft_pred / ttft_obs / pull_pred per request for the predictor-error analysis.
# Waits for mn_slo_veto_8gpu_20261017.sh (same pods; reuses its calibration). Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
until grep -q SLO_VETO_DONE /tmp/h20/mn_slo_veto.log; do sleep 30; done
# The transfer-aware router.py is pushed only now, so the veto experiment above ran one router throughout.
python3 - <<'PY'
import base64
from px_multi import Cluster
c = Cluster("bb854a10b8954223", "cn-seed", nodes=["worker-0"])
b = base64.b64encode(open("router.py", "rb").read()).decode()
print(c.run(0, "echo " + b + " | base64 -d > /dev/shm/w/tasks/pd_colocation/router.py; md5sum /dev/shm/w/tasks/pd_colocation/router.py")[1])
PY
md5sum router.py
RA=$(grep -oE "ROUTER_ARGS=.*" /tmp/h20/mn_slo_calibrate.log | tail -1 | cut -d= -f2-)
export SID=bb854a10b8954223 NPODS=4 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export KERNEL=mtpsplit PREEMPT=1 CHUNK=2048
export AGENT_CONC=16 SESSIONS=48 CHAT_RATE=8 CHAT_N=2400
RUN_TAG=_8g_pool_c16_r8 python3 -u mn_run.py coloc_cache_edf_mtp_hc
RUN_TAG=_8g_pool_c16_r8_notransfer ROUTER_ARGS="$RA" python3 -u mn_run.py coloc_slo_edf_mtp_hc
RUN_TAG=_8g_pool_c16_r8_transfer ROUTER_ARGS="$RA --pool-kv-tokens 4468000" python3 -u mn_run.py coloc_slo_edf_mtp_hc
echo POOL_VETO_DONE
