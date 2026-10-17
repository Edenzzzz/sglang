#!/bin/bash
# PD item 7, pool pulls with host memory as a transfer buffer only (2026-10-17). The prefetch diagnostic showed every
# deferred / unfulfilled pool prefetch was host_capacity (553K / 608K tokens): write-through fills the L2 host pool
# (100% used) and sglang defers L3 staging instead of evicting L2. HC_HOST_MODE=buffer_only on every worker
# (--hicache-host-memory-mode buffer_only). 192 x 16K shared-prefix chat at 8 req/s + agent C=16, pool on, timeout
# prefetch, write_through_selective; cache affinity | SLO check + transfer term (resident check, cache fallback),
# interleaved x2, spread figure. Waits for the margin A/B. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
until grep -q TTFT_MARGIN_DONE /tmp/h20/mn_ttft_margin.log 2>/dev/null; do sleep 30; done
python3 - <<'PY'
import base64
from px_multi import Cluster
c = Cluster("bb854a10b8954223", "cn-seed", nodes=["worker-0"])
b = base64.b64encode(open("router.py", "rb").read()).decode()
print(c.run(0, "echo " + b + " | base64 -d > /dev/shm/w/tasks/pd_colocation/router.py; md5sum /dev/shm/w/tasks/pd_colocation/router.py")[1])
PY
md5sum router.py
RA=$(grep -oE "ROUTER_ARGS=.*" /tmp/h20/mn_slo_calibrate.log | tail -1 | cut -d= -f2-)
RA="$RA --pool-kv-tokens 3348000 --prefetch-policy timeout --pool-write-policy write_through_selective --chunk-tokens 2048 --check-residents 1 --no-feasible-fallback cache"
export SID=bb854a10b8954223 NPODS=4 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 HC_PREFETCH=timeout MC_EVICT_HIGH=0.85 HC_HOST_MODE=buffer_only
export AGENT_CONC=16 SESSIONS=48 HOT=1 HOT_GROUPS=192 HOT_PREFIX=16384 CHAT_RATE=8 CHAT_N=3840
for rep in 1 2; do
  RUN_TAG=_8g_many_r8_bufferonly_ctl_rep$rep python3 -u mn_run.py coloc_cache_edf_mtp_hc
  RUN_TAG=_8g_many_r8_bufferonly_transfer_rep$rep ROUTER_ARGS="$RA" python3 -u mn_run.py coloc_slo_edf_mtp_hc
done
PYTHONPATH=/tmp/mplpkgs python3 mn_ab_figure.py figures_20261017/h20x8_manyprefix_r8_bufferonly \
  "cache=coloc_cache_edf_mtp_hc_8g_many_r8_bufferonly_ctl,slo_check+transfer=coloc_slo_edf_mtp_hc_8g_many_r8_bufferonly_transfer"
echo BUFFERONLY_DONE
