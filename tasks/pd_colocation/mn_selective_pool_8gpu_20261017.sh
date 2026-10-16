#!/bin/bash
# PD item 7, transfer term with a pool that keeps the shared prefixes (2026-10-17). Under HC_WRITE=write_through the
# agents' one-off ~50K-token contexts filled the 192 GB mooncake pool to 93-94%, the master evicted only above 0.95,
# BatchPut failed for lack of space, and no predicted pull landed (router realisation EWMA -> 0.0). Here the workers
# keep sglang's default write_through_selective (a block is written after its second hit: shared prefixes in,
# one-off contexts out), the router mirrors that rule (--pool-write-policy), and the master evicts above 0.85.
# 192 x 16K shared-prefix chat at 16 req/s + agent C=16, timeout prefetch. Arms interleaved x2: cache affinity | SLO
# check ignoring the pool | SLO check + transfer term; both SLO-check arms with the resident ITL check and the cache
# fallback. Spread figure written at the end (mn_ab_figure.py). Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
python3 - <<'PY'
import base64
from px_multi import Cluster
c = Cluster("bb854a10b8954223", "cn-seed", nodes=["worker-0"])
b = base64.b64encode(open("router.py", "rb").read()).decode()
print(c.run(0, "echo " + b + " | base64 -d > /dev/shm/w/tasks/pd_colocation/router.py; md5sum /dev/shm/w/tasks/pd_colocation/router.py")[1])
PY
md5sum router.py
RA=$(grep -oE "ROUTER_ARGS=.*" /tmp/h20/mn_slo_calibrate.log | tail -1 | cut -d= -f2-)
RA="$RA --chunk-tokens 2048 --check-residents 1 --no-feasible-fallback cache"
POOL="--pool-kv-tokens 3348000 --prefetch-policy timeout --pool-write-policy write_through_selective"
export SID=bb854a10b8954223 NPODS=4 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 HC_PREFETCH=timeout MC_EVICT_HIGH=0.85
export AGENT_CONC=16 SESSIONS=48 HOT=1 HOT_GROUPS=192 HOT_PREFIX=16384 CHAT_RATE=16 CHAT_N=3840
for rep in 1 2; do
  RUN_TAG=_8g_many_c16_selective_ctl_rep$rep python3 -u mn_run.py coloc_cache_edf_mtp_hc
  RUN_TAG=_8g_many_c16_selective_notransfer_rep$rep ROUTER_ARGS="$RA" python3 -u mn_run.py coloc_slo_edf_mtp_hc
  RUN_TAG=_8g_many_c16_selective_transfer_rep$rep ROUTER_ARGS="$RA $POOL" python3 -u mn_run.py coloc_slo_edf_mtp_hc
done
PYTHONPATH=/tmp/mplpkgs python3 mn_ab_figure.py figures_20261017/h20x8_manyprefix_selective_pool \
  "cache=coloc_cache_edf_mtp_hc_8g_many_c16_selective_ctl,slo_check_no_transfer=coloc_slo_edf_mtp_hc_8g_many_c16_selective_notransfer,slo_check+transfer=coloc_slo_edf_mtp_hc_8g_many_c16_selective_transfer"
echo SELECTIVE_POOL_DONE
