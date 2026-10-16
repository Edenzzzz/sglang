#!/bin/bash
# PD item 7: what the SLO check does when NO node can meet the SLO (2026-10-17). On the 192 x 16K shared-prefix mix
# 99% of decisions (3 896 of 3 940) had no feasible node, and the most-slack fallback scattered requests off their
# prefixes: chat TTFT 28 -> 73 s, agent 30 -> 49 s vs cache affinity. --no-feasible-fallback cache takes the top-hit
# node instead (least added work). Arms interleaved x2: cache affinity | SLO check + transfer term + resident ITL
# check + cache fallback. Same mix as mn_manyprefix_transfer_8gpu_20261017.sh. Waits for the resident-ITL A/B. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
until grep -q RESIDENT_ITL_DONE /tmp/h20/mn_resident_itl.log 2>/dev/null; do sleep 30; done
python3 - <<'PY'
import base64
from px_multi import Cluster
c = Cluster("bb854a10b8954223", "cn-seed", nodes=["worker-0"])
b = base64.b64encode(open("router.py", "rb").read()).decode()
print(c.run(0, "echo " + b + " | base64 -d > /dev/shm/w/tasks/pd_colocation/router.py; md5sum /dev/shm/w/tasks/pd_colocation/router.py")[1])
PY
md5sum router.py
RA=$(grep -oE "ROUTER_ARGS=.*" /tmp/h20/mn_slo_calibrate.log | tail -1 | cut -d= -f2-)
RA="$RA --pool-kv-tokens 3348000 --prefetch-policy timeout --chunk-tokens 2048 --check-residents 1"
export SID=bb854a10b8954223 NPODS=4 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
# HC_WRITE=write_through: a computed prefix reaches the pool at once (default write_through_selective waits
# for a second hit, so the router's pool mirror predicted pulls of pages that were not there).
export KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 HC_PREFETCH=timeout HC_WRITE=write_through
export AGENT_CONC=16 SESSIONS=48 HOT=1 HOT_GROUPS=192 HOT_PREFIX=16384 CHAT_RATE=16 CHAT_N=3840
for rep in 1 2; do
  RUN_TAG=_8g_many_c16_fbcache_ctl_rep$rep python3 -u mn_run.py coloc_cache_edf_mtp_hc
  RUN_TAG=_8g_many_c16_fbcache_rep$rep ROUTER_ARGS="$RA --no-feasible-fallback cache" python3 -u mn_run.py coloc_slo_edf_mtp_hc
done
echo OVERLOAD_FALLBACK_DONE
