#!/bin/bash
# PD item 7, SLO check that also protects requests already decoding on the target node (user, 2026-10-17: "not only
# consider ttft & itl slo for the current request, but also whether adding it will violate slo for the existing
# requests on that node"). On the 4 x 16K hot-prefix mix the SLO check moved chat off its overloaded holders (chat TTFT
# -90%) but raised agent ITL +16%: moved requests' prefill chunks ride in the decode steps of agent requests already
# on the target node. --check-residents 1 rejects a node if any resident's predicted mean ITL (time spent so far +
# remaining steps at the step time with the new request + one prefill chunk per step while its prefill runs) would
# exceed its ITL SLO. Same router build in both arms; only --check-residents differs. Pool on, timeout prefetch,
# transfer term on, agent C=16, hot chat 16 req/s, interleaved x2, with a cache-affinity control.
# Rerun 2026-10-17 04:5x on the fixed TPOT prediction: the first run's router counted every in-flight request (queued
# ones included) as decoding, so ~15K node rejections came from an inflated predicted ITL (grid at 40-60 rows) and
# 56-63% of decisions found no feasible node. Rerun again 05:0x on a decode grid timed without prefill in the
# steps (calibrate_h20.py fix) and the prefill curve. Rerun 06:1x with the measured prefill/decode interference
# (ratio 1.07-1.12: serialised) and --no-feasible-fallback cache in both SLO-check arms, so that the resident check
# changes only which FEASIBLE node is chosen (the most-slack fallback ignores ITL and decided 52% of routes). Claude.
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
RA="$RA --pool-kv-tokens 3348000 --prefetch-policy timeout --chunk-tokens 2048 --no-feasible-fallback cache"
export SID=bb854a10b8954223 NPODS=4 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
# HC_WRITE=write_through: a computed prefix reaches the pool at once (default write_through_selective waits
# for a second hit, so the router's pool mirror predicted pulls of pages that were not there).
export KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 HC_PREFETCH=timeout HC_WRITE=write_through
export AGENT_CONC=16 SESSIONS=48 HOT=1 HOT_GROUPS=4 HOT_PREFIX=16384 CHAT_RATE=16 CHAT_N=3200
for rep in 1 2; do
  RUN_TAG=_8g_hot_c16_cleangrid2_ctl_rep$rep python3 -u mn_run.py coloc_cache_edf_mtp_hc
  RUN_TAG=_8g_hot_c16_cleangrid2_residents0_rep$rep ROUTER_ARGS="$RA --check-residents 0" python3 -u mn_run.py coloc_slo_edf_mtp_hc
  RUN_TAG=_8g_hot_c16_cleangrid2_residents1_rep$rep ROUTER_ARGS="$RA --check-residents 1" python3 -u mn_run.py coloc_slo_edf_mtp_hc
done
echo RESIDENT_ITL_DONE
