#!/bin/bash
# PD item 7, transfer-aware veto with a prefetch policy that actually pulls (2026-10-17). Under sglang's default
# best_effort, a request sent to an idle non-holder starts before any of its prefix lands, so the router's
# transfer term predicted 0 and the veto recomputed (mn_pool_veto_8gpu_20261017.sh). Here every worker runs
# --hicache-storage-prefetch-policy timeout (wait for the pull, up to 1 s + 0.25 s/Ktok) and the router predicts
# the same. Pool on in every arm; cache affinity vs veto + transfer term, interleaved x2 to measure the noise the
# best_effort A/B exposed. Agent C=16, chat 8 req/s. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
RA=$(grep -oE "ROUTER_ARGS=.*" /tmp/h20/mn_slo_calibrate.log | tail -1 | cut -d= -f2-)
export SID=bb854a10b8954223 NPODS=4 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 HC_PREFETCH=timeout
export AGENT_CONC=16 SESSIONS=48 CHAT_RATE=8 CHAT_N=2400
for rep in 1 2; do
  RUN_TAG=_8g_pooltmo_c16_r8_rep$rep python3 -u mn_run.py coloc_cache_edf_mtp_hc
  RUN_TAG=_8g_pooltmo_c16_r8_rep$rep ROUTER_ARGS="$RA --pool-kv-tokens 3348000 --prefetch-policy timeout" \
    python3 -u mn_run.py coloc_slo_edf_mtp_hc
done
echo POOL_TMO_DONE
