#!/bin/bash
# PD item 7, transfer-aware veto where requests WITH a prefix must move (2026-10-17). In the TraceLab + ShareGPT
# mix the veto only ever moved cold first-round agent requests (local and pool hit 0), so the transfer term
# predicted a pull for 1 of 99 agent decisions and could not be tested (mn_pool_timeout_8gpu_20261017.sh). Here
# the chat tenant is HOT: 4 groups sharing 16K system prompts at 16 req/s, so cache affinity piles each group on
# its holder and the veto must move requests whose 16K prefix sits in the pool (pull ~0.9 GB, ~0.1 s, vs ~2.4 s
# recompute). Pool on, timeout prefetch on workers and router, agent C=16. Arms interleaved x2:
#   coloc_cache_edf_mtp_hc   cache affinity
#   coloc_slo_edf_mtp_hc     veto, router ignores the pool (prices a moved request as a recompute)
#   coloc_slo_edf_mtp_hc     veto + transfer term
# Workers carry edf_one_chunked_req.patch (applied to the pods' tree 2026-10-17 02:5x): the first run crashed a
# worker on `assert self.chunked_req is None`. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
RA=$(grep -oE "ROUTER_ARGS=.*" /tmp/h20/mn_slo_calibrate.log | tail -1 | cut -d= -f2-)
export SID=bb854a10b8954223 NPODS=4 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 HC_PREFETCH=timeout
export AGENT_CONC=16 SESSIONS=48 HOT=1 HOT_GROUPS=4 HOT_PREFIX=16384 CHAT_RATE=16 CHAT_N=3200
for rep in 1 2; do
  RUN_TAG=_8g_hot_c16_rep$rep python3 -u mn_run.py coloc_cache_edf_mtp_hc
  RUN_TAG=_8g_hot_c16_notransfer_rep$rep ROUTER_ARGS="$RA" python3 -u mn_run.py coloc_slo_edf_mtp_hc
  RUN_TAG=_8g_hot_c16_transfer_rep$rep ROUTER_ARGS="$RA --pool-kv-tokens 3348000 --prefetch-policy timeout" \
    python3 -u mn_run.py coloc_slo_edf_mtp_hc
done
echo HOT_TRANSFER_DONE
