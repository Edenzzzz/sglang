#!/bin/bash
# PD item 7, transfer-aware veto with MANY shared prefixes (2026-10-17). With 4 hot groups (mn_hot_transfer_8gpu_
# 20261017.sh) the first few vetoed moves recompute each 16K prefix on the new node, after which every node holds all
# four locally: only 37 of 1 623 moved requests had a pool hit beyond their local hit. Here 192 groups x 16K shared
# prefixes (~3.1M tokens): fits the pool (8 x 24 GB = ~3.35M tokens) but not one node (device ~1.1M + host 48 GB
# ~0.84M), so a moved request usually finds its prefix in the pool and not locally. Pool on, timeout prefetch, agent
# C=16, chat 16 req/s. cache | veto ignoring the pool | veto + transfer, interleaved x2. Waits for the hot A/B. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
until grep -q HOT_TRANSFER_DONE /tmp/h20/mn_hot_transfer.log; do sleep 30; done
RA=$(grep -oE "ROUTER_ARGS=.*" /tmp/h20/mn_slo_calibrate.log | tail -1 | cut -d= -f2-)
export SID=bb854a10b8954223 NPODS=4 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 HC_PREFETCH=timeout
export AGENT_CONC=16 SESSIONS=48 HOT=1 HOT_GROUPS=192 HOT_PREFIX=16384 CHAT_RATE=16 CHAT_N=3840
for rep in 1 2; do
  RUN_TAG=_8g_many_c16_rep$rep python3 -u mn_run.py coloc_cache_edf_mtp_hc
  RUN_TAG=_8g_many_c16_notransfer_rep$rep ROUTER_ARGS="$RA" python3 -u mn_run.py coloc_slo_edf_mtp_hc
  RUN_TAG=_8g_many_c16_transfer_rep$rep ROUTER_ARGS="$RA --pool-kv-tokens 3348000 --prefetch-policy timeout" \
    python3 -u mn_run.py coloc_slo_edf_mtp_hc
done
echo MANY_TRANSFER_DONE
