#!/bin/bash
# Why pool pulls do not land (2026-10-17). In the 192-prefix selective-pool A/B at chat 8 req/s, 111 decisions
# predicted a pull of ~21K tokens and ~3.8K landed; the workers log "HiCache storage prefetch reissue cap reached
# ... attempts=8" for ~107 requests -- each prefetch was retried 8 times and dropped. The retry reasons (rate limit,
# no host staging space, L3 hit below threshold, ...) only reach HiCache's storage metrics, so this reruns the
# SLO-check + transfer arm once with --enable-metrics and saves every worker's /metrics before teardown.
# Waits for mn_selective_pool_r8_8gpu_20261017.sh. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
until grep -q SELECTIVE_POOL_R8_DONE /tmp/h20/mn_selective_pool_r8.log; do sleep 30; done
RA=$(grep -oE "ROUTER_ARGS=.*" /tmp/h20/mn_slo_calibrate.log | tail -1 | cut -d= -f2-)
RA="$RA --chunk-tokens 2048 --check-residents 1 --no-feasible-fallback cache"
POOL="--pool-kv-tokens 3348000 --prefetch-policy timeout --pool-write-policy write_through_selective"
export SID=bb854a10b8954223 NPODS=4 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 HC_PREFETCH=timeout MC_EVICT_HIGH=0.85
export AGENT_CONC=16 SESSIONS=48 HOT=1 HOT_GROUPS=192 HOT_PREFIX=16384 CHAT_RATE=8 CHAT_N=3840
export WORKER_EXTRA=--enable-metrics SCRAPE_METRICS=1
RUN_TAG=_8g_many_c16_r8_prefetchdiag ROUTER_ARGS="$RA $POOL" python3 -u mn_run.py coloc_slo_edf_mtp_hc
echo PREFETCH_DIAG_DONE
