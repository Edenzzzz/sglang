#!/bin/bash
# Rerun of the 8-GPU C=32 / chat 8 req/s arms behind OUR router (router.py), 2026-10-16:
#  - coloc_cache_mtps: its agent client aborted when one upstream POST hit EPIPE on a stale keep-alive
#    connection (router retried only ServerDisconnectedError);
#  - both: router.py's aiohttp session used the default connector, capped at 100 concurrent upstream
#    connections; at C=32 + chat 8 req/s ~200+ requests are in flight, so the cap queued requests inside
#    the router -- the suspected cause of the chat "starved tail" (mean TTFT 19.7 / 35.1 s vs stock 7.3 s).
#    router.py now uses limit=0 and keepalive 3 s. Waits for
# mn_disagg_rerun_8gpu_20261016.sh (same pods) to print DISAGG_RERUN_DONE. Claude, 2026-10-16.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
until grep -q DISAGG_RERUN_DONE /tmp/h20/mn_disagg_rerun.log; do sleep 30; done
export SID=bb854a10b8954223 NPODS=4 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export AGENT_CONC=32 CHAT_RATE=8 SESSIONS=96 CHAT_N=2400 RUN_TAG=_8g_c32_r8_rerun2
KERNEL=lpt PREEMPT=0 python3 -u mn_run.py coloc_cache_mtps
KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 python3 -u mn_run.py coloc_cache_edf_mtp
echo CACHE_MTPS_RERUN_DONE
