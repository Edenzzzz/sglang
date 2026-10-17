#!/bin/bash
# pdmux + EDF on 8 H20s with backlog-triggered SM sharing (apply_pdmux_backlog_share.py): a split-prefill batch runs on
# the whole GPU when >= PDMUX_SHARE uncached prompt tokens wait, else in pdmux's decode-isolating SM split. Thresholds
# 32K (about one cold agent prompt queued) and 128K, chat 16 req/s first, then the better one at 32 req/s. Same cohort
# and router as mn_pdmux_8gpu_20261017.sh (whose baselines are the reference). Claude, 2026-10-17.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=bb854a10b8954223 NPODS=4 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export AGENT_CONC=16 SESSIONS=48
for spec in ${POINTS:-16:32768 16:131072}; do
  R=${spec%%:*}; T=${spec##*:}
  N=$((R * 120)); [ $N -lt 2400 ] && N=2400
  TAGT=$T; [ "${PDMUX_SHARE_DEADLINE:-0}" = 1 ] && TAGT=deadline
  export CHAT_RATE=$R CHAT_N=$N RUN_TAG=_8g_pdmuxshare${TAGT}_c16_r${R}${REP_SUFFIX:-}
  KERNEL=lpt PREEMPT=0 EDF_UNCACHED=1 PDMUX_DIV=30 MEM_FRAC=0.70 PDMUX_SHARE=$T python3 -u mn_run.py coloc_cache_edf_pdmux
done
echo PDMUX_SHARE_DONE
