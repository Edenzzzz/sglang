#!/bin/bash
# Attribution arms for the 32-GPU sweep (2026-10-21): ours used the cluster-wide mooncake KV pool and disagg did not
# (prefix cached fraction 0.975 vs 0.62-0.63). Same load as mn_scale32_20261021.sh; after pid $1: ours without the
# pool (coloc_slo_edf: SLO-checked hit-first routing + EDF, local caches only, router told there is no pool) and disagg at its best ratio from the
# sweep (BEST_P prefill pods) WITH the same pool on every worker (_hc). One repeat each. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=8 GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b
export FO_TEXT=1 RA_LOG=/tmp/h20/mn_32g_calibrate.log FO_TASKS=64 AGENT_CONC=128 CHAT_RATE=32 CHAT_N=28800
export POOL_KV_TOKENS=13392000
POOL_KV_TOKENS=0 EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_text_32g_r1 ARMS=coloc_slo_edf bash mn_fanout_8gpu_20261018.sh
REP=_text_32g_r1 ARMS=disagg_p${BEST_P}_hc bash mn_fanout_8gpu_20261018.sh
echo MN_SCALE32_FAIR_DONE
