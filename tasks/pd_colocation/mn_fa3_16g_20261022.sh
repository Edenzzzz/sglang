#!/bin/bash
# Multi-node on the SAME fair footing as the single-node table (16 H20s = 4 pods x 4, job 2f608c1b1a67cf09,
# 2026-10-22, after $1). The 32- and 16-GPU wins so far had ours on sglang's flashinfer backend (with our FA3 patches)
# and disagg on FA3; the backend-class control showed that class is half the throughput of FA3 on this workload, so
# ours was carrying a backend handicap and the comparison was not schedule-vs-schedule. Here EVERY arm runs sglang's
# FA3 backend (KERNEL=lpt, no split planning at all), so the only difference is the schedule and the routing:
#   ours        cache-first SLO router + worker EDF + deadline chunk + cluster KV pool   (coloc_slo_edf_hc)
#   disagg      PD disaggregation at 2+2 and 3+1, sglang's own router                    (disagg_p2 / disagg_p3)
# Qwen2.5-7B no spec first, then with real MTP on every role. Load as in the 16-GPU runs: 32 tasks / 64 programs,
# chat 16 req/s, 900 s. Reports agent tok/s, TTFT, ITL p99 and prefix cached fraction. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=4 GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1 FO_TEXT=1
export RA_LOG=/tmp/h20/mn_32g_calibrate.log POOL_KV_TOKENS=6696000
export FO_TASKS=32 AGENT_CONC=64 CHAT_RATE=16 CHAT_N=14400
for SPEC in nospec mtp; do
  if [ $SPEC = nospec ]; then OURS=coloc_slo_edf_hc; DIS="disagg_p2 disagg_p3"; else OURS=coloc_slo_edf_mtp_hc; DIS="disagg_p2_mtps disagg_p3_mtps"; fi
  ATTN_BACKEND=fa3 KERNEL=lpt EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" \
    REP=_fa3only_16g_${SPEC}_r1 ARMS=$OURS bash mn_fanout_8gpu_20261018.sh
  for arm in $DIS; do ATTN_BACKEND=fa3 KERNEL=lpt REP=_fa3only_16g_${SPEC}_r1 ARMS=$arm bash mn_fanout_8gpu_20261018.sh; done
done
echo MN_FA3_16G_DONE
