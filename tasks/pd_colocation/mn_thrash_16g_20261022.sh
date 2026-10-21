#!/bin/bash
# Point 2, the unrun experiment: cache THRASHING in agentic serving (ThunderAgent's setting), 16 H20s = 4 pods x 4,
# job 2f608c1b1a67cf09, 2026-10-22. Both systems on FA3 (ours: split rows + policy + fastpath; disagg:
# --attention-backend fa3). Agent load is program CONCURRENCY, fixed at 32; the WORKING SET is swept by the number of
# distinct agent tasks, each carrying its own 64K prefix: FO_TASKS 16 / 64 / 128 = ~1M / 4M / 8M KV tokens live against
# a cluster KV capacity of ~6.7M (POOL_KV_TOKENS), so the last point evicts and must re-prefill.
# The claim under test is point 2's: colocation plus the shared pool keeps prefixes resident (or re-fetches them over
# RDMA) where disagg's prefill pods must recompute them, so ours should degrade more slowly as the working set grows.
# Reported per point: agent tok/s, TTFT, ITL p99 and FANOUT prefix cached fraction (the re-prefill proxy).
# Starts after $1 (the GLM soak) so the pods are free. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=4 GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1 FO_TEXT=1
export RA_LOG=/tmp/h20/mn_32g_calibrate.log POOL_KV_TOKENS=6696000
export AGENT_CONC=32 CHAT_RATE=8 CHAT_N=7200 FO_PREFIX=65536
for T in 16 64 128; do
  export FO_TASKS=$T
  SPLIT_ROWS_FA3=1 SPLIT_POLICY=1 SPLIT_FASTPATH=1 \
    EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_thrash_t${T}_16g_r1 \
    ARMS=coloc_slo_edf_hc bash mn_fanout_8gpu_20261018.sh
  ATTN_BACKEND=fa3 REP=_thrash_t${T}_16g_r1 ARMS=disagg_p2 bash mn_fanout_8gpu_20261018.sh
done
echo MN_THRASH_16G_DONE
