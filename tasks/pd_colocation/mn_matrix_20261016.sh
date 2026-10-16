#!/bin/bash
# H20 multi-node matrix (2026-10-16), 8 pods x 2 H20 = 16 TP1 workers, EAGLE3 MTP everywhere.
# Workload: frozen-style TraceLab 32-64K cohort, SESSIONS x TURNS rounds at program concurrency
# AGENT_CONC (ThunderAgent protocol) + ShareGPT chat at CHAT_RATE req/s. Arms:
#   coloc_cache_mtps   stock MTP, our cache-affinity router (colocation, stock kernel/schedule)
#   coloc_sgl_mtps     stock MTP, sglang_router cache_aware (stock colocation)
#   coloc_cache_edf_mtp  ours: 4-mode MTP on the split + worker EDF + uncapped chunk preemption, chunk 2048
#   disagg_2p2d_mtps   stock PD disaggregation (8 prefill + 8 decode GPUs), stock MTP
# Runs locally (drives the pods through px_multi). Claude, 2026-10-16.
set -u
cd "$(dirname "$0")"
export SID=1c90eac0a6848b48 NPODS=8 SESSIONS=${SESSIONS:-192} TURNS=${TURNS:-2} AGENT_CONC=${AGENT_CONC:-64}
export CHAT_SHAREGPT=1 CHAT_RATE=${CHAT_RATE:-16} CHAT_N=${CHAT_N:-4800} MIN_CTX=32768 MAX_CTX=65536
export RUN_TAG=_c${AGENT_CONC}_r${CHAT_RATE}
KERNEL=lpt PREEMPT=0 python3 -u mn_run.py coloc_cache_mtps coloc_sgl_mtps
KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 python3 -u mn_run.py coloc_cache_edf_mtp
KERNEL=lpt PREEMPT=0 python3 -u mn_run.py disagg_2p2d_mtps
echo MATRIX_DONE
