#!/bin/bash
# 8-GPU chat request-rate sweep for the throughput vs tok/s/user figure (user, 2026-10-16: "use real
# request rates from ref list papers"). Per-GPU chat rates 1 / 2 / 4 / 8 req/s (8 / 16 / 32 / 64 req/s on
# 8 H20s), anchored on the reference list's per-GPU rates: SLOs-Serve 1.5-4.5 req/s per A100
# (OPT-7B/13B), semi-PD 11-16 req/s per A100 (Llama3-8B ShareGPT), DistServe / Sarathi-Serve
# 0.5-10 req/s. (The earlier matrix's chat 4 / 8 req/s = 0.5 / 1 per GPU sat at or below the bottom of
# that range.) Agent tenant fixed at program concurrency C=16 (ThunderAgent). Arms: stock colocation,
# ours, stock PD disagg (clean upstream). CHAT_N = 120 s of offered chat, at least 2400 requests.
# Waits for mn_rerun_c32_cache_mtps_20261016.sh (same pods). Claude, 2026-10-16.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
until grep -q CACHE_MTPS_RERUN_DONE /tmp/h20/mn_rerun_c32_cache_mtps.log; do sleep 30; done
export SID=bb854a10b8954223 NPODS=4 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export AGENT_CONC=16 SESSIONS=48
for R in 8 16 32 64; do
  N=$((R * 120)); [ $N -lt 2400 ] && N=2400
  export CHAT_RATE=$R CHAT_N=$N RUN_TAG=_8g_chatsweep_c16_r${R}
  KERNEL=lpt PREEMPT=0 python3 -u mn_run.py coloc_sgl_mtps
  KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 python3 -u mn_run.py coloc_cache_edf_mtp
  SGL_PY=/dev/shm/w/up_clean/python KERNEL=lpt PREEMPT=0 python3 -u mn_run.py disagg_2p2d_mtps
done
echo CHAT_SWEEP_DONE
