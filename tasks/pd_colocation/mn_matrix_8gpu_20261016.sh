#!/bin/bash
# H20 multi-node matrix on 8 GPUs (user, 2026-10-16: "launch 8 h20 gpus"): job bb854a10b8954223, 4 pods x
# 2 H20 = 8 TP1 workers, EAGLE3 MTP everywhere. Workload: TraceLab 32-64K agent sessions at program
# concurrency AGENT_CONC (ThunderAgent) + ShareGPT chat at CHAT_RATE req/s; load halved from the 16-GPU
# matrix (C = 16 / 32, chat 4 / 8 req/s). Arms:
#   coloc_sgl_mtps       stock MTP, sglang_router cache_aware (stock colocation)
#   coloc_cache_mtps     stock MTP, our cache-affinity router
#   coloc_cache_edf_mtp  ours: our router + 4-mode MTP on the split + worker EDF + chunk preemption, chunk
#                        2048, load+context gate (mix iff >= 8K context or >= 4 waiting)
#   disagg_2p2d_mtps     stock PD disaggregation (2 prefill pods + 2 decode pods), stock MTP
# Runs locally (drives the pods through px_multi). Claude, 2026-10-16.
set -u
cd "$(dirname "$0")"
export SID=bb854a10b8954223 NPODS=4 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 TURNS=${TURNS:-2}
for point in "16 4" "32 8"; do
  read -r C R <<< "$point"
  export AGENT_CONC=$C CHAT_RATE=$R SESSIONS=$((C * 3)) CHAT_N=$((R * 300)) RUN_TAG=_8g_c${C}_r${R}
  KERNEL=lpt PREEMPT=0 python3 -u mn_run.py coloc_sgl_mtps coloc_cache_mtps
  KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 python3 -u mn_run.py coloc_cache_edf_mtp
  KERNEL=lpt PREEMPT=0 python3 -u mn_run.py disagg_2p2d_mtps
done
echo MATRIX8_DONE
