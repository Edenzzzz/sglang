#!/bin/bash
# The ~10% of the MTP gap that is NOT the draft model (H200, 2026-10-21, after h200_draft_fa3_20261021.sh, pid $1).
# Draft extend on FA3 moved stock's schedule on our backend 142.5 -> 145.3 tok/s, vs 161.5 on stock's own FA3 backend.
# What our backend still pays per extend step: indices_updater_prefill.update builds the full-batch flashinfer CSR
# (kv_indices for every row) and plans the paged wrapper, which under SPLIT_POLICY is never read -- every extend step is
# served by FA3. SGLANG_FUSED_SKIP_BASE_PLAN=1 skips it (guarded on no SWA pool / no fp8 workspace / no MIS / no ragged).
#   1. stock's prefill-prioritized + MTP schedule on our backend, DRAFT_FA3 + SKIP_BASE_PLAN -- does it reach 161?
#   2. ours + MTP (rows + SLO admission) with both, two repeats
#   3. ours without MTP with both, two repeats (base_plan_skipped counter logged in each)
# Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
export CHAT_TEMPLATE=1
FA="SPLIT_ROWS_FA3=1 SPLIT_POLICY=1 DRAFT_FA3=1 SKIP_BASE_PLAN=1"
WX="--max-running-requests 128 --cuda-graph-max-bs-decode 48"
(export SID=dadf94f0e1da7238 MN_CP=i18n-bd NPODS=1 GPUS_PER_POD=1 SM_FA3=112 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 \
   TURNS=2 AGENT_CONC=8 SESSIONS=24 CHAT_RATE=8 CHAT_N=2400
 env $FA VERIFY_FA3=1 MIXED_CHUNK=0 KERNEL=mtpsplit PREEMPT=0 RUN_TAG=_h200_pp_c8_ct_ourbe_dr3sb python3 -u mn_run.py coloc_sgl_mtps)
for REPN in 1 2; do
  env $FA VERIFY_FA3=1 SPLIT_OVERLAP=1 DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" \
    TAG=_ct_rows48adm_dr3sb_rep$REPN ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
  env $FA DECODE_FA3=1 TAG=_ct_pol_dr3sb_rep$REPN ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 \
    bash mn_h200_nospec_20261018.sh
done
echo H200_SKIP_BASE_PLAN_DONE
