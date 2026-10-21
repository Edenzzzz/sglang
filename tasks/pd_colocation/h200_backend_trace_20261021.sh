#!/bin/bash
# Where our backend's remaining ~9% MTP gap goes (H200, 2026-10-21, after h200_skip_base_plan_20261021.sh, pid $1).
# Stock's prefill-prioritized + MTP schedule reads 161.5 tok/s on stock's own FA3 backend and 147.8 on ours with every
# target attention call on FA3, the draft extend on FA3 and the unused base plan skipped (+2% draft, +1.7% skip). The
# same schedule and workload under STEP_TRACE=1 on both backends, so steptrace_compare.py / steptrace_gaps.py can say
# whether the rest is GPU time per step type or host launch gap (plan work the FA3 backend does not do: per-step
# split_classes, the split plan's numpy index builds and synchronous H2D copies, the decode wrapper plan, and the
# draft model's 3 per-cycle decode steps, which DRAFT_FA3 does not cover). One repeat each. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
export CHAT_TEMPLATE=1 STEP_TRACE=1
export SID=dadf94f0e1da7238 MN_CP=i18n-bd NPODS=1 GPUS_PER_POD=1 SM_FA3=112 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536
export TURNS=2 AGENT_CONC=8 SESSIONS=24 CHAT_RATE=8 CHAT_N=2400
# stock's schedule on stock's FA3 backend
env ATTN_BACKEND=fa3 MIXED_CHUNK=0 KERNEL=lpt PREEMPT=0 RUN_TAG=_h200_pp_c8_ct_fa3be_tr python3 -u mn_run.py coloc_sgl_mtps
# the same schedule on ours
env SPLIT_ROWS_FA3=1 SPLIT_POLICY=1 DRAFT_FA3=1 SKIP_BASE_PLAN=1 VERIFY_FA3=1 MIXED_CHUNK=0 KERNEL=mtpsplit PREEMPT=0 \
  RUN_TAG=_h200_pp_c8_ct_ourbe_tr python3 -u mn_run.py coloc_sgl_mtps
python3 - <<'PY'
import sys; sys.path.insert(0, "/tmp")
from px import run
A="/dev/shm/w/mn/coloc_sgl_mtps_h200_pp_c8_ct_fa3be_tr/worker_r0_g0.log"
B="/dev/shm/w/mn/coloc_sgl_mtps_h200_pp_c8_ct_ourbe_tr/worker_r0_g0.log"
print(run(f"cd /dev/shm/w/perf && /dev/shm/w/venv/bin/python steptrace_compare.py {A} {B} 2>&1 | tail -40")[1])
print(run(f"cd /dev/shm/w/perf && /dev/shm/w/venv/bin/python steptrace_gaps.py {A} {B} 2>&1 | tail -30")[1])
PY
echo H200_BACKEND_TRACE_DONE
