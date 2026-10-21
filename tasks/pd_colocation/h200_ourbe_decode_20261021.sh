#!/bin/bash
# The last untested per-step cost in "stock's schedule on our backend" (H200, 2026-10-21): every such arm so far ran
# DECODE_FA3=0, so its decode / verify steps still used flashinfer's decode wrapper. The step trace
# (h200_backend_trace_20261021.sh) shows decode steps are 211 of ours' 297 GPU-seconds, at 7.26 ms mean vs stock's 6.09,
# and that total GPU time and host gap are equal between backends -- so what is left is inside those steps.
# Same schedule and load as the trace arms, with DECODE_FA3=1, under STEP_TRACE so the decode step time is comparable.
# Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export CHAT_TEMPLATE=1 STEP_TRACE=1
export SID=dadf94f0e1da7238 MN_CP=i18n-bd NPODS=1 GPUS_PER_POD=1 SM_FA3=112 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536
export TURNS=2 AGENT_CONC=8 SESSIONS=24 CHAT_RATE=8 CHAT_N=2400
env SPLIT_ROWS_FA3=1 SPLIT_POLICY=1 DRAFT_FA3=1 SKIP_BASE_PLAN=1 VERIFY_FA3=1 DECODE_FA3=1 MIXED_CHUNK=0 \
  KERNEL=mtpsplit PREEMPT=0 RUN_TAG=_h200_pp_c8_ct_ourbe_dec python3 -u mn_run.py coloc_sgl_mtps
python3 - <<'PY'
import sys; sys.path.insert(0, "/tmp")
from px import run
A="/dev/shm/w/mn/coloc_sgl_mtps_h200_pp_c8_ct_fa3be_tr/worker_r0_g0.log"
B="/dev/shm/w/mn/coloc_sgl_mtps_h200_pp_c8_ct_ourbe_dec/worker_r0_g0.log"
print(run(f"cd /dev/shm/w/perf && /dev/shm/w/venv/bin/python steptrace_compare.py {A} {B} 2>&1 | grep -E 'DECODE|EXTEND|==' | head -12")[1])
print(run(f"grep -h 'DECODE_FA3_CHECK\\|VERIFY_FA3 calls' {B} | tail -1 | cut -c22-120")[1])
PY
echo H200_OURBE_DECODE_DONE
