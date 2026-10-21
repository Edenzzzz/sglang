#!/bin/bash
# Is the remaining backend gap our SPLIT, or FA3-in-our-backend? (H200, 2026-10-21.) The step trace says ours' agent TPOT is
# better than stock's (20.3 vs 24.8 ms) but its agent TTFT is worse (8.5 vs 7.8 s), and prefill (EXTEND) steps cost
# ours +8% GPU overall and +43% in the shallow bin (ptok<600, prefix<16K: 10.68 vs 7.48 ms). At fixed agent concurrency
# TTFT sets throughput, so prefill-step time is the gap. Those shallow steps are exactly the ones the policy SPLITS
# (predicted chunk < 1 ms). SGLANG_SPLIT_SINGLE_CHUNK_MS=0 makes every extend step ONE FA3 call (never split), which
# separates our planning + second launch from FA3 itself. Same schedule / load as the trace arms, STEP_TRACE on.
# Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export CHAT_TEMPLATE=1 STEP_TRACE=1
export SID=dadf94f0e1da7238 MN_CP=i18n-bd NPODS=1 GPUS_PER_POD=1 SM_FA3=112 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536
export TURNS=2 AGENT_CONC=8 SESSIONS=24 CHAT_RATE=8 CHAT_N=2400
env SPLIT_ROWS_FA3=1 SPLIT_POLICY=1 SPLIT_NEVER=1 DRAFT_FA3=1 SKIP_BASE_PLAN=1 VERIFY_FA3=1 MIXED_CHUNK=0 \
  KERNEL=mtpsplit PREEMPT=0 RUN_TAG=_h200_pp_c8_ct_ourbe_nosplit python3 -u mn_run.py coloc_sgl_mtps
python3 - <<'PY'
import sys; sys.path.insert(0, "/tmp")
from px import run
A="/dev/shm/w/mn/coloc_sgl_mtps_h200_pp_c8_ct_fa3be_tr/worker_r0_g0.log"
B="/dev/shm/w/mn/coloc_sgl_mtps_h200_pp_c8_ct_ourbe_nosplit/worker_r0_g0.log"
print(run(f"cd /dev/shm/w/perf && /dev/shm/w/venv/bin/python steptrace_compare.py {A} {B} 2>&1 | grep -E 'DECODE|EXTEND|==|mixed ptok' | head -20")[1])
print(run(f"grep -h 'SPLIT_POLICY split' {B} | tail -1 | cut -c22-90")[1])
PY
echo H200_NOSPLIT_PROBE_DONE
