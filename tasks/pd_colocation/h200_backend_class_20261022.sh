#!/bin/bash
# Is the cost our SPLIT CODE or the flashinfer BACKEND CLASS? (H200, 2026-10-22, after $1.)
# The shallow mixed bin (ptok 164, prefix 37, nd 0) costs 10.67 ms on our backend vs 7.48 on stock's FA3 backend, and
# none of these changed it: forcing one FA3 call per extend step (10.67), the page table's row stride (-4..+4% in a
# kernel test), or the per-layer dispatch fastpath (throughput 146.9 either way). So the control that was missing:
# the SAME stock schedule on the flashinfer backend with NO split at all (KERNEL=lpt, SGLANG_FUSED_FA3_SM_SPLIT unset),
# traced. If the shallow bin is still ~10.6 ms, the cost is the flashinfer backend class itself and porting the split
# onto sglang's FA3 backend class is the fix; if it drops to ~7.5 ms, the cost is in our split code after all.
# Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
export CHAT_TEMPLATE=1 STEP_TRACE=1
export SID=dadf94f0e1da7238 MN_CP=i18n-bd NPODS=1 GPUS_PER_POD=1 SM_FA3=112 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536
export TURNS=2 AGENT_CONC=8 SESSIONS=24 CHAT_RATE=8 CHAT_N=2400
env ATTN_BACKEND=flashinfer MIXED_CHUNK=0 KERNEL=lpt PREEMPT=0 RUN_TAG=_h200_pp_c8_ct_fibe_nosplit \
  python3 -u mn_run.py coloc_sgl_mtps
python3 - <<'PY'
import sys; sys.path.insert(0, "/tmp")
from px import run
L="/dev/shm/w/mn"
logs = " ".join(f"{L}/{d}/worker_r0_g0.log" for d in (
    "coloc_sgl_mtps_h200_pp_c8_ct_fa3be_tr",        # stock schedule, FA3 backend
    "coloc_sgl_mtps_h200_pp_c8_ct_fibe_nosplit",    # stock schedule, flashinfer backend, NO split
    "coloc_sgl_mtps_h200_pp_c8_ct_ourbe_nosplit",   # ours, split planned but never taken
    "coloc_sgl_mtps_h200_pp_c8_ct_ourbe_fp",        # ours, split policy + per-layer fastpath
))
print(run(f"cd /dev/shm/w/perf && /dev/shm/w/venv/bin/python binfeat.py {logs}")[1])
PY
echo H200_BACKEND_CLASS_DONE
