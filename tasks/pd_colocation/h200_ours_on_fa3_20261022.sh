#!/bin/bash
# Ours' SCHEDULE on sglang's FA3 backend (H200, 2026-10-22, after $1). The backend-class control showed the cost is the
# flashinfer backend class, not our split: same stock schedule, slow bin 7.48 ms on the FA3 backend vs 11.98 ms on the
# flashinfer backend with no split at all (81.1 vs 161.2 agent tok/s) -- upstream deprecated that backend on Hopper for
# exactly this regression. Our FA3 patches recover 81 -> 147, but the class's residual overhead is the remaining ~9%.
# Our scheduling lives in scheduler.py, not the attention backend, so it should run on the FA3 backend as-is:
#   KERNEL=lpt (no split planning at all) + ATTN_BACKEND=fa3 + our EDF / deadline chunk / rows cap / 4-mode MTP.
# Arms (one repeat each, chat 8 req/s, chat template, STEP_TRACE on the first):
#   ours no MTP on FA3   : EDF + chunk preemption + deadline chunk
#   ours + MTP on FA3    : the 4-mode step + rows cap + SLO admission
# If the 4-mode step cannot run on this backend it will say so in the worker log (fallbacks / traceback), which is
# itself the answer about whether the split's backend is load-bearing for the schedule.
# Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
export CHAT_TEMPLATE=1
WX="--max-running-requests 128 --cuda-graph-max-bs-decode 48"
STEP_TRACE=1 ATTN_BACKEND=fa3 KERNEL=lpt TAG=_ct_fa3be_ours ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 \
  bash mn_h200_nospec_20261018.sh
ATTN_BACKEND=fa3 KERNEL=lpt DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" TAG=_ct_fa3be_rows48adm ARMS=ours_mtp \
  bash mn_h200_nospec_20261018.sh
python3 - <<'PY'
import sys; sys.path.insert(0, "/tmp")
from px import run
L="/dev/shm/w/mn"
A=f"{L}/coloc_sgl_mtps_h200_pp_c8_ct_fa3be_tr/worker_r0_g0.log"
B=f"{L}/coloc_cache_edf_h200_split_ovl_ch2048_dl1024-2048-6000_c8_ct_fa3be_ours/worker_r0_g0.log"
print(run(f"cd /dev/shm/w/perf && /dev/shm/w/venv/bin/python binfeat.py {A} {B}")[1])
for d in ("coloc_cache_edf_h200_split_ovl_ch2048_dl1024-2048-6000_c8_ct_fa3be_ours",
          "coloc_cache_edf_mtp_h200_4mode_unc_ch2048_dl1024-2048-6000_c8_ct_fa3be_rows48adm"):
    print(f"## {d}")
    print(run(f"grep -c 'Traceback' {L}/{d}/worker_r0_g0.log; grep -o 'MIXEDSPEC_STEPS [0-9]*' {L}/{d}/worker_r0_g0.log | tail -1; "
              f"grep -o 'accept len: [0-9.]*' {L}/{d}/worker_r0_g0.log | tail -1; "
              f"/dev/shm/w/venv/bin/python /dev/shm/w/perf/bench_per_tenant.py {L}/{d} | tail -2 | cut -c1-150")[1])
PY
echo H200_OURS_ON_FA3_DONE
