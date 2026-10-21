#!/bin/bash
# Point 3 needs the OURS arms traced (H200, 2026-10-22, after $1). The first re-fit scored only the two
# backend-comparison arms because STEP_TRACE was on for those alone; the ours arms logged 0 STEPTRACE lines, so the
# claim that matters -- our 4-mode schedule is easier to model than stock's unfused one -- was not tested. Collect with
# STEP_TRACE=1 and fit: ours + MTP (4-mode, rows + SLO admission) and ours no MTP, each on our backend and on the FA3
# backend, against the stock+MTP FA3 baseline already collected. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
export CHAT_TEMPLATE=1 STEP_TRACE=1
FA="SPLIT_ROWS_FA3=1 SPLIT_POLICY=1 DRAFT_FA3=1 SKIP_BASE_PLAN=1 SPLIT_FASTPATH=1"
WX="--max-running-requests 128 --cuda-graph-max-bs-decode 48"
env $FA VERIFY_FA3=1 SPLIT_OVERLAP=1 DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" TAG=_ct_rows48adm_fptr \
  ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
ATTN_BACKEND=fa3 KERNEL=lpt DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" TAG=_ct_fa3be_rows48adm_tr ARMS=ours_mtp \
  bash mn_h200_nospec_20261018.sh
python3 - <<'PY'
import sys; sys.path.insert(0, "/tmp")
from px import run
L = "/dev/shm/w/mn"
for tag, d in (("stock+MTP, FA3 backend (baseline)", "coloc_sgl_mtps_h200_pp_c8_ct_fa3be_tr"),
               ("ours+MTP 4-mode, our backend", "coloc_cache_edf_mtp_h200_4mode_unc_ch2048_dl1024-2048-6000_c8_ct_rows48adm_fptr"),
               ("ours+MTP 4-mode, FA3 backend", "coloc_cache_edf_mtp_h200_4mode_unc_ch2048_dl1024-2048-6000_c8_ct_fa3be_rows48adm_tr"),
               ("ours no MTP, FA3 backend", "coloc_cache_edf_h200_split_ovl_ch2048_dl1024-2048-6000_c8_ct_fa3be_ours")):
    print(f"##### {tag}", flush=True)
    print(run(f"cd /dev/shm/w/perf && /dev/shm/w/venv/bin/python fit_4mode_step.py --target gpu {L}/{d}/worker_r0_g0.log 2>&1 | tail -9", timeout=900)[1], flush=True)
PY
echo H200_POINT3_TRACES_DONE
