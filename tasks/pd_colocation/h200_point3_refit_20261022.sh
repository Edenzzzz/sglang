#!/bin/bash
# Point 3, fair re-fit (H200, 2026-10-22, after $1 so it cannot perturb a live measurement). The 2026-10-20 fits that
# answered "is the 4-mode step easier to model?" used stock on the FLASHINFER backend; upstream's Hopper default is
# FA3, so the baseline was wrong. Same fitter, same protocol (first half fit, second half scored, MAPE and p90 APE per
# step kind), on the STEPTRACE logs collected with both systems on FA3:
#   stock's prefill-prioritized + MTP schedule on stock's FA3 backend   (fa3be_tr)
#   the same schedule on our backend                                     (ourbe_fp)
#   ours + MTP, 4-mode, rows + SLO admission                             (rows48adm_fp)
#   ours without MTP                                                     (pol_fp)
# Point 3 claims ours is EASIER to model; the honest test is whether ours' MAPE is lower than stock's on the same
# workload with the same features. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
python3 - <<'PY'
import sys; sys.path.insert(0, "/tmp")
from px import run
L = "/dev/shm/w/mn"
for tag, d in (("stock+MTP on FA3 backend", "coloc_sgl_mtps_h200_pp_c8_ct_fa3be_tr"),
               ("stock+MTP on our backend", "coloc_sgl_mtps_h200_pp_c8_ct_ourbe_fp"),
               ("ours+MTP 4-mode", "coloc_cache_edf_mtp_h200_4mode_unc_ch2048_dl1024-2048-6000_c8_ct_rows48adm_fp"),
               ("ours no MTP", "coloc_cache_edf_h200_split_ovl_ch2048_dl1024-2048-6000_c8_ct_pol_fp")):
    print(f"##### {tag}  ({d})", flush=True)
    print(run(f"cd /dev/shm/w/perf && /dev/shm/w/venv/bin/python fit_4mode_step.py --target gpu {L}/{d}/worker_r0_g0.log 2>&1 | tail -14", timeout=900)[1], flush=True)
PY
echo H200_POINT3_REFIT_DONE
