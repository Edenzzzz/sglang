#!/bin/bash
# SLO-sized prefill chunk (apply_edf_itl_slo_chunk.py) on the standard mix, H200 dadf94f0e1da7238 (2026-10-19, user):
# the largest chunk that keeps running decodes within a 20 ms ITL SLO unless the prefill is past its TTFT deadline,
# on ours without MTP (FA2 overlapped) and ours + MTP, plus a third stock repeat for drift. Same load as
# h200_per_tenant_20261019.sh. Claude.
set -u
cd "$(dirname "$0")"
for REPN in ${REPS:-1}; do
  TAG=_pt_slo_rep$REPN ITL_SLO=${ITL_SLO_NOSPEC:-20:1:7:0.02:5e-7:128} ARMS=ours_ovl CHUNK=2048 \
    DEADLINE_CHUNK=1024:2048:6000 bash mn_h200_nospec_20261018.sh
  TAG=_pt_slo_rep$REPN ITL_SLO=${ITL_SLO_MTP:-20:2.0:9:0.02:5e-7:128} ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
done
[ "${STOCK_REP:-1}" = 1 ] && TAG=_pt_rep3 ARMS="stock stock_mtp" bash mn_h200_nospec_20261018.sh
echo H200_ITL_SLO_DONE
