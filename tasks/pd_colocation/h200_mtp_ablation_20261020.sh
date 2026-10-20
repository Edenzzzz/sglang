#!/bin/bash
# Per-component MTP ablation and CUDA graphs off (H200 dadf94f0e1da7238, standard mix, 2026-10-20): ours + MTP minus
# the split kernel / EDF / the deadline chunk (mn_h200_nospec_20261018.sh ours_mtp_unfused / _noedf / _nodc), then ours
# + MTP and stock + MTP with --disable-cuda-graph. Two repeats; compared with the in-session ours_mtp / stock_mtp
# repeats of h200_decode_rows_20261020.sh and the six-repeat table. Claude.
set -u
cd "$(dirname "$0")"
for REPN in 1 2; do
  TAG=_pt_abl_rep$REPN ARMS="ours_mtp_unfused ours_mtp_noedf ours_mtp_nodc" bash mn_h200_nospec_20261018.sh
  WORKER_EXTRA="--disable-cuda-graph" TAG=_pt_nograph_rep$REPN ARMS="ours_mtp stock_mtp" bash mn_h200_nospec_20261018.sh
done
echo H200_MTP_ABLATION_DONE
