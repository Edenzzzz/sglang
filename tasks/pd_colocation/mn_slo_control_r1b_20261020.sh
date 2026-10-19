#!/bin/bash
# Rerun of rep 1's SLO-control arm after the sub-page-chunk fix (apply_edf_itl_slo_chunk.py: < 1 page -> no prefill
# this step; the first run crashed every worker on a zero-token extend). Waits for the A/B script's pid ($1).
# Claude, 2026-10-20.
set -u
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
export FO_TEXT=1
LOAD_PUB=1 ITL_SLO=20:1:6.972:0.2228:0.1137:0.1091:4.652e-6 \
  EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5 --slo-chunk 20:0.1091:4.652e-6" \
  REP=_text_sloctl_r1b ARMS=coloc_slo_edf_hc bash mn_fanout_8gpu_20261018.sh
echo MN_SLO_R1B_DONE
