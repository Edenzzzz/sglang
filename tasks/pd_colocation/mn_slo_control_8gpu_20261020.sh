#!/bin/bash
# Multi-node SLO control (user, 2026-10-19/20) on 8 H20s, fan-out on natural text, full fabric, 180 s rotation:
#   coloc_slo_edf_hc (ref)   ours without SLO control: SLO-checked hit-first router, pool, EDF (the 10-19 best)
#   coloc_slo_edf_hc (ctl)   + decode-first workers (apply_edf_itl_slo_chunk.py, 20 ms ITL SLO, live decode base from the
#                            H20 decode grid fit: c0 6.972 + 0.2228/row + 0.1137/KV-ktok; prefill a 0.1091 ms/tok,
#                            b 4.652e-6 from the cold-prefill curve) + node-reported load (apply_load_stat_4mode.py,
#                            LOAD_PUB=1) + router --slo-chunk (TTFT under each node's decode load)
#   disagg_2p2d              PD disaggregation, no spec (up_clean, default chunk)
# Two matched repeats. Claude, 2026-10-20.
set -u
cd "$(dirname "$0")"
export FO_TEXT=1
BASE_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5"
for REPN in ${REPS:-1 2}; do
  EXTRA_RA="$BASE_RA" REP=_text_sloref_r$REPN ARMS=coloc_slo_edf_hc bash mn_fanout_8gpu_20261018.sh
  LOAD_PUB=1 ITL_SLO=20:1:6.972:0.2228:0.1137:0.1091:4.652e-6 \
    EXTRA_RA="$BASE_RA --slo-chunk 20:0.1091:4.652e-6" REP=_text_sloctl_r$REPN ARMS=coloc_slo_edf_hc \
    bash mn_fanout_8gpu_20261018.sh
  REP=_text_sloref_r$REPN ARMS=disagg_2p2d bash mn_fanout_8gpu_20261018.sh
done
echo MN_SLO_CONTROL_DONE
