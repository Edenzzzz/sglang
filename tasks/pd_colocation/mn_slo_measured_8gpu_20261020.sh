#!/bin/bash
# Multi-node SLO control with measured decode-step time (2026-10-20): the grid-fit decode model over-predicted H20
# fan-out decode steps ~2-3x, capping prefill to ~10 tokens/step at 68K depth. Here the workers' decode-first rule uses
# their own measured decode-step EWMA (ITL_SLO ...:measured) and the router uses the reported decode_step_ms. Two
# repeats after the A/B script (pid $1), same fan-out config. Claude.
set -u
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
export FO_TEXT=1
for REPN in 1 2; do
  LOAD_PUB=1 ITL_SLO=20:1:6.972:0.2228:0.1137:0.1091:4.652e-6:measured \
    EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5 --slo-chunk 20:0.1091:4.652e-6" \
    REP=_text_sloctlm_r$REPN ARMS=coloc_slo_edf_hc bash mn_fanout_8gpu_20261018.sh
done
echo MN_SLO_MEASURED_DONE
