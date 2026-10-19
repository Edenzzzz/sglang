#!/bin/bash
# 5e mixture shift (2026-10-20): 8 H20s, text fan-out C=32 + ShareGPT chat 8 req/s; ~300 s in a second fan-out driver
# adds 32 agent programs on the same task prefixes for 240 s (agent load x2, chat unchanged; FO_SPIKE in mn_bench.sh).
# Ours vs PD disagg 2P2D, two repeats. Scored by tasks/perf_model/spike_windows.py. Claude.
set -u
cd "$(dirname "$0")"
export FO_TEXT=1 FO_SPIKE=300:32:240
for REPN in 1 2; do
  EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_text_shift_r$REPN ARMS=coloc_slo_edf_hc \
    bash mn_fanout_8gpu_20261018.sh
  REP=_text_shift_r$REPN ARMS=disagg_2p2d bash mn_fanout_8gpu_20261018.sh
done
echo MN_SHIFT_DONE
