#!/bin/bash
# Single-node per-row slack decode-first (2026-10-20): chat attainment vs prefill-prioritized stock is +16-20% and the
# misses are TPOT (chat TPOT p90 ~61 ms from deep-context mixed steps). Per-row budget against each request's mean-TPOT
# SLO (chat 20 ms, agents >= 24K prompt 80 ms), measured decode base; H200 decode steps (~7-9 ms) leave room that H20's
# ~20 ms did not. Ours without and with MTP, two repeats; standard mix. Claude.
set -u
cd "$(dirname "$0")"
for REPN in 1 2; do
  TAG=_pt_slack_rep$REPN ITL_SLO=20:1:7:0.2:0.03:0.02:5e-7:measured:slack:cls24000_80 ARMS=ours_ovl CHUNK=2048 \
    DEADLINE_CHUNK=1024:2048:6000 bash mn_h200_nospec_20261018.sh
  TAG=_pt_slack_rep$REPN ITL_SLO=20:2.0:9:0.2:0.03:0.02:5e-7:measured:slack:cls24000_80 ARMS=ours_mtp \
    bash mn_h200_nospec_20261018.sh
done
echo H200_SLACK_DONE
