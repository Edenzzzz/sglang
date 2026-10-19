#!/bin/bash
# Stock is bimodal on the standard mix (2026-10-19: no-spec agent TTFT 5.4 s in reps 1-2, 3.0 s in rep 3, same pod,
# tree and seeds; the slow mode queues 140-240 requests behind the first agent prefills). Three more matched repeats
# of the four arms after h200_pp_and_trace_20261019.sh, so every comparison is reported over the distribution.
set -u
cd "$(dirname "$0")"
while pgrep -f h200_pp_and_trace_20261019.sh >/dev/null; do sleep 30; done
for REPN in 4 5 6; do
  TAG=_pt_rep$REPN ARMS="stock ours_ovl stock_mtp ours_mtp" CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 \
    bash mn_h200_nospec_20261018.sh
done
echo H200_MORE_REPS_DONE
