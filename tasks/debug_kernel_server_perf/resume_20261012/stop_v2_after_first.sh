#!/bin/bash
# Pause the S2b v2 sweep once its stock arm at 0.01/s is done, so the race-fix validation runs
# before any further 4-mode arm (those then run on whichever tree the verdict selects).
W=/dev/shm/w
until grep -q "START full r0.01 MTPMIX_LPT" $W/tracelab_full_v2_20261013.log 2>/dev/null; do
  pgrep -f '[t]racelab_full_v2_20261013.sh' >/dev/null || break; sleep 5; done
pkill -f 'tracelab_full_v2_20261013.sh'; pkill -f 'run_mixed_workloads.py --scenario tracelab_full_env'
sleep 3; pkill -9 -f 'sglang.launch_server'; pkill -9 -f 'sglang::'; pkill -9 -f 'sglang.benchmark.serving'; sleep 3
rm -rf $W/tracelab_full_v2_20261013/full_r0.01/*MTPMIX_LPT* $W/tracelab_full_v2_20261013/full_r0.01/barrier
echo "V2_PAUSED $(date +%T) after: $(grep 'END full r0.01 MTP_LPT' $W/tracelab_full_v2_20261013.log)"
