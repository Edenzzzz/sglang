#!/bin/bash
# Pause the queue for the A4 overlap A/B: let the full-length replay finish its stock arm at
# 0.005/s, then stop the replay before any 4-mode arm (those must use the kernel this A/B picks),
# SIGSTOP the queue chain so it launches nothing, and run the overlap A/B. The chain stays stopped
# until the replay's 4-mode arms and S3 are re-pointed at the winner.
W=/dev/shm/w
until grep -q "START full r0.005 MTPMIX_LPT" $W/tracelab_full_v2b_20261013.log 2>/dev/null; do
  pgrep -f '[t]racelab_full_v2b_20261013.sh' >/dev/null || break; sleep 5; done
chain=$(pgrep -f 'bash queue_chain2_20261013.sh' | head -1); [ -n "$chain" ] && kill -STOP $chain
pkill -f 'tracelab_full_v2b_20261013.sh'; pkill -f 'run_mixed_workloads.py --scenario tracelab_full_env'
sleep 3; pkill -9 -f 'sglang.launch_server'; pkill -9 -f 'sglang::'; pkill -9 -f 'sglang.benchmark.serving'; sleep 5
rm -rf $W/tracelab_full_v3_20261013/full_r0.005/*MTPMIX_LPT* $W/tracelab_full_v3_20261013/full_r0.005/barrier
echo "V3_PAUSED $(date +%T) chain=$chain stopped; after: $(grep 'END full r0.005 MTP_LPT' $W/tracelab_full_v2b_20261013.log)" | tee -a $W/tracelab_full_v2b_20261013.log
bash $W/split_overlap_20261013.sh > $W/split_overlap_20261013.log 2>&1
echo "OVERLAP_AB_FINISHED $(date +%T) -- chain $chain still STOPPED" >> $W/split_overlap_20261013.log
