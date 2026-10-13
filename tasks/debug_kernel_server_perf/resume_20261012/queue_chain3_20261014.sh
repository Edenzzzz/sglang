#!/bin/bash
# Ordered GPU queue (replaces the paused queue_chain2): another session's scripts hold the GPU and
# use `flock -n`, so start only after 10 consecutive idle minutes (no serving/kernel process and the
# GPU lock free), then run everything left, 4-mode arms on the split (MTPMIX_SPLIT).
cd /dev/shm/w
idle=0
while [ $idle -lt 10 ]; do
  if pgrep -f '[r]un_mixed_workloads|[s]glang.launch_server|[f]used_reference|[s]rv_correctness|[n]cu ' >/dev/null \
     || ! flock -n /dev/shm/w/a2_kernel.lock true; then idle=0; else idle=$((idle+1)); fi
  sleep 60
done
echo "CHAIN3_START $(date +%T) after 10 idle minutes"
bash tracelab_full_v2b_20261013.sh > tracelab_full_v3resume_20261014.log 2>&1
bash b3_append_ceiling_20261013.sh > b3_append_ceiling_20261013.log 2>&1
bash unloaded_ref_20261013.sh > unloaded_ref_20261013.log 2>&1
bash a3b_serving_ab_20261013.sh > a3b_serving_ab_20261013.log 2>&1
bash s3_standalone_20261013.sh > s3_standalone_20261013.log 2>&1
bash s4_edf_20261013.sh > s4_edf_20261013.log 2>&1
bash baselines_pdmux_20261013.sh > baselines_pdmux_20261013.log 2>&1
echo QUEUE_CHAIN3_DONE
