#!/bin/bash
# Ordered GPU queue (replaces queue_chain_20261013): pause S2b v2 after its stock arm, validate the
# race fix, resume S2b on the verdict's tree, then the rest.
cd /dev/shm/w
bash stop_v2_after_first.sh >> tracelab_full_v2_20261013.log 2>&1
bash graphs_racefix_20261013.sh > graphs_racefix_20261013.log 2>&1
bash racefix_verdict.sh >> graphs_racefix_20261013.log 2>&1
bash tracelab_full_v2b_20261013.sh > tracelab_full_v2b_20261013.log 2>&1
bash b3_append_ceiling_20261013.sh > b3_append_ceiling_20261013.log 2>&1
bash unloaded_ref_20261013.sh > unloaded_ref_20261013.log 2>&1
bash a3b_serving_ab_20261013.sh > a3b_serving_ab_20261013.log 2>&1
bash s3_standalone_20261013.sh > s3_standalone_20261013.log 2>&1
bash s4_edf_20261013.sh > s4_edf_20261013.log 2>&1
echo QUEUE_CHAIN_DONE
