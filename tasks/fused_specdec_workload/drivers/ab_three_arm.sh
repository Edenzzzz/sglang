#!/bin/bash
# THE comparison the goal actually asks for: pure advantage over BOTH baselines.
#
# Every number reported before this was split vs FA2-on-everything. There was no
# FA3-on-everything arm at all, so half the claim ("pure advantage over baseline fa3")
# had never been measured. This runs all three on one fixture.
#
# n_fa3 selects how many leading rows go to the FA3 executor:
#   0        -> FA2 on everything          (baseline FA2)
#   <all>    -> FA3 on everything          (baseline FA3)
#   2        -> the split                  (appends to FA3, rest to FA2)
#
# FA3-all also sidesteps the split-row corruption: that needs a task0 row AFTER a split
# row, and with no FA2 rows there are none. So this arm is valid today.
cd /dev/shm/w/fi
. /dev/shm/w/venv/bin/activate
export PYTHONPATH=/dev/shm/w/python:/dev/shm/w/fi HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export TORCH_CUDA_ARCH_LIST=9.0a
mkdir -p /dev/shm/w/res3arm

TL='qo:2048,2048,512,1024,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 prefix:65536,98304,0,0,81920,73728,90112,65536,65536,98304,65536,81920,98304,65536,98304,73728,65536,65536,90112,90112,65536,73728'
NROWS=22

arm() {   # $1 tag   $2 n_fa3
  export TORCH_EXTENSIONS_DIR=/dev/shm/w/ex3_$1
  echo "===== $1  (n_fa3=$2)"
  env PAGE_SIZE=128 FA3_PAGED_TMA=0 TWO_EXEC=1 TWO_EXEC_OVERLAP=0 FA3_UPSTREAM=1 \
    UP_STAGES=2 MMA_PV_RS=1 SCHED_LPT=0 FA3_SM_BUDGET=132 TIME_ITERS=5 \
    FIXTURE="$TL n_fa3:$2" UNIFIED=1 SPLITS=auto \
    FLASHINFER_SM_PARTITION=1 FLASHINFER_FA3_SM_SPLIT_CLUSTERS=124 \
    timeout 2500 python tests/test_fused_server_batch.py > /dev/shm/w/res3arm/$1.full 2>&1
  grep -hE '\[time\]|PASS:|FAILURES' /dev/shm/w/res3arm/$1.full | head -6
  # VALIDITY: fallback count beside the timings, always.
  echo "fallbacks: $(grep -ci fallback /dev/shm/w/res3arm/$1.full)"
}

arm fa2_all 0
arm fa3_all $NROWS
arm split   2
echo THREEARMDONE
