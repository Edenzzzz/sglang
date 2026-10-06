#!/bin/bash
# FA3 trait A/B on the TraceLab batch. Writes each arm's result to its own file the moment
# it finishes, because three trials have now expired mid-run and taken /dev/shm with them.
cd /dev/shm/w/fi
. /dev/shm/w/venv/bin/activate
export PYTHONPATH=/dev/shm/w/python:/dev/shm/w/fi HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export TORCH_CUDA_ARCH_LIST=9.0a
mkdir -p /dev/shm/w/res

TL='qo:2048,2048,512,1024,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 prefix:65536,98304,0,0,81920,73728,90112,65536,65536,98304,65536,81920,98304,65536,98304,73728,65536,65536,90112,90112,65536,73728'

# $1 tag  $2 rs  $3 stages  $4 lpt  $5 n_fa3  $6 fa2_mma_kv (default 2)
arm() {
  export TORCH_EXTENSIONS_DIR=/dev/shm/w/ex_$1
  rm -rf $TORCH_EXTENSIONS_DIR
  {
    echo "=== $1  (RS=$2 stages=$3 LPT=$4 n_fa3=$5 fa2_mma_kv=${6:-2})"
    env PAGE_SIZE=128 FA3_PAGED_TMA=0 TWO_EXEC=1 TWO_EXEC_OVERLAP=0 FA3_UPSTREAM=1 \
      FA2_MMA_KV=${6:-2} \
      UP_STAGES=$3 MMA_PV_RS=$2 SCHED_LPT=$4 FA3_SM_BUDGET=132 TIME_ITERS=5 \
      FIXTURE="$TL n_fa3:$5" UNIFIED=1 SPLITS=auto \
      FLASHINFER_SM_PARTITION=1 FLASHINFER_FA3_SM_SPLIT_CLUSTERS=124 \
      timeout 2500 python tests/test_fused_server_batch.py > /dev/shm/w/res/$1.full 2>&1
    # Summary derived from the kept full log. On a fault, print the tail of the full log
    # instead of grep lines -- a filter that matches "Traceback" and drops the frames
    # after it hides the only useful part.
    if grep -qiE 'Traceback|CUDA Error|illegal|error:|FAIL' /dev/shm/w/res/$1.full; then
      echo "[FAULT] tail of full log:"; tail -25 /dev/shm/w/res/$1.full
    else
      grep -iE '\[nsplit\]|rel err|PASS|\[time\]|fallback' /dev/shm/w/res/$1.full | head -14
    fi
    echo "--- end $1"
  } > /dev/shm/w/res/$1.txt 2>&1
  cat /dev/shm/w/res/$1.txt
}

arm ctl    1 2 0 0   # control: FA2 on everything
arm rs0    0 2 1 2   # the OLD configuration -- must reproduce ~480 ms
arm rs1    1 2 0 2   # the fix
arm rs1s3  1 3 0 2   # fix + the pipeline depth it unlocks
arm rs1l1  1 2 1 2   # fix, old LPT ordering -- isolates LPT

# Priority item 1: K/V pipelining on the FA2 side. Same long-context batch, FA2 on
# everything, so this is an FA2-vs-FA2 comparison and independent of the FA3 traits.
arm fa2k2  1 2 0 0 2
arm fa2k3  1 2 0 0 3
arm fa2k4  1 2 0 0 4

# Tests the prediction recorded in the "1604x off roofline" entry: does the same trait
# explain THAT fixture's 17 358 us? Separate fixture, so it gets its own arm pair.
SHORT='qo:8,8,100,1 prefix:16384,16384,0,0'
armf() {
  export TORCH_EXTENSIONS_DIR=/dev/shm/w/ex_$1
  rm -rf $TORCH_EXTENSIONS_DIR
  {
    echo "=== $1  (RS=$2 stages=$3 LPT=$4, fixture 16384-KV)"
    env PAGE_SIZE=128 FA3_PAGED_TMA=0 TWO_EXEC=1 TWO_EXEC_OVERLAP=0 FA3_UPSTREAM=1 \
      FA2_MMA_KV=2 UP_STAGES=$3 MMA_PV_RS=$2 SCHED_LPT=$4 FA3_SM_BUDGET=132 TIME_ITERS=5 \
      FIXTURE="$SHORT n_fa3:2" UNIFIED=1 SPLITS=auto \
      FLASHINFER_SM_PARTITION=1 FLASHINFER_FA3_SM_SPLIT_CLUSTERS=124 \
      timeout 2500 python tests/test_fused_server_batch.py > /dev/shm/w/res/$1.full 2>&1
    # Summary derived from the kept full log. On a fault, print the tail of the full log
    # instead of grep lines -- a filter that matches "Traceback" and drops the frames
    # after it hides the only useful part.
    if grep -qiE 'Traceback|CUDA Error|illegal|error:|FAIL' /dev/shm/w/res/$1.full; then
      echo "[FAULT] tail of full log:"; tail -25 /dev/shm/w/res/$1.full
    else
      grep -iE '\[nsplit\]|rel err|PASS|\[time\]|fallback' /dev/shm/w/res/$1.full | head -14
    fi
    echo "--- end $1"
  } > /dev/shm/w/res/$1.txt 2>&1
  cat /dev/shm/w/res/$1.txt
}
armf sh_rs0 0 2 1
armf sh_rs1 1 2 0
echo AB5DONE
