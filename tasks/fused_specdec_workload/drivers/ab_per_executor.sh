#!/bin/bash
# Kernel optimisation items, measured PER EXECUTOR against that executor's own baseline.
# No FA3-vs-FA2 comparison anywhere: FA3 items are FA3-vs-FA3, FA2 items are FA2-vs-FA2.
#
# WHY THIS RE-RUNS WORK: every trait A/B before this was measured BEFORE the seqused fix,
# when FA3's cost was Theta(kv^2) and the bug dominated the runtime. "MmaPV_is_RS worth
# 2.3%", "LPT 1.5%", "stages=3 1.6% slower" were all percentages of a broken baseline and
# are void as optimisation results.
cd /dev/shm/w/fi
. /dev/shm/w/venv/bin/activate
export PYTHONPATH=/dev/shm/w/python:/dev/shm/w/fi HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export TORCH_CUDA_ARCH_LIST=9.0a
mkdir -p /dev/shm/w/resx

# FA3-SHAPED fixture: the two TraceLab append rows only. The full 22-row mixed batch CANNOT
# produce an FA3 baseline -- pointing the FA3 executor at it leaves 5632 of 5650 rows
# unwritten, because that executor is planned for the leading n_fa3 rows and does not serve
# qo=1 decode rows. So per-executor comparison uses a fixture BOTH executors can serve in
# full, and the two arms then do identical work.
TL='qo:2048,2048 prefix:65536,98304'
NROWS=2

# $1 tag  $2 n_fa3  $3.. env overrides
run() {
  local tag=$1 nf=$2; shift 2
  export TORCH_EXTENSIONS_DIR=/dev/shm/w/exx_$tag
  echo "===== $tag   (n_fa3=$nf  $*)"
  env PAGE_SIZE=128 TWO_EXEC=1 TWO_EXEC_OVERLAP=0 FA3_UPSTREAM=1 \
    UP_STAGES=2 MMA_PV_RS=1 SCHED_LPT=0 FA3_PAGED_TMA=0 FA2_MMA_KV=2 \
    FA3_SM_BUDGET=132 TIME_ITERS=5 FIXTURE="$TL n_fa3:$nf" UNIFIED=1 SPLITS=auto \
    FLASHINFER_SM_PARTITION=1 FLASHINFER_FA3_SM_SPLIT_CLUSTERS=124 "$@" \
    timeout 2500 python tests/test_fused_server_batch.py > /dev/shm/w/resx/$tag.full 2>&1
  grep -hE '\[time\]|PASS:|FAILURES' /dev/shm/w/resx/$tag.full | head -5
  echo "fallbacks: $(grep -ci fallback /dev/shm/w/resx/$tag.full)"
}

echo "##### FA3 executor: baseline, then one item at a time (FA3 vs FA3)"
run fa3_base   $NROWS
run fa3_stage3 $NROWS UP_STAGES=3
run fa3_lpt1   $NROWS SCHED_LPT=1
run fa3_rs0    $NROWS MMA_PV_RS=0
run fa3_tma    $NROWS FA3_PAGED_TMA=1

echo "##### FA2 executor: baseline, then one item at a time (FA2 vs FA2)"
run fa2_base   0
run fa2_kv3    0 FA2_MMA_KV=3
run fa2_kv4    0 FA2_MMA_KV=4
echo PEREXECDONE
