#!/bin/bash
# Kernel A/B with PARALLEL builds: compile every arm's extension concurrently on the CPUs,
# then run the arms back to back on the GPU. Builds (~5-10 min each) dominated serial sweeps.
#
# Usage: ab_parallel.sh "<fixture>" "tag1:ENV=V ENV=V" "tag2:ENV=V" ...
# Each arm's ENV overrides the shared defaults below and gets its own TORCH_EXTENSIONS_DIR.
#
# Scope: compile-time (-D) variants of the extension. Variants that change a HEADER DEFAULT
# compiled by flashinfer's own JIT module (the FA2 path) still need cached_ops cleared and
# cannot share a build round -- the harness globs ~/.cache/flashinfer for that module.
set -u
W=/dev/shm/w
cd $W/fi
. $W/venv/bin/activate
export PYTHONPATH=$W/python:$W/fi HF_HOME=$W/hf FI_ROOT=$W/fi TORCH_CUDA_ARCH_LIST=9.0a
FIX="$1"; shift
BASE="PAGE_SIZE=128 TWO_EXEC=1 TWO_EXEC_OVERLAP=0 FA3_UPSTREAM=1 UP_STAGES=2 MMA_PV_RS=1 SCHED_LPT=0 FA3_PAGED_TMA=0 FA2_MMA_KV=2 FA3_SM_BUDGET=132 UNIFIED=1 SPLITS=auto FLASHINFER_SM_PARTITION=1 FLASHINFER_FA3_SM_SPLIT_CLUSTERS=124"
mkdir -p $W/abp
echo "##### phase 1: building ${#@} arms in parallel"
pids=()
for a in "$@"; do
  tag=${a%%:*}; envs=${a#*:}; [ "$envs" = "$a" ] && envs=""
  rm -rf $W/pb_$tag
  ( env $BASE $envs TORCH_EXTENSIONS_DIR=$W/pb_$tag BUILD_ONLY=1 FIXTURE="$FIX" \
      timeout 2400 python tests/test_fused_server_batch.py > $W/abp/$tag.build 2>&1
    echo "$tag build: $(grep -c 'BUILD_ONLY: extension built' $W/abp/$tag.build)" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
echo "##### phase 2: running arms on the GPU"
for a in "$@"; do
  tag=${a%%:*}; envs=${a#*:}; [ "$envs" = "$a" ] && envs=""
  env $BASE $envs TORCH_EXTENSIONS_DIR=$W/pb_$tag TIME_ITERS=5 FIXTURE="$FIX" \
    timeout 1800 python tests/test_fused_server_batch.py > $W/abp/$tag.full 2>&1
  t=$(grep -oE 'TWO_EXEC iters=[0-9]+ total=[0-9.]+' $W/abp/$tag.full | grep -oE '[0-9.]+$')
  ok=$(grep -c '^PASS' $W/abp/$tag.full)
  unw=$(grep -oE 'unwritten \(NaN\) rows = [0-9]+' $W/abp/$tag.full | head -1 | grep -oE '[0-9]+$')
  printf "%-14s total=%-10s us  PASS=%s  unwritten=%s  fallbacks=%s  [%s]\n" \
    "$tag" "${t:-NA}" "$ok" "${unw:-NA}" "$(grep -ci fallback $W/abp/$tag.full)" "$envs"
done
echo ABPDONE
