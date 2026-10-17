#!/bin/bash
# A5.0d first build: our dedicated FA3 primary with FLASHINFER_FA3_UPSTREAM=1 (flash-attention's mainloop + persistent
# scheduler on the sm_split CTAs) + fast math, one fixture, full 9-check harness. Usage: bash <this> <fixture> <sm>
set -uo pipefail
W=/dev/shm/w
out=$W/a5_fa3_upstream_20261017; mkdir -p $out
exec 9>"$W/a2_kernel.lock"; flock 9
cd "$W"
export PYTHONPATH=$W/up_a5/python HF_HOME=$W/hf
export SGLANG_FUSED_FI_ROOT=$W/fi_a5 TORCH_CUDA_ARCH_LIST=9.0a
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 FUSED_FA3_DEDICATED_PRIMARY=1
export FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7 FA3_STAGES=2 FUSED_FA3_TMA_TRACE=0
# PTMA=1: upstream paged-TMA KV loads (PagedKVNonTMA=false; page 128 is a multiple of kBlockN 128).
PTMA=${PTMA:-0}
# XN: extra nvcc flag set name (FUSED_EXTRA_NVCC), each set built into its own extension dir.
XN=${XN:-none}
case $XN in none) export FUSED_EXTRA_NVCC= ;; ndebug) export FUSED_EXTRA_NVCC=-DNDEBUG ;;
  fa) export FUSED_EXTRA_NVCC="-DNDEBUG -DCUTLASS_ENABLE_GDC_FOR_SM90 -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DCUTLASS_DEBUG_TRACE_LEVEL=0" ;;
  # c2: flash-attention's ClusterM=2 (TMA K/V multicast to the two CTAs of a cluster); the dedicated launch must form
  # the same cluster.
  c2) export FUSED_EXTRA_NVCC="-DNDEBUG -DFLASHINFER_FA3_CLUSTER_M=2" FLASHINFER_FUSED_CLUSTER=2 ;; esac
export SGLANG_FUSED_MERGED_DIR=$W/merged_a5 TORCH_EXTENSIONS_DIR=$W/torch_ext_a5_upstream$([ "${UPS:-1}" = 0 ] && echo _fimain)$([ $PTMA = 1 ] && echo _ptma)$([ $XN != none ] && echo _$XN)
# fa3_upstream.cuh includes ../../../../../flash-attention/hopper relative to the merged include tree.
mkdir -p $W/merged_a5; ln -sfn $W/flash-attention $W/merged_a5/flash-attention
export FUSED_PAGED_TMA_BUILD=0 FUSED_CLUSTER=1 FLASHINFER_FUSED_CLUSTER=${FLASHINFER_FUSED_CLUSTER:-1} FUSED_FA3_CLUSTER_PAIR=0 FUSED_FA3_TMA_MULTICAST=0
export FA3_UPSTREAM=${UPS:-1} FUSED_FAST_MATH=1
export TIME_AB=1 TIME_ITERS=100 SEED=31 FA2_REQUESTS=${FA2_OVERRIDE:-,}
unset FUSED_NOSPLIT FLASHINFER_FUSED_SYMMETRIC FLASHINFER_FUSED_SKIP_FA3 FA3_SKIP_KV_LOAD TIME_FORCE
export FA3_PAGED_TMA=$PTMA
env FA3_REQUESTS="$1" SM_SPLIT=$2 timeout --kill-after=20 1500 ${SANITIZE:+/usr/local/cuda/bin/compute-sanitizer --tool memcheck --show-backtrace device --print-limit 8} $W/venv/bin/python -u $W/drivers/a5_fa3_reference_20261017.py \
  > $out/try${SANITIZE:+_sanitize}${TAG:+_$TAG}_up${UPS:-1}_ptma${PTMA}_x${XN}_$(echo $1 | tr ':,' '_-' | cut -c1-40)_sm$2.log 2>&1
echo "A5UP rc=$? $(date -Is)"
