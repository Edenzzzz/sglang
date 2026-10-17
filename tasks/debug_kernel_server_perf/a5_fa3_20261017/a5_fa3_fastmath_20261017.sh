#!/bin/bash
# A5.0b: our FA3 built with vs without --use_fast_math (upstream FA3 builds with it), same fixtures, 112 CTAs,
# arms interleaved, upstream FA3 timed in every run. Tree up_a5 = up_tma_multicast + FUSED_FAST_MATH loader switch.
# Otherwise the binaries and fixtures of a5_fa3_gap_20261017.sh (the OFF arm reuses its build). Every arm checked
# against the fp32 causal reference (9 checks). Waits for a2_kernel.lock; yields to /dev/shm/w/codex_needs_gpu.
set -uo pipefail
W=/dev/shm/w
out=$W/a5_fa3_fastmath_20261017
trap 'printf "A5FM_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
exec 9>"$W/a2_kernel.lock"; flock 9
cd "$W"; mkdir -p "$out"
export PYTHONPATH=$W/up_a5/python HF_HOME=$W/hf
export SGLANG_FUSED_FI_ROOT=$W/fi_tma_multicast TORCH_CUDA_ARCH_LIST=9.0a
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 FUSED_FA3_DEDICATED_PRIMARY=1
export FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7 FA3_STAGES=2 FUSED_FA3_TMA_TRACE=0
export SGLANG_FUSED_MERGED_DIR=$W/merged_tma_gather TORCH_EXTENSIONS_DIR=$W/torch_ext_tma_gather
export FUSED_PAGED_TMA_BUILD=0 FUSED_CLUSTER=1 FLASHINFER_FUSED_CLUSTER=1 FUSED_FA3_CLUSTER_PAIR=0 FUSED_FA3_TMA_MULTICAST=0
export TIME_AB=1 TIME_ITERS=100 SEED=31 FA2_REQUESTS=,
unset FUSED_NOSPLIT FUSED_KV_SHARE_BUILD FUSED_FA3_KV_SHARE FUSED_FA3_KV_SHARE_BULK
unset FLASHINFER_FUSED_SYMMETRIC FLASHINFER_FUSED_SKIP_FA3 FLASHINFER_FUSED_PLAN_DEBUG FA3_SKIP_KV_LOAD TIME_FORCE
unset FA3_PAGED_TMA NATIVE_PDL_REQUIRE_OVERLAP NATIVE_PDL_TRACE
rep() { local n=$1 s=$2 o=""; for _ in $(seq $n); do o+="$s,"; done; echo "${o%,}"; }
# name|FA3_REQUESTS (q_len:kv_len): NCU fixture; chat prefill chunks; TraceLab appends; long prefill; short rows (A1 q64 region)
shapes=("ncu8x1024_kv32k|$(rep 8 1024:32768)"
        "chat_fresh2048|2048:2048"
        "chat_fresh4x512|$(rep 4 512:512)"
        "append512_kv48k|512:49152"
        "agent4x512_kv64k|$(rep 4 512:65536)"
        "prefill2048_kv64k|2048:65536"
        "short16x16_kv1k|$(rep 16 16:1024)")
for repeat in 1 2; do
  arms=(OFF ON); [ $repeat = 2 ] && arms=(ON OFF)
  for shape in "${shapes[@]}"; do
    name=${shape%%|*}
    for arm in "${arms[@]}"; do
      [ -s $W/codex_needs_gpu ] && { echo "A5FM_YIELD $name $arm $repeat"; exit 0; }
      if [ $arm = ON ]; then export FUSED_FAST_MATH=1 TORCH_EXTENSIONS_DIR=$W/torch_ext_a5_fastmath
      else export FUSED_FAST_MATH=0 TORCH_EXTENSIONS_DIR=$W/torch_ext_tma_gather; fi
      label=${name}_fm${arm}_$repeat
      echo "A5FM_START $label $(date -Is)"
      env FA3_REQUESTS="${shape#*|}" SM_SPLIT=112 timeout --kill-after=20 900 "$W/venv/bin/python" -u \
        "$W/drivers/a5_fa3_reference_20261017.py" > "$out/$label.log" 2>&1
      rc=$?
      echo "A5FM_END $label rc=$rc checks=$(grep -c '^CHECK .*bad_heads=0/' $out/$label.log)/9 $(grep -o '^TIME [a-z]* graph=True ms=[0-9.]*' $out/$label.log | awk '{print $2"="$4}' | tr '\n' ' ')"
    done
  done
done
echo "A5FM_DONE $(date -Is)"
