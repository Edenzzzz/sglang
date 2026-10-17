#!/bin/bash
# A5.0: where does our persistent FA3 lose to stock? FA3-only fixtures shaped like serving's FA3 rows; our dedicated
# gather primary at 112 and 132 CTAs vs flashinfer FA3 BatchPrefill ("stock") vs upstream sgl-kernel FA3 ("upstream",
# what serving's split runs). Same validated binaries as tma_sm_budget_20261013.sh's GATHER arm. Every arm checked
# against the fp32 causal reference (9 checks). Waits for a2_kernel.lock; yields to /dev/shm/w/codex_needs_gpu.
set -uo pipefail
W=/dev/shm/w
out=$W/a5_fa3_gap_20261017
trap 'printf "A5GAP_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
exec 9>"$W/a2_kernel.lock"; flock 9
cd "$W"; mkdir -p "$out"
export PYTHONPATH=$W/up_tma_multicast/python HF_HOME=$W/hf
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
  budgets=(112 132); [ $repeat = 2 ] && budgets=(132 112)
  for shape in "${shapes[@]}"; do
    name=${shape%%|*}
    for b in "${budgets[@]}"; do
      [ -s $W/codex_needs_gpu ] && { echo "A5GAP_YIELD $name sm$b $repeat"; exit 0; }
      label=${name}_sm${b}_$repeat
      echo "A5GAP_START $label $(date -Is)"
      env FA3_REQUESTS="${shape#*|}" SM_SPLIT=$b timeout --kill-after=20 600 "$W/venv/bin/python" -u \
        "$W/drivers/a5_fa3_reference_20261017.py" > "$out/$label.log" 2>&1
      rc=$?
      echo "A5GAP_END $label rc=$rc checks=$(grep -c '^CHECK .*bad_heads=0/' $out/$label.log)/9 $(grep -o '^TIME [a-z]* graph=True ms=[0-9.]*' $out/$label.log | awk '{print $2"="$4}' | tr '\n' ' ')"
    done
  done
done
echo "A5GAP_DONE $(date -Is)"
