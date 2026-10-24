#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
prototype=$workdir/stock_fa3_pdl_work_20261014
root=$workdir/stock_fa3_public_roles_20261014
predecessor=939307
trap 'status=$?; printf "STOCK_PUBLIC_ROLES_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
echo "STOCK_PUBLIC_ROLES_WAIT pid=$predecessor $(date -Is)"
while kill -0 "$predecessor" 2>/dev/null; do
  state=$(ps -p "$predecessor" -o stat= || true)
  if [[ "$state" == Z* ]]; then break; fi
  command=$(tr '\0' ' ' < "/proc/$predecessor/cmdline")
  [[ "$command" == 'bash stock_fa3_role_control_20261014.sh '* ]]
  sleep 15
done
grep -q '^STOCK_ROLE_CONTROL_COMPLETE ' "$workdir/stock_fa3_role_control_20261014.log"
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[c]orrectness.py|[k]ernel_concurrency_driver|[s]trict_phase_reference.py' >/dev/null; then exit 1; fi
test ! -e "$root"
mkdir "$root"
cd "$prototype"
sha256sum -c sources.sha256 > "$root/source_check.log"
export PYTHONPATH=$workdir/up_fa3_kv_geometry_v2/python HF_HOME=$workdir/hf
export SGLANG_FUSED_FI_ROOT=$workdir/fi_fa3_kv_geometry_v2
export SGLANG_FUSED_MERGED_DIR=$workdir/merged_native_release_stockcmp
export TORCH_EXTENSIONS_DIR=$workdir/torch_ext_native_release_stockcmp
export NATIVE_RELEASE_BUILD=1
export STOCK_FA3_EXTENSIONS=$workdir/torch_ext_stock_fa3_pdl
export STOCK_FA3_MERGED=$workdir/merged_stock_fa3_pdl
export STOCK_FLASH_ATTENTION=$prototype/flash-attention
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 HEADS=28,4 SM_SPLIT=112
export FUSED_REFERENCE_WORKSPACE_BYTES=1073741824 FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7
export FA3_CTA_KV=128 FA3_STAGES=2 SGLANG_FUSED_FA3_CONSUMER_REGS=216
export FUSED_FA3_DEDICATED_PRIMARY=1 FUSED_FA3_CLUSTER_PAIR=1
export FUSED_PAGED_TMA_BUILD=0 FUSED_CLUSTER=1 FLASHINFER_FUSED_CLUSTER=1
export TIME_CUDA_GRAPH=1 TIME_ITERS=100 TIME_WARMUP=20
unset FUSED_NOSPLIT FUSED_KV_SHARE_BUILD FUSED_FA3_COMPACT_ABI FUSED_FA3_NOSPLIT_SPECIALIZE
unset SGLANG_FUSED_FA3_STATIC_GQA FUSED_KV_CHUNK_MULT FA2_REQUESTS FA3_REQUESTS FUSED_FA3_UNPACKED
unset FLASHINFER_FUSED_SYMMETRIC FLASHINFER_FUSED_SKIP_FA3 FLASHINFER_FUSED_PLAN_DEBUG
unset FA3_SKIP_KV_LOAD FLASHINFER_FA3_SKIP_KV_LOAD TIME_FORCE SGLANG_EDF_SCHEDULING SGLANG_DEBUG_STEP_SYNC

export PYTHONPATH=$prototype:$PYTHONPATH
export TIME_AB=0 NATIVE_PDL_MODE=stock_pdl STRICT_PDL_COMPARE=1 ORACLE=fp32
driver=$workdir/stock_fa3_public_role_reference_20261014.py
sha256sum "$driver" stock_fa3_pdl.cu load_stock_fa3.py > "$root/sources.sha256"
shapes=(
  'prefill48|PREFILL_QO=512 PREFILL_KV=49152 VERIFY_ROWS=8 VERIFY_KV=4096 STOCK_FA3_SPLITS=1'
  'split48|PREFILL_QO=16 PREFILL_KV=49152 VERIFY_ROWS=8 VERIFY_KV=49152 STOCK_FA3_SPLITS=7'
  'prefill128|PREFILL_QO=512 PREFILL_KV=131072 VERIFY_ROWS=8 VERIFY_KV=32768 STOCK_FA3_SPLITS=4'
)
for shape in "${shapes[@]}"; do
  read -r -a fixture <<< "${shape#*|}"
  for budget in 112 all; do
    label=${shape%%|*}_sm$budget
    echo "STOCK_PHASE_CASE_START $label $(date -Is)"
    MODE=mixed_verify VERIFY_QO=4 SM_SPLIT=$budget env "${fixture[@]}" \
      timeout --kill-after=30 1200 "$workdir/venv/bin/python" -u "$driver" > "$root/$label.log" 2>&1
    grep -q '^STRICT_PDL_COMPARE_PASS$' "$root/$label.log"
    test "$(grep -c '^STRICT_PDL_COMPARE iteration=' "$root/$label.log")" = 4
    test "$(grep -c '^PUBLIC_ROLE_COMPARE iteration=' "$root/$label.log")" = 8
    tail -9 "$root/$label.log"
    echo "STOCK_PHASE_CASE_PASS $label $(date -Is)"
  done
done
echo "STOCK_PUBLIC_ROLES_COMPLETE $(date -Is)"
