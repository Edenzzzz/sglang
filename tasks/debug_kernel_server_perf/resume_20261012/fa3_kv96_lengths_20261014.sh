#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
output_dir=$workdir/fa3_kv96_lengths_20261014
trap 'status=$?; printf "FA3_KV96_LENGTHS_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
cd "$workdir"
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[t]ma_.*_reference|[f]a3_kv_geometry_reference|[k]v96_debug.py' >/dev/null; then
  echo "GPU workload still running"
  exit 1
fi
test ! -e "$output_dir"
mkdir "$output_dir"
export PYTHONPATH=$workdir/up_fa3_kv_geometry/python HF_HOME=$workdir/hf
export SGLANG_FUSED_FI_ROOT=$workdir/fi_fa3_kv_geometry
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 FUSED_FA3_DEDICATED_PRIMARY=1
export FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7 FA3_STAGES=2 SM_SPLIT=112
export FUSED_PAGED_TMA_BUILD=0 FUSED_CLUSTER=1 FLASHINFER_FUSED_CLUSTER=1 FUSED_FA3_CLUSTER_PAIR=0
export SGLANG_FUSED_FA3_CONSUMER_REGS=216 FA3_CTA_KV=96 TIME_AB=0 SEED=31
export SGLANG_FUSED_MERGED_DIR=$workdir/merged_fa3_kv_96 TORCH_EXTENSIONS_DIR=$workdir/torch_ext_fa3_kv_96
unset NATIVE_PDL_REQUIRE_OVERLAP NATIVE_PDL_TRACE FUSED_NOSPLIT FA3_PAGED_TMA SPLIT_SMEM_O
export FUSED_NOSPLIT=1
for context in 1 2 95 96 97 128 129 192 193 384 4096; do
  queries=17
  if [ "$context" -lt "$queries" ]; then queries=$context; fi
  label=q${queries}_kv${context}
  echo "START $label $(date -Is)"
  status=0
  FA3_REQUESTS=$queries:$context FA2_REQUESTS=, timeout --kill-after=20 180 \
    "$workdir/venv/bin/python" -u "$workdir/drivers/kv96_debug.py" > "$output_dir/$label.log" 2>&1 || status=$?
  echo "RESULT $label status=$status $(date -Is)"
  grep '^KV96_DEBUG ' "$output_dir/$label.log" || tail -n 5 "$output_dir/$label.log"
done
echo "FA3_KV96_LENGTHS_DONE $(date -Is)"

