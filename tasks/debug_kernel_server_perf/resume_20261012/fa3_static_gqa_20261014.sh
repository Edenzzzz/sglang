#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
root=$workdir/fa3_static_gqa_20261014
overlay=$workdir/fa3_static_gqa_overlay_20261014
owner=766826
trap 'status=$?; printf "FA3_STATIC_GQA_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
if [ "${INHERIT_LOCK:-0}" = 1 ]; then
  test "${GPU_LOCK_OWNER_PID:-}" = "$owner"
  kill -0 "$owner"
  test "$(readlink /proc/$$/fd/9)" = "$workdir/a2_kernel.lock"
  test "$(readlink /proc/$owner/fd/9)" = "$workdir/a2_kernel.lock"
  test -s "$workdir/b1_kernel_concurrency_20261014/mix64_c48/BASE_2/kernel_concurrency_audit.json"
else
  exec 9>"$workdir/a2_kernel.lock"
fi
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[k]ernel_concurrency_driver|[f]used_reference' >/dev/null; then
  echo 'GPU predecessor still running; refusing duplicate launch'
  exit 1
fi
cd "$workdir"
sha256sum -c "$overlay/baseline.sha256"
test ! -e fi_fa3_static_gqa
test ! -e up_fa3_static_gqa
test ! -e "$root"
mkdir "$root" up_fa3_static_gqa
cp -a fi_fa3_kv_geometry_v2 fi_fa3_static_gqa
cp -a up_fa3_kv_geometry_v2/python up_fa3_static_gqa/python
cp -a "$overlay/include/." fi_fa3_static_gqa/include/
cp "$overlay/fused_fa3_persistent_ext.cu" fi_fa3_static_gqa/tests/fused_fa3_persistent_ext.cu
cp "$overlay/fused_fa3_loader.py" up_fa3_static_gqa/python/sglang/srt/layers/attention/fused_fa3_loader.py
cp "$overlay/fused_reference_native_pdl.py" drivers/fused_reference_static_gqa.py
sha256sum -c "$overlay/candidate.sha256"
cp "$overlay/"{baseline,candidate}.sha256 "$root/"
export PYTHONPATH=$workdir/up_fa3_static_gqa/python HF_HOME=$workdir/hf
export SGLANG_FUSED_FI_ROOT=$workdir/fi_fa3_static_gqa
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 HEADS=28,4 SM_SPLIT=112
export FUSED_REFERENCE_WORKSPACE_BYTES=1073741824 FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7
export FA3_CTA_KV=128 FA3_STAGES=2 SGLANG_FUSED_FA3_CONSUMER_REGS=216
export FUSED_FA3_DEDICATED_PRIMARY=1 FUSED_FA3_CLUSTER_PAIR=1
export FUSED_PAGED_TMA_BUILD=0 FUSED_CLUSTER=1 FLASHINFER_FUSED_CLUSTER=1
export TIME_CUDA_GRAPH=1 TIME_ITERS=100 TIME_WARMUP=20 NATIVE_PDL_MODE=independent
unset FUSED_NOSPLIT FUSED_KV_SHARE_BUILD FUSED_FA3_COMPACT_ABI FUSED_FA3_NOSPLIT_SPECIALIZE
unset FUSED_KV_CHUNK_MULT FA2_REQUESTS FA3_REQUESTS FUSED_FA3_UNPACKED
unset FLASHINFER_FUSED_SYMMETRIC FLASHINFER_FUSED_SKIP_FA3 FLASHINFER_FUSED_PLAN_DEBUG
unset FA3_SKIP_KV_LOAD TIME_FORCE SGLANG_EDF_SCHEDULING SGLANG_DEBUG_STEP_SYNC
driver=$workdir/drivers/fused_reference_static_gqa.py
select_gqa() {
  export SGLANG_FUSED_FA3_STATIC_GQA=$1
  export SGLANG_FUSED_MERGED_DIR=$workdir/merged_fa3_static_gqa$1
  export TORCH_EXTENSIONS_DIR=$workdir/torch_ext_fa3_static_gqa$1
}
for group in 0 7; do
  select_gqa "$group"
  echo "FA3_STATIC_GQA_BUILD_START group=$group $(date -Is)"
  timeout --kill-after=30 1800 "$workdir/venv/bin/python" -u -c \
    'from sglang.srt.layers.attention.fused_fa3_loader import load_fused_fa3; print(load_fused_fa3(verbose=True, cta_q=128))' \
    > "$root/build_gqa$group.log" 2>&1
  grep -q -- "-DFUSED_FA3_STATIC_GQA=$group" "$root/build_gqa$group.log"
  sha256sum "$TORCH_EXTENSIONS_DIR/fused_fa3_q128/fused_fa3_q128.so" > "$root/binary_gqa$group.sha256"
  cuobjdump --dump-resource-usage "$TORCH_EXTENSIONS_DIR/fused_fa3_q128/fused_fa3_q128.so" > "$root/resources_gqa$group.txt"
  for query in 32 2048; do
    overlap=0
    if [ "$query" = 2048 ]; then overlap=1; fi
    label=gate_gqa${group}_q${query}
    echo "FA3_STATIC_GQA_GATE_START $label $(date -Is)"
    NATIVE_PDL_TRACE=1 NATIVE_PDL_REQUIRE_OVERLAP=$overlap NATIVE_PDL_GRAPH_PLAN=1 \
      MODE=mixed_verify PREFILL_QO=$query PREFILL_KV=65536 VERIFY_QO=4 VERIFY_ROWS=8 VERIFY_KV=8192 TIME_AB=0 \
      timeout --kill-after=30 900 "$workdir/venv/bin/python" -u "$driver" > "$root/$label.log" 2>&1
    "$workdir/venv/bin/python" - "$root/$label.log" "$query" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
log = path.read_text()
assert log.rstrip().endswith("PASS") and "Traceback" not in log and "NON-FINITE OUTPUT" not in log
assert log.count("NATIVE_PDL_TRACE mode=independent") == 2
if sys.argv[2] == "2048":
    assert log.count("completed_fa2_during_fa3=True") == 2
rows = re.findall(r"bad rows (\d+)/(\d+)", log)
assert rows and all(int(bad) == 0 and int(total) > 0 for bad, total in rows)
print("FA3_STATIC_GQA_GATE_PASS", path.name, flush=True)
PY
  done
done
select_gqa 7
set +e
HEADS=28,7 MODE=fa3_1chunk TIME_AB=0 timeout --kill-after=30 300 "$workdir/venv/bin/python" -u "$driver" \
  > "$root/wrong_ratio.log" 2>&1
wrong_ratio_status=$?
set -e
test "$wrong_ratio_status" -ne 0
grep -q 'FA3 static GQA specialization does not match head ratio' "$root/wrong_ratio.log"
echo "FA3_STATIC_GQA_REJECTION_PASS status=$wrong_ratio_status $(date -Is)"
shapes=(
  'prefill2048_v8|MODE=mixed_verify PREFILL_QO=2048 PREFILL_KV=65536 VERIFY_QO=4 VERIFY_ROWS=8 VERIFY_KV=16384'
  'append512_v46|MODE=mixed_verify PREFILL_QO=512 PREFILL_KV=49152 VERIFY_QO=4 VERIFY_ROWS=46 VERIFY_KV=49152'
  'fa3_8x1024|MODE=band1 BAND_SIDE=fa3 BAND_QO=1024 BAND_KV=32768 BAND_ROWS=8 FA2_REQUESTS=,'
  'split_q32_v8|MODE=mixed_verify PREFILL_QO=32 PREFILL_KV=65536 VERIFY_QO=4 VERIFY_ROWS=8 VERIFY_KV=8192'
  'prefill8192_v24|MODE=mixed_verify PREFILL_QO=8192 PREFILL_KV=65536 VERIFY_QO=4 VERIFY_ROWS=24 VERIFY_KV=32768'
)
for shape in "${shapes[@]}"; do
  read -r -a fixture <<< "${shape#*|}"
  for repeat in 1 2; do
    budgets='112 all'
    groups='0 7'
    if [ "$repeat" = 2 ]; then budgets='all 112'; groups='7 0'; fi
    for budget in $budgets; do
      export SM_SPLIT=$budget
      for group in $groups; do
        select_gqa "$group"
        label=${shape%%|*}_sm${budget}_gqa${group}_$repeat
        echo "FA3_STATIC_GQA_CASE_START $label $(date -Is)"
        TIME_AB=1 env "${fixture[@]}" timeout --kill-after=30 1200 "$workdir/venv/bin/python" -u "$driver" \
          > "$root/$label.log" 2>&1
        "$workdir/venv/bin/python" "$overlay/audit_static_gqa.py" "$root/$label.log" "$group" "$budget" "$repeat"
      done
    done
  done
done
echo "FA3_STATIC_GQA_DONE $(date -Is)"
