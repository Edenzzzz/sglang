#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
prototype=$workdir/stock_fa3_pdl_work_20261014
root=$workdir/stock_fa3_pdl_20261014
trap 'status=$?; printf "STOCK_FA3_PDL_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
if [ "${INHERIT_LOCK:-0}" = 1 ]; then
  test "${GPU_LOCK_OWNER_PID:-}" = 766826
  kill -0 766826
  test "$(readlink /proc/$$/fd/9)" = "$workdir/a2_kernel.lock"
  test -s "$workdir/b1_kernel_concurrency_20261014/mix128_c24/BASE_2/kernel_concurrency_audit.json"
else
  exec 9>"$workdir/a2_kernel.lock"
fi
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[k]ernel_concurrency_driver|[s]tock_fa3_reference.py' >/dev/null; then
  echo 'GPU predecessor still running; refusing duplicate launch'
  exit 1
fi
test ! -e "$root"
mkdir "$root"
cd "$prototype"
sha256sum -c sources.sha256
cp sources.sha256 "$root/"
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
echo "STOCK_FA3_BUILD_START $(date -Is)"
TORCH_EXTENSIONS_DIR=$STOCK_FA3_EXTENSIONS timeout --kill-after=30 1800 "$workdir/venv/bin/python" -u load_stock_fa3.py > "$root/build.log" 2>&1
grep -q -- '--use_fast_math' "$root/build.log"
grep -q -- '-DNDEBUG' "$root/build.log"
cuobjdump --dump-resource-usage "$STOCK_FA3_EXTENSIONS/stock_fa3_pdl/stock_fa3_pdl.so" > "$root/resources.txt"
echo "STOCK_FA3_UNIT_START $(date -Is)"
TORCH_EXTENSIONS_DIR=$STOCK_FA3_EXTENSIONS timeout --kill-after=30 900 "$workdir/venv/bin/python" -u test_stock_fa3.py > "$root/tests.log" 2>&1
echo "STOCK_FA3_UNIT_PASS $(date -Is)"
driver=$prototype/stock_fa3_reference.py
echo "NATIVE_RELEASE_GATE_START $(date -Is)"
NATIVE_PDL_MODE=independent MODE=fa3_1chunk TIME_AB=0 \
  timeout --kill-after=30 1800 "$workdir/venv/bin/python" -u "$driver" > "$root/native_release_gate.log" 2>&1
grep -q '^PASS$' "$root/native_release_gate.log"
grep -q -- '-DNDEBUG' "$TORCH_EXTENSIONS_DIR/fused_fa3_q128/build.ninja"
grep -q -- '--use_fast_math' "$TORCH_EXTENSIONS_DIR/fused_fa3_q128/build.ninja"
cp "$TORCH_EXTENSIONS_DIR/fused_fa3_q128/build.ninja" "$root/native_release_build.ninja"
for mode in stock_wrapped stock_pdl; do
  for query in 32 2048; do
    require_overlap=0
    if [ "$mode" = stock_pdl ] && [ "$query" = 2048 ]; then require_overlap=1; fi
    label=gate_${mode}_q${query}
    echo "STOCK_FA3_GATE_START $label $(date -Is)"
    NATIVE_PDL_MODE=$mode NATIVE_PDL_TRACE=1 NATIVE_PDL_REQUIRE_OVERLAP=$require_overlap \
      MODE=mixed_verify PREFILL_QO=$query PREFILL_KV=65536 VERIFY_QO=4 VERIFY_ROWS=8 VERIFY_KV=8192 TIME_AB=0 \
      timeout --kill-after=30 900 "$workdir/venv/bin/python" -u "$driver" > "$root/$label.log" 2>&1
    "$workdir/venv/bin/python" - "$root/$label.log" "$mode" "$query" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
log = path.read_text()
assert log.rstrip().endswith("PASS") and "Traceback" not in log and "NON-FINITE OUTPUT" not in log
assert log.count("NATIVE_PDL_TRACE mode=" + sys.argv[2]) == 2
rows = re.findall(r"bad rows (\d+)/(\d+)", log)
assert rows and all(int(bad) == 0 and int(total) > 0 for bad, total in rows)
if sys.argv[2] == "stock_pdl" and sys.argv[3] == "2048":
    assert log.count("completed_fa2_during_fa3=True") == 2
print("STOCK_FA3_GATE_PASS", path.name, flush=True)
PY
  done
done
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
    modes='independent stock stock_wrapped stock_pdl'
    if [ "$repeat" = 2 ]; then budgets='all 112'; modes='stock_pdl stock_wrapped stock independent'; fi
    for budget in $budgets; do
      for mode in $modes; do
        label=${shape%%|*}_sm${budget}_${mode}_$repeat
        echo "STOCK_FA3_CASE_START $label $(date -Is)"
        SM_SPLIT=$budget NATIVE_PDL_MODE=$mode TIME_AB=1 env "${fixture[@]}" \
          timeout --kill-after=30 1200 "$workdir/venv/bin/python" -u "$driver" > "$root/$label.log" 2>&1
        "$workdir/venv/bin/python" "$prototype/audit_stock_fa3.py" "$root/$label.log" "$mode" "$budget" "$repeat"
      done
    done
  done
done
echo "STOCK_FA3_PDL_DONE $(date -Is)"
