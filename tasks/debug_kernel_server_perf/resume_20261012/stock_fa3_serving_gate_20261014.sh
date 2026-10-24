#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
root=$workdir/stock_fa3_serving_gate_20261014
overlay=$workdir/stock_fa3_serving_overlay_20261014
predecessor=766826
trap 'status=$?; printf "STOCK_FA3_SERVING_GATE_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
echo "STOCK_FA3_SERVING_WAIT parent=$predecessor $(date -Is)"
while kill -0 "$predecessor" 2>/dev/null; do
  state=$(ps -p "$predecessor" -o stat= || true)
  if [[ "$state" == Z* ]]; then break; fi
  command=$(tr '\0' ' ' < "/proc/$predecessor/cmdline")
  [[ "$command" == 'bash b1_kernel_concurrency_20261014.sh '* ]]
  sleep 30
done
grep -q '^B1_KERNEL_CONCURRENCY_DONE ' "$workdir/b1_kernel_concurrency_20261014.log"
grep -q '^STOCK_FA3_PDL_DONE ' "$workdir/stock_fa3_pdl_midpoint_20261014.log"
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[k]ernel_concurrency_driver|[s]tock_fa3_reference.py' >/dev/null; then
  echo 'GPU predecessor still running; refusing duplicate launch'
  exit 1
fi
cd "$workdir"
sha256sum -c "$overlay/baseline.sha256"
test ! -e up_stock_fa3_serving
test ! -e "$root"
mkdir up_stock_fa3_serving "$root"
cp -a up_fa3_kv_geometry_v2/python up_stock_fa3_serving/python
cp "$overlay/flashinfer_backend.py" up_stock_fa3_serving/python/sglang/srt/layers/attention/flashinfer_backend.py
cp "$overlay/fused_fa3_loader.py" up_stock_fa3_serving/python/sglang/srt/layers/attention/fused_fa3_loader.py
sha256sum -c "$overlay/candidate.sha256"
cp "$overlay/"{baseline,candidate}.sha256 "$root/"
source "$workdir/tree_env.sh"
export PYTHONPATH=$workdir/up_stock_fa3_serving/python:$workdir/stock_fa3_pdl_work_20261014
export HF_HOME=$workdir/hf MODEL=$workdir/qwen7b
export SGLANG_FUSED_FI_ROOT=$workdir/fi_fa3_kv_geometry_v2
export SGLANG_FUSED_MERGED_DIR=$workdir/merged_stock_serving_native_release
export TORCH_EXTENSIONS_DIR=$workdir/torch_ext_stock_serving_native_release
export STOCK_FA3_EXTENSIONS=$workdir/torch_ext_stock_fa3_pdl
export STOCK_FA3_MERGED=$workdir/merged_stock_fa3_pdl
export STOCK_FLASH_ATTENTION=$workdir/stock_fa3_pdl_work_20261014/flash-attention
export NATIVE_RELEASE_BUILD=1 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 SGLANG_FUSED_SM_FA3=112
export FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7 FA3_CTA_KV=128 FA3_STAGES=2
export SGLANG_FUSED_FA3_CONSUMER_REGS=216 FUSED_FA3_DEDICATED_PRIMARY=1 FUSED_FA3_CLUSTER_PAIR=1
export FUSED_PAGED_TMA_BUILD=0 FUSED_CLUSTER=1 FLASHINFER_FUSED_CLUSTER=1
export SGLANG_SPLIT_OVERLAP=0 MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
export CHAT_DATASET=$workdir/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
unset FUSED_NOSPLIT FUSED_KV_SHARE_BUILD FUSED_FA3_COMPACT_ABI FUSED_FA3_NOSPLIT_SPECIALIZE
unset SGLANG_FUSED_FA3_STATIC_GQA FUSED_KV_CHUNK_MULT FUSED_FA3_UNPACKED
unset SGLANG_EDF_SCHEDULING SGLANG_EDF_LATENCY_MODEL SGLANG_EDF_ACCEPT_MODE SGLANG_DEBUG_STEP_SYNC
unset SGLANG_FUSED_CHAIN_TREE SGLANG_FUSED_VERIFY_EPILOGUE DISABLE_CUDA_GRAPH SGLANG_STEP_TRACE
unset SGLANG_TWO_EXEC SGLANG_MIXED_ATTENTION_KERNEL SGLANG_ALLOW_MIXED_CHUNK_SPEC SGLANG_MIXED_SPEC
unset SGLANG_FUSED_KERNEL_EXEC SGLANG_FUSED_DISPATCH SGLANG_FUSED_DECODE_GUARD
driver=$workdir/drivers/srv_correctness_edf_20261014.py
for arm in MTPMIX_SPLIT_STOCK_CONTROL MTPMIXFUSED_RELEASE_NATIVE MTPMIXFUSED_STOCK_SERIAL MTPMIXFUSED_STOCK_VERIFY MTPMIXFUSED_STOCK_PDL; do
  native=1
  stock=1
  phase=2
  verify=0
  case "$arm" in
    MTPMIX_SPLIT_STOCK_CONTROL) native=0; stock=0 ;;
    MTPMIXFUSED_RELEASE_NATIVE) stock=0 ;;
    MTPMIXFUSED_STOCK_SERIAL) phase=0 ;;
    MTPMIXFUSED_STOCK_VERIFY) verify=1 ;;
  esac
  echo "STOCK_FA3_SERVING_ARM_START $arm $(date -Is)"
  CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET="$CHAT_DATASET" BENCH_CHUNK=512 \
    CORRECTNESS_DECODE_TOKENS=1200 SGLANG_FUSED_DECODE_GUARD=0 \
    SGLANG_FUSED_NATIVE_PDL=$native SGLANG_STOCK_FA3_PDL=$stock SGLANG_STOCK_FA3_PHASE=$phase \
    SGLANG_FUSED_SPLIT_VERIFY=$verify SGLANG_FUSED_VERIFY_SNAPSHOT=$verify SGLANG_FUSED_SPLIT_TOL=0.02 \
    timeout --kill-after=30 1800 "$workdir/venv/bin/python" -u "$driver" "$arm" "$root" \
    > "$root/$arm.run.log" 2>&1
  if [ "$arm" != MTPMIX_SPLIT_STOCK_CONTROL ]; then
    "$workdir/venv/bin/python" "$driver" --compare "$root" MTPMIX_SPLIT_STOCK_CONTROL "$arm" \
      > "$root/$arm.compare.log" 2>&1
    tail -1 "$root/$arm.compare.log"
  fi
  "$workdir/venv/bin/python" - "$root/server_$arm.log" "$stock" "$phase" "$verify" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
log = path.read_text()
assert "Traceback" not in log and "fallback" not in log.lower()
assert "cuda graph: True" in log and "WIDTH MISMATCH" not in log
assert "EDF_RUNTIME_" not in log
matches = re.findall(r"STOCK_FA3_SERVING path=stock_fa3_calls calls=(\d+) phase=(\d+)", log)
assert bool(matches) == (sys.argv[2] == "1")
if matches:
    assert int(matches[-1][0]) > 0 and all(phase == sys.argv[3] for _, phase in matches)
if sys.argv[4] == "1":
    assert "FUSED_KERNEL_VERIFY PASS" in log and "native_pdl=1" in log
audit = {"stock_calls": int(matches[-1][0]) if matches else 0,
         "phase": int(sys.argv[3]), "same_input_reference": sys.argv[4] == "1",
         "server_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print("STOCK_FA3_SERVING_GATE_PASS", path.name, json.dumps(audit), flush=True)
PY
done
echo "STOCK_FA3_SERVING_GATE_DONE $(date -Is)"
