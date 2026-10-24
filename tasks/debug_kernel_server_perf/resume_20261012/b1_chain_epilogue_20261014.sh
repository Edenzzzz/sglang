#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
overlay=$workdir/${B1_CHAIN_OVERLAY:-b1_chain_epilogue_overlay_20261014}
root=$workdir/${B1_CHAIN_RUN_NAME:-b1_chain_epilogue_20261014}
tree=${B1_CHAIN_TREE:-up_b1_chain_epilogue}
trap 'status=$?; printf "B1_CHAIN_EPILOGUE_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[w]orkload_driver|[r]un_mixed_workloads' >/dev/null; then
  echo 'GPU predecessor still running; refusing duplicate launch'
  exit 1
fi
cd "$workdir"
sha256sum -c "$overlay/baseline.sha256"
test ! -e "$root"
mkdir "$root"
if [ ! -d "$tree" ]; then
  mkdir "$tree"
  cp -a up_racefix/python "$tree/python"
  cp "$overlay/eagle_worker_common.py" "$tree/python/sglang/srt/speculative/eagle_worker_common.py"
  cp "$overlay/fused_chain_tree.py" "$tree/python/sglang/srt/speculative/fused_chain_tree.py"
fi
sha256sum -c "$overlay/candidate.sha256"
cp "$overlay/"{baseline,candidate}.sha256 "$root/"
source "$workdir/tree_env.sh"
export PYTHONPATH=$workdir/$tree/python HF_HOME=$workdir/hf MODEL=$workdir/qwen7b
export SGLANG_FUSED_FI_ROOT=$workdir/port07
export SGLANG_FUSED_MERGED_DIR=$workdir/merged_colaunch TORCH_EXTENSIONS_DIR=$workdir/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112 SGLANG_SPLIT_OVERLAP=0
export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
export CHAT_DATASET=$workdir/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export CHAIN_TREE_SOURCE=$workdir/$tree/python/sglang/srt/speculative/fused_chain_tree.py
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH SGLANG_STEP_TRACE
unset SGLANG_EDF_SCHEDULING SGLANG_EDF_LATENCY_MODEL SGLANG_EDF_ACCEPT_MODE
unset SGLANG_FUSED_NATIVE_PDL SGLANG_FUSED_KERNEL_EXEC SGLANG_FUSED_DISPATCH SGLANG_FUSED_DECODE_GUARD
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE SGLANG_FUSED_VERIFY_SNAPSHOT SGLANG_FUSED_CHAIN_TREE
unset SGLANG_TWO_EXEC SGLANG_MIXED_ATTENTION_KERNEL SGLANG_ALLOW_MIXED_CHUNK_SPEC SGLANG_MIXED_SPEC
echo "B1_CHAIN_TEST_START $(date -Is)"
"$workdir/venv/bin/python" -u "$overlay/test_fused_chain_tree.py" > "$root/tests.log" 2>&1
echo "B1_CHAIN_TEST_PASS $(date -Is)"
"$workdir/venv/bin/python" -u "$overlay/bench_chain_tree_epilogue.py" --output "$root/timing.json" > "$root/timing.log" 2>&1
echo "B1_CHAIN_TIMING_DONE $(date -Is)"
gate=$root/q4_gate
mkdir "$gate"
driver=$workdir/drivers/srv_correctness_edf_20261014.py
for arm in MTPMIX_SPLIT_B1_BASE MTPMIX_SPLIT_B1_CHAIN MTPMIX_SPLIT_B1_BOTH MTPMIXFUSED_B1_BOTH; do
  chain=0
  verify=0
  if [[ "$arm" == *CHAIN || "$arm" == *BOTH ]]; then chain=1; fi
  if [[ "$arm" == *BOTH ]]; then verify=1; fi
  echo "B1_CHAIN_GATE_START $arm $(date -Is)"
  CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET="$CHAT_DATASET" BENCH_CHUNK=512 \
    CORRECTNESS_DECODE_TOKENS=1200 SGLANG_FUSED_DECODE_GUARD=0 \
    SGLANG_FUSED_CHAIN_TREE=$chain SGLANG_FUSED_VERIFY_EPILOGUE=$verify \
    timeout --kill-after=30 1200 "$workdir/venv/bin/python" -u "$driver" "$arm" "$gate" \
    > "$gate/$arm.run.log" 2>&1
  if [ "$arm" != MTPMIX_SPLIT_B1_BASE ]; then
    "$workdir/venv/bin/python" -u "$driver" --compare "$gate" MTPMIX_SPLIT_B1_BASE "$arm" \
      > "$gate/$arm.compare.log" 2>&1
    tail -1 "$gate/$arm.compare.log"
  fi
  "$workdir/venv/bin/python" - "$gate/server_$arm.log" "$chain" "$verify" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
log = path.read_text()
assert "Traceback" not in log and "fallback" not in log.lower()
assert "cuda graph: True" in log and "WIDTH MISMATCH" not in log
counts = re.findall(r"FUSED_CHAIN_TREE calls=(\d+)", log)
calls = int(counts[-1]) if counts else 0
assert (calls > 0) == (sys.argv[2] == "1")
assert ("FUSED_VERIFY_EPILOGUE active" in log) == (sys.argv[3] == "1")
audit = {"chain_calls": calls, "verify_enabled": sys.argv[3] == "1",
         "server_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print("B1_CHAIN_GATE_AUDIT", path.name, json.dumps(audit), flush=True)
PY
done
echo "B1_CHAIN_EPILOGUE_DONE $(date -Is)"
