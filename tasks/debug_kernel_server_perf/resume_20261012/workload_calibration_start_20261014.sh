#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
root=$workdir/workload_matrix_20261014
overlay=$workdir/workload_protocol_overlay_20261014
trap 'status=$?; printf "WORKLOAD_CALIBRATION_START_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[r]un_mixed_workloads|[w]orkload_driver_20261014.py' >/dev/null; then
  echo 'GPU predecessor still running; refusing duplicate launch'
  exit 1
fi
cd "$workdir"
sha256sum -c "$overlay/baseline.sha256"
test ! -e up_workload_matrix
mkdir up_workload_matrix
cp -a up_edf_budgetfix/python up_workload_matrix/python
cp "$overlay/serving.py" up_workload_matrix/python/sglang/benchmark/serving.py
cp "$overlay/workload_protocol.py" up_workload_matrix/python/sglang/benchmark/workload_protocol.py
cp "$overlay/driver.py" drivers/workload_driver_20261014.py
sha256sum -c "$overlay/candidate.sha256"
mkdir -p "$root/references"
cp "$overlay/catalog.json" "$root/catalog.json"
cp "$overlay/"{baseline,candidate}.sha256 "$root/"
source "$workdir/tree_env.sh"
export PYTHONPATH=$workdir/up_workload_matrix/python HF_HOME=$workdir/hf MODEL=$workdir/qwen7b
export SGLANG_FUSED_FI_ROOT=$workdir/port07 SGLANG_FUSED_MERGED_DIR=$workdir/merged_colaunch
export TORCH_EXTENSIONS_DIR=$workdir/torch_ext_colaunch_auto2
export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
export CHAT_DATASET=$workdir/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export TRACE_PATH=$workdir/workloads/tracelab/syfi_full_min.jsonl.gz
export CONTEXT_LEN=131072 YARN_FACTOR=4.0 STANDALONE=sharegpt STANDALONE_N=128 STANDALONE_RATE=inf
export BENCH_MAX_CONCURRENCY=1 HF_DATASETS_OFFLINE=1
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH SGLANG_STEP_TRACE BENCH_CHUNK STREAMS
unset SGLANG_EDF_SCHEDULING SGLANG_EDF_LATENCY_MODEL SGLANG_EDF_ACCEPT_MODE
unset SGLANG_FUSED_NATIVE_PDL SGLANG_FUSED_KERNEL_EXEC SGLANG_FUSED_DISPATCH SGLANG_FUSED_DECODE_GUARD
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE SGLANG_FUSED_VERIFY_SNAPSHOT
unset SGLANG_TWO_EXEC SGLANG_MIXED_ATTENTION_KERNEL SGLANG_ALLOW_MIXED_CHUNK_SPEC SGLANG_MIXED_SPEC
output=$root/references/sharegpt_ctx128
test ! -e "$output"
mkdir "$output"
echo "REFERENCE_START sharegpt_ctx128 $(date -Is)"
timeout --kill-after=30 7200 "$workdir/venv/bin/python" -u "$workdir/drivers/workload_driver_20261014.py" \
  --scenario standalone_env --methods MTP_LPT --outdir "$output" > "$output/driver.log" 2>&1
"$workdir/venv/bin/python" - "$output" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from sglang.benchmark.workload_protocol import validate_reference

root = Path(sys.argv[1])
path = root / "MTP_LPT_main.json"
data = json.loads(path.read_text())
validate_reference(data)
assert data["completed"] == 128
validation = json.loads((root / "MTP_LPT_validation.json").read_text())
assert validation["completed"] and validation["mixed_steps"] == 0
assert validation["fallbacks"] == validation["tracebacks"] == 0
audit = {"completed": True, "requests": 128, "client_concurrency": 1,
         "reference_matching": "per-request fingerprint", "output": path.name,
         "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
(root / "reference_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print("REFERENCE_PASS", json.dumps(audit), flush=True)
PY
echo "WORKLOAD_CALIBRATION_START_DONE $(date -Is)"
