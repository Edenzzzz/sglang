#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
output_dir=$workdir/fa2_auto4096_realcheck_20261013
predecessor_pid=430924
predecessor_log=$workdir/fa2_auto4096_serving_20261013.log
trap 'status=$?; printf "FA2_REALCHECK_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
echo "WAIT_CAP_SERVING $(date -Is)"
while kill -0 "$predecessor_pid" 2>/dev/null; do
  predecessor_state=$(ps -p "$predecessor_pid" -o stat= || true)
  if [[ "$predecessor_state" == Z* ]]; then break; fi
  if ! predecessor_command=$(tr '\0' ' ' < "/proc/$predecessor_pid/cmdline"); then break; fi
  [[ "$predecessor_command" == "bash /dev/shm/w/fa2_auto4096_serving_20261013.sh "* ]]
  sleep 15
done
grep -q '^FA2_AUTO4096_SERVING_DONE ' "$predecessor_log"
grep -q '^FA2_AUTO4096_SERVING_EXIT status=0 ' "$predecessor_log"
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running; refusing duplicate launch"
  exit 1
fi
mkdir "$output_dir"
source "$workdir/tree_env.sh"
sha256sum -c "$workdir/fi_auto4096_header.sha256"
export HF_HOME=$workdir/hf MODEL=$workdir/qwen7b
export SGLANG_FUSED_FI_ROOT=$workdir/fi_auto4096 SGLANG_FUSED_MERGED_DIR=$workdir/merged_auto4096
export TORCH_EXTENSIONS_DIR=$workdir/torch_ext_auto4096
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export SGLANG_FUSED_SM_FA3=112 SGLANG_FUSED_SM_MARGIN=0 SGLANG_SPLIT_OVERLAP=0
export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3 BENCH_CHUNK=512 SGLANG_FUSED_DECODE_GUARD=0
export CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DECODE_TOKENS=1200
export CORRECTNESS_DATASET=$workdir/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export SGLANG_FUSED_SPLIT_VERIFY=1 SGLANG_FUSED_VERIFY_SNAPSHOT=1
unset SGLANG_FUSED_DISPATCH SGLANG_FUSED_VERIFY_EPILOGUE SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH
unset FUSED_KV_SHARE_BUILD FUSED_CLUSTER FLASHINFER_FUSED_CLUSTER FUSED_FA3_CLUSTER_PAIR
unset FUSED_FA3_KV_SHARE FUSED_FA3_KV_SHARE_TRACE FLASHINFER_FUSED_PLAN_DEBUG
cp "$workdir/q4_gate_graphs_20261012/MTP_LPT_Q4G.json" "$output_dir/"
driver=$workdir/$TREE/tasks/fused_specdec_workload/drivers/srv_correctness.py
for policy in auto auto4096; do
  arm=MTPMIXFUSED_${policy}_REAL_Q4G
  export FUSED_FA2_KV_CHUNK=$policy SGLANG_FUSED_FAILURE_DUMP=$output_dir/failure_$policy.pt
  echo "START $policy $(date -Is)"
  "$workdir/venv/bin/python" -u "$driver" "$arm" "$output_dir" > "$output_dir/$arm.run.log" 2>&1
  "$workdir/venv/bin/python" -u "$driver" --compare "$output_dir" MTP_LPT_Q4G "$arm" \
    > "$output_dir/$arm.compare.log" 2>&1
  "$workdir/venv/bin/python" - "$output_dir" "$arm" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
arm = sys.argv[2]
logs = list(root.glob(f"*{arm}*.log"))
records = []
for path in logs:
    text = path.read_text()
    records.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert "Traceback" not in text, path
combined = "\n".join(path.read_text() for path in logs)
checks = re.findall(r"FUSED_KERNEL_VERIFY PASS worst_head=([0-9.]+)", combined)
assert checks, "real-input attention verification did not execute"
assert "FUSED_KERNEL_FAILURE_DUMP" not in combined, "real-input numerical failure"
audit = {"arm": arm, "layer0_checks": len(checks), "layer0_worst": max(map(float, checks)), "logs": records}
(root / f"{arm}.audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit), flush=True)
PY
  tail -1 "$output_dir/$arm.compare.log"
  echo "END $policy $(date -Is)"
done
echo "FA2_REALCHECK_DONE $(date -Is)"
