#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
root=$workdir/s4_edf_budgetfix_gate_20261014
previous=$workdir/s4_edf_piecewise_gate_20261014
trap 'status=$?; printf "S4_EDF_BUDGETFIX_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
grep -q '^S4_EDF_SERVING_EXIT status=1 ' "$workdir/s4_edf_serving_20261014.log"
test ! -e "$root"
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[r]un_mixed_workloads|[f]used_reference' >/dev/null; then
  echo 'GPU predecessor still running; refusing duplicate launch'
  exit 1
fi
cd "$workdir"
sha256sum -c edf_budgetfix_overlay_20261014/baseline.sha256
sha256sum -c "$previous/models.sha256"
test ! -e up_edf_budgetfix
mkdir "$root" up_edf_budgetfix
cp -a up_edf_piecewise/python up_edf_budgetfix/python
cp edf_budgetfix_overlay_20261014/python/sglang/srt/managers/scheduler.py \
  up_edf_budgetfix/python/sglang/srt/managers/scheduler.py
sha256sum -c edf_budgetfix_overlay_20261014/candidate.sha256
cp edf_budgetfix_overlay_20261014/{baseline,candidate}.sha256 "$root/"
cp -a "$previous/models" "$root/models"
sha256sum "$root"/models/*.json > "$root/models.sha256"
source "$workdir/tree_env.sh"
export PYTHONPATH=$workdir/up_edf_budgetfix/python HF_HOME=$workdir/hf MODEL=$workdir/qwen7b
export SGLANG_FUSED_FI_ROOT=$workdir/port07
export SGLANG_FUSED_MERGED_DIR=$workdir/merged_colaunch TORCH_EXTENSIONS_DIR=$workdir/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112 SGLANG_SPLIT_OVERLAP=0
export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
export CHAT_DATASET=$workdir/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export SGLANG_EDF_MAX_PROFILED_PREFILL_PREFIX=262144 SGLANG_EDF_LONG_PREFIX_PROJECTION_THRESHOLD=262145
export SGLANG_TTFT_SLO_MS=2000 SGLANG_ITL_SLO_MS=50 SGLANG_EDF_PLAN_OVERHEAD_MS=0
export SGLANG_EDF_ACCEPT_MODE=req SGLANG_EDF_ACCEPT_LOG_EVERY=1000
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH SGLANG_FUSED_NATIVE_PDL SGLANG_FUSED_KERNEL_EXEC
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE SGLANG_FUSED_VERIFY_SNAPSHOT
unset SGLANG_FUSED_DISPATCH SGLANG_FUSED_DECODE_GUARD FLASHINFER_FUSED_PLAN_DEBUG SGLANG_STEP_TRACE
unset FUSED_NOSPLIT FUSED_FA3_UNPACKED FUSED_FA3_CLUSTER_PAIR FUSED_PAGED_TMA_BUILD
driver=$workdir/drivers/srv_correctness_edf_20261014.py
gate=$root/q4_gate
mkdir "$gate"
for method in MTP_LPT MTPMIX_LPT_EDFR MTPMIX_SPLIT_EDFR; do
  arm=${method}_BUDGETFIX_Q4G
  enabled=0
  model=
  if [[ "$method" == *_EDFR ]]; then enabled=1; model=linear:$root/models/${method%_EDFR}_step_model.json; fi
  echo "GATE_START $arm $(date -Is)"
  CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET="$CHAT_DATASET" BENCH_CHUNK=512 \
    CORRECTNESS_DECODE_TOKENS=1200 SGLANG_FUSED_DECODE_GUARD=0 \
    SGLANG_EDF_SCHEDULING=$enabled SGLANG_EDF_LATENCY_MODEL=$model \
    timeout --kill-after=30 1200 "$workdir/venv/bin/python" -u "$driver" "$arm" "$gate" \
    > "$gate/$arm.run.log" 2>&1
  if [ "$method" != MTP_LPT ]; then
    "$workdir/venv/bin/python" -u "$driver" --compare "$gate" MTP_LPT_BUDGETFIX_Q4G "$arm" \
      > "$gate/$arm.compare.log" 2>&1
    tail -1 "$gate/$arm.compare.log"
  fi
  "$workdir/venv/bin/python" - "$gate/server_$arm.log" "$enabled" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
log = path.read_text()
assert "Traceback" not in log and "fallback" not in log.lower()
assert "WIDTH MISMATCH" not in log and "cuda graph: True" in log
counters = {}
for feature in ("QUEUE", "CHUNK", "OBSERVE"):
    matches = re.findall(r"EDF_RUNTIME_" + feature + r" calls=(\d+)", log)
    counters[feature.lower()] = int(matches[-1]) if matches else 0
budgets = [tuple(map(int, match)) for match in re.findall(
    r"EDF_RUNTIME_CHUNK calls=\d+ requested=(\d+) admitted=(\d+) prefill=(\d+) reserved=(\d+)", log)]
if sys.argv[2] == "1":
    assert "EDF piecewise step model active" in log and all(counters.values())
    assert budgets
    for requested, admitted, prefill, reserved in budgets:
        assert 0 <= admitted <= requested
        assert admitted == (prefill + reserved if prefill > 0 else 0)
else:
    assert not any(counters.values())
audit = {"edf": sys.argv[2] == "1", **counters, "budgets": budgets,
         "server_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print("EDF_BUDGET_GATE", path.name, json.dumps(audit), flush=True)
PY
done
echo "S4_EDF_BUDGETFIX_GATE_DONE $(date -Is)"
