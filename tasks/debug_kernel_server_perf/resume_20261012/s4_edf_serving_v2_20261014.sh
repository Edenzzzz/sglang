#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
root=$workdir/s4_edf_serving_v2_20261014
gate_root=$workdir/s4_edf_budgetfix_gate_20261014
predecessor_pid=${EDF_GATE_PID:?EDF_GATE_PID must identify the budget-fix gate}
trap 'status=$?; printf "S4_EDF_SERVING_V2_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
echo "WAIT_BUDGETFIX_GATE $(date -Is)"
while kill -0 "$predecessor_pid" 2>/dev/null; do
  state=$(ps -p "$predecessor_pid" -o stat= || true)
  if [[ "$state" == Z* ]]; then break; fi
  if ! command=$(tr '\0' ' ' < "/proc/$predecessor_pid/cmdline"); then break; fi
  [[ "$command" == "bash s4_edf_budgetfix_gate_20261014.sh "* || "$command" == "bash /dev/shm/w/s4_edf_budgetfix_gate_20261014.sh "* ]]
  sleep 15
done
grep -q '^S4_EDF_BUDGETFIX_GATE_DONE ' "$gate_root.log"
grep -q '^S4_EDF_BUDGETFIX_EXIT status=0 ' "$gate_root.log"
test ! -e "$root"
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[r]un_mixed_workloads|[f]used_reference' >/dev/null; then
  echo 'GPU predecessor still running; refusing duplicate launch'
  exit 1
fi
mkdir "$root"
cd "$workdir"
sha256sum -c "$gate_root/candidate.sha256"
sha256sum -c "$gate_root/models.sha256"
source "$workdir/tree_env.sh"
export PYTHONPATH=$workdir/up_edf_budgetfix/python HF_HOME=$workdir/hf MODEL=$workdir/qwen7b
export SGLANG_FUSED_FI_ROOT=$workdir/port07
export SGLANG_FUSED_MERGED_DIR=$workdir/merged_colaunch TORCH_EXTENSIONS_DIR=$workdir/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112 SGLANG_SPLIT_OVERLAP=0
export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
export TRACE_PATH=$workdir/workloads/tracelab/syfi_coding_trace_all.jsonl.gz
export CHAT_DATASET=$workdir/workloads/ShareGPT_V3_unfiltered_cleaned_split.json CODEX_RATE=0.1
export SGLANG_EDF_MAX_PROFILED_PREFILL_PREFIX=262144 SGLANG_EDF_LONG_PREFIX_PROJECTION_THRESHOLD=262145
export TTFT_SLO_MS=2000 ITL_SLO_MS=50 SGLANG_EDF_ACCEPT_LOG_EVERY=20000
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH BENCH_CHUNK SGLANG_FUSED_NATIVE_PDL SGLANG_FUSED_KERNEL_EXEC
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE SGLANG_FUSED_VERIFY_SNAPSHOT
unset SGLANG_FUSED_DISPATCH SGLANG_FUSED_DECODE_GUARD FLASHINFER_FUSED_PLAN_DEBUG SGLANG_STEP_TRACE
unset SGLANG_EDF_SCHEDULING SGLANG_EDF_LATENCY_MODEL SGLANG_EDF_ACCEPT_MODE
unset FUSED_NOSPLIT FUSED_FA3_UNPACKED FUSED_FA3_CLUSTER_PAIR FUSED_PAGED_TMA_BUILD
runner=$workdir/up_racefix/tasks/realistic_workload_analysis/run_mixed_workloads.py
sha256sum "$runner" "$PYTHONPATH/sglang/srt/managers/scheduler.py" \
  "$PYTHONPATH/sglang/srt/managers/edf_scheduler.py" > "$root/sources.sha256"
cp "$gate_root/models.sha256" "$root/"
env | LC_ALL=C sort | grep -E '^(SGLANG_|FUSED_|FLASHINFER_|MTP_|MODEL=|TRACE_PATH=|CHAT_DATASET=|PYTHONPATH=|TTFT_SLO_MS=|ITL_SLO_MS=)' \
  > "$root/environment.txt"
fixture_index=0
for fixture in 'mix32 32768 65536 36 8 4.0 131072' \
               'mix64 65536 131072 24 8 4.0 131072' \
               'mix128 131072 262144 16 4 8.0 262144'; do
  read -r mixture CODEX_MIN_CTX CODEX_MAX_CTX CODEX_N CODEX_GROUPS YARN_FACTOR CONTEXT_LEN <<< "$fixture"
  export CODEX_MIN_CTX CODEX_MAX_CTX CODEX_N CODEX_GROUPS YARN_FACTOR CONTEXT_LEN
  export CHAT_SECONDS=$((CODEX_N * 10))
  for CHAT_RATE in 16 24; do
    export CHAT_RATE
    methods=(MTPMIX_LPT MTPMIX_LPT_EDFR MTPMIX_SPLIT_EDFR MTPMIX_SPLIT)
    if (( fixture_index % 2 )); then methods=(MTPMIX_SPLIT MTPMIX_SPLIT_EDFR MTPMIX_LPT_EDFR MTPMIX_LPT); fi
    fixture_index=$((fixture_index + 1))
    parent=$root/${mixture}_r${CHAT_RATE}
    mkdir "$parent"
    for method in "${methods[@]}"; do
      output=$parent/$method
      mkdir "$output"
      model=$gate_root/models/${method%_EDFR}_step_model.json
      echo "START $mixture r$CHAT_RATE $method $(date -Is)"
      MTP_STEP_MODEL=$model timeout --kill-after=30 3600 "$workdir/venv/bin/python" -u "$runner" \
        --scenario mix_env --methods "$method" --outdir "$output" > "$output/driver.log" 2>&1
      "$workdir/venv/bin/python" - "$output" "$method" "$mixture" "$CHAT_RATE" \
        "$((CHAT_RATE * CHAT_SECONDS))" "$CODEX_N" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

directory = Path(sys.argv[1])
method = sys.argv[2]
log_path = directory / f"server_{method}.log"
log = log_path.read_text()
assert "Traceback" not in log and "fallback" not in log.lower()
assert "cuda graph: True" in log
validation = json.loads((directory / f"{method}_validation.json").read_text())
assert validation.get("completed") and validation.get("mixed_steps", 0) > 0
results = {}
for stream, expected in zip(("chat", "codex"), map(int, sys.argv[5:7])):
    path = directory / f"{method}_{stream}.json"
    data = json.loads(path.read_text())
    assert data["completed"] == expected and len(data["ttfts"]) == expected
    assert not any(data.get("errors", []))
    results[stream] = hashlib.sha256(path.read_bytes()).hexdigest()
counters = {}
for feature in ("QUEUE", "CHUNK", "OBSERVE"):
    matches = re.findall(r"EDF_RUNTIME_" + feature + r" calls=(\d+)", log)
    counters[feature.lower()] = int(matches[-1]) if matches else 0
if method.endswith("_EDFR"):
    assert "EDF piecewise step model active" in log
    assert "EDF_RUNTIME enabled=1 model=True" in log
    assert all(value > 0 for value in counters.values()), counters
    budgets = [tuple(map(int, match)) for match in re.findall(
        r"EDF_RUNTIME_CHUNK calls=\d+ requested=(\d+) admitted=(\d+) prefill=(\d+) reserved=(\d+)", log)]
    assert budgets
    for requested, admitted, prefill, reserved in budgets:
        assert 0 <= admitted <= requested
        assert admitted == (prefill + reserved if prefill > 0 else 0)
else:
    assert not any(counters.values()), counters
split = re.findall(r"FUSED_SPLIT_CALLS (\d+).*overlapped=(\d+)", log)
if "SPLIT" in method:
    assert split and int(split[-1][0]) > 0 and int(split[-1][1]) == 0
else:
    assert not split
audit = {"method": method, "mixture": sys.argv[3], "chat_rate": int(sys.argv[4]),
         "mixed_steps": validation["mixed_steps"], "edf_counters": counters,
         "split_calls": int(split[-1][0]) if split else 0,
         "tracebacks": 0, "fallbacks": 0, "result_sha256": results,
         "server_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest()}
(directory / "execution_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit), flush=True)
PY
      for stream in chat codex; do cp "$output/${method}_${stream}.json" "$parent/"; done
      echo "END $mixture r$CHAT_RATE $method $(date -Is)"
    done
  done
done
"$workdir/venv/bin/python" "$workdir/drivers/serving/eval_curves.py" --root "$root" \
  --slo '{"chat":[2000,50],"codex":[10000,80]}' --csv "$root/s4_edf_serving.csv"
echo "S4_EDF_SERVING_V2_DONE $(date -Is)"
