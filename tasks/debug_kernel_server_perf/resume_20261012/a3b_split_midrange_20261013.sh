#!/bin/bash
set -euo pipefail
W=/dev/shm/w
ROOT=$W/a3b_split_midrange_20261013
trap 'status=$?; printf "A3B_SPLIT_MIDRANGE_RUN_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
predecessor_pid=388565
predecessor_log=$W/b3_append_resume_20261013.log
echo "WAIT_B3_APPEND $(date -Is)"
while kill -0 "$predecessor_pid" 2>/dev/null; do
  predecessor_state=$(ps -p "$predecessor_pid" -o stat= || true)
  if [[ "$predecessor_state" == Z* ]]; then break; fi
  if ! predecessor_command=$(tr '\0' ' ' < "/proc/$predecessor_pid/cmdline"); then break; fi
  [[ "$predecessor_command" == "bash /dev/shm/w/b3_append_resume_20261013.sh "* ]]
  sleep 15
done
grep -q '^B3_APPEND_RESUME_DONE ' "$predecessor_log"
grep -q '^B3_APPEND_RESUME_EXIT status=0 ' "$predecessor_log"
test ! -e "$ROOT"
exec 9>"$W/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running; refusing duplicate launch"
  exit 1
fi
mkdir -p "$ROOT"
grep -q '^PASS' "$W/racefix_verdict"
source "$W/tree_env.sh"
export HF_HOME=$W/hf MODEL=$W/qwen7b
export SGLANG_FUSED_FI_ROOT=$W/port07
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch
export TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1
export FUSED_CONST_PERS_PARAMS=1 FUSED_WORK_BUDGET_SPLIT=1
export FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112
export SGLANG_SPLIT_OVERLAP=0 SGLANG_FUSED_SM_MARGIN=0
export SGLANG_A3_WORK_MAX_QO=32 MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
export TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace_all.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
unset SGLANG_FUSED_KERNEL_EXEC SGLANG_FUSED_DISPATCH SGLANG_FUSED_DECODE_GUARD
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE FLASHINFER_FUSED_PLAN_DEBUG
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH BENCH_CHUNK STREAMS
test -s "$TRACE_PATH"
test -s "$CHAT_DATASET"
sha256sum "$W/$TREE/python/sglang/srt/layers/attention/mixed_dispatch.py" \
  "$W/$TREE/python/sglang/srt/layers/attention/flashinfer_backend.py" \
  "$W/$TREE/tasks/realistic_workload_analysis/run_mixed_workloads.py" \
  "$W/$TREE/tasks/fused_specdec_workload/drivers/srv_correctness.py" > "$ROOT/sources.sha256"
env | LC_ALL=C sort | grep -E '^(SGLANG_FUSED_|SGLANG_SPLIT_|SGLANG_A3_|FUSED_|FLASHINFER_|MTP_|MODEL=|TRACE_PATH=|CHAT_DATASET=|PYTHONPATH=)' \
  > "$ROOT/environment.txt"

GATE=$ROOT/q4_gate
mkdir -p "$GATE"
cp "$W/q4_gate_graphs_20261012/MTP_LPT_Q4G.json" "$GATE/"
DRIVER=$W/$TREE/tasks/fused_specdec_workload/drivers/srv_correctness.py
for rule in DEFAULT A3B; do
  routing=""
  if [ "$rule" = A3B ]; then routing=a3; fi
  arm=MTPMIX_SPLIT_${rule}_Q4G
  echo "GATE_START $arm $(date -Is)"
  CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET="$CHAT_DATASET" BENCH_CHUNK=512 \
    CORRECTNESS_DECODE_TOKENS=1200 SGLANG_FUSED_DECODE_GUARD=0 SGLANG_FUSED_DISPATCH=$routing \
    "$W/venv/bin/python" -u "$DRIVER" "$arm" "$GATE" > "$GATE/$arm.run.log" 2>&1
  "$W/venv/bin/python" -u "$DRIVER" --compare "$GATE" MTP_LPT_Q4G "$arm" \
    > "$GATE/$arm.compare.log" 2>&1
  tail -1 "$GATE/$arm.compare.log"
done
echo "A3B_SPLIT_GATE_DONE $(date -Is)"

cd "$W/$TREE/tasks/realistic_workload_analysis"
for mixture in "mix64 65536 131072 24 8 4.0 131072"; do
  read -r name CODEX_MIN_CTX CODEX_MAX_CTX CODEX_N CODEX_GROUPS YARN_FACTOR CONTEXT_LEN <<< "$mixture"
  export CODEX_MIN_CTX CODEX_MAX_CTX CODEX_N CODEX_GROUPS YARN_FACTOR CONTEXT_LEN
  export CHAT_RATE=24 CODEX_RATE=0.1 CHAT_SECONDS=$((CODEX_N * 10))
  for label in DEFAULT_1 A3B_1 DEFAULT_2 A3B_2; do
    OUT=$ROOT/${name}_r24/$label
    test ! -e "$OUT/MTPMIX_SPLIT_validation.json"
    mkdir -p "$OUT"
    routing=""
    if [[ "$label" == A3B_* ]]; then routing=a3; fi
    echo "START $name $label $(date -Is)"
    SGLANG_FUSED_DISPATCH=$routing timeout 2400 "$W/venv/bin/python" -u run_mixed_workloads.py \
      --scenario mix_env --methods MTPMIX_SPLIT --outdir "$OUT" > "$OUT/driver.log" 2>&1
    "$W/venv/bin/python" - "$OUT" "$name" "$label" "$((CHAT_RATE * CHAT_SECONDS))" "$CODEX_N" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

directory = Path(sys.argv[1])
log_path = directory / "server_MTPMIX_SPLIT.log"
log = log_path.read_text()
matches = re.findall(r"FUSED_SPLIT_CALLS (\d+).*overlapped=(\d+)", log)
assert matches, "no split calls"
calls, overlapped = map(int, matches[-1])
assert calls > 0 and overlapped == 0, (calls, overlapped)
assert "cuda graph: True" in log, "no graph decode observed"
assert "Traceback" not in log and "fallback" not in log.lower(), "runtime errors"
validation = json.loads((directory / "MTPMIX_SPLIT_validation.json").read_text())
assert validation.get("completed") and validation.get("mixed_steps", 0) > 0, validation
for stream, expected in zip(("chat", "codex"), map(int, sys.argv[4:6])):
    result = json.loads((directory / f"MTPMIX_SPLIT_{stream}.json").read_text())
    assert result["completed"] == expected and len(result["ttfts"]) == expected
    assert not any(result.get("errors", [])), stream
summary = {
    "mixture": sys.argv[2], "arm": sys.argv[3], "split_calls": calls,
    "overlapped": overlapped, "mixed_steps": validation["mixed_steps"],
    "graph_decode_lines": log.count("cuda graph: True"),
    "fallbacks": log.lower().count("fallback"), "tracebacks": log.count("Traceback"),
    "plan_counters": re.findall(r"FUSED_SPLIT_PLAN ok .*", log)[-1:],
    "one_sided_counters": re.findall(r"FUSED_SPLIT_ONESIDED .*", log)[-1:],
    "server_log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
}
(directory / "execution_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
PY
    for stream in chat codex; do
      cp "$OUT/MTPMIX_SPLIT_${stream}.json" "$ROOT/${name}_r24/${label}_${stream}.json"
    done
    echo "END $name $label $(date -Is)"
  done
done
"$W/venv/bin/python" "$W/drivers/serving/eval_curves.py" --root "$ROOT" \
  --slo '{"chat":[2000,50],"codex":[10000,80]}' --csv "$ROOT/a3b_split.csv"
echo "A3B_SPLIT_MIDRANGE_ALL_DONE $(date -Is)"

