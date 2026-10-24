#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
root=$workdir/b1_kernel_concurrency_20261014
overlay=$workdir/b1_kernel_concurrency_overlay_20261014
trap 'status=$?; printf "B1_KERNEL_CONCURRENCY_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[w]orkload_driver|[r]un_mixed_workloads|[k]ernel_concurrency_driver.py' >/dev/null; then
  echo 'GPU predecessor still running; refusing duplicate launch'
  exit 1
fi
cd "$workdir"
grep -q '^B1_CHAIN_SHADOW_DONE ' b1_chain_shadow_20261014.log
sha256sum -c "$overlay/baseline.sha256"
test ! -e up_b1_kernel_serving
test ! -e "$root"
mkdir up_b1_kernel_serving "$root"
cp -a up_b1_chain_epilogue_v4/python up_b1_kernel_serving/python
cp "$overlay/eagle_worker_common.py" up_b1_kernel_serving/python/sglang/srt/speculative/eagle_worker_common.py
cp "$overlay/serving.py" up_b1_kernel_serving/python/sglang/benchmark/serving.py
cp "$overlay/session_replay.py" up_b1_kernel_serving/python/sglang/benchmark/session_replay.py
cp "$overlay/session_concurrency.py" up_b1_kernel_serving/python/sglang/benchmark/session_concurrency.py
cp "$overlay/workload_protocol.py" up_b1_kernel_serving/python/sglang/benchmark/workload_protocol.py
cp "$overlay/tracelab.py" up_b1_kernel_serving/python/sglang/benchmark/datasets/tracelab.py
cp "$overlay/driver.py" drivers/kernel_concurrency_driver.py
sha256sum -c "$overlay/candidate.sha256"
cp "$overlay/"{baseline,candidate}.sha256 "$root/"
source "$workdir/tree_env.sh"
export PYTHONPATH=$workdir/up_b1_kernel_serving/python HF_HOME=$workdir/hf MODEL=$workdir/qwen7b
export SGLANG_FUSED_FI_ROOT=$workdir/port07
export SGLANG_FUSED_MERGED_DIR=$workdir/merged_colaunch TORCH_EXTENSIONS_DIR=$workdir/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112 SGLANG_SPLIT_OVERLAP=0
export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
export CHAT_DATASET=$workdir/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export TRACE_PATH=$workdir/workloads/tracelab/syfi_full_min.jsonl.gz
export AGENT_N=96 AGENT_GROUPS=48 AGENT_SEED=20261014 OTHER_N=2048 OTHER_RATE=8
export BENCH_TAG=kernel_concurrency_screen_no_edf
export BENCH_WARMUP_CHAIN=1
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH SGLANG_STEP_TRACE
unset SGLANG_EDF_SCHEDULING SGLANG_EDF_LATENCY_MODEL SGLANG_EDF_ACCEPT_MODE
unset SGLANG_FUSED_NATIVE_PDL SGLANG_FUSED_KERNEL_EXEC SGLANG_FUSED_DISPATCH SGLANG_FUSED_DECODE_GUARD
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_SNAPSHOT
unset SGLANG_TWO_EXEC SGLANG_MIXED_ATTENTION_KERNEL SGLANG_ALLOW_MIXED_CHUNK_SPEC SGLANG_MIXED_SPEC
unset BENCH_MAX_CONCURRENCY BENCH_NO_BARRIER BENCH_CHUNK SERVER_EXTRA_ARGS STREAMS
"$workdir/venv/bin/python" - "$TRACE_PATH" "$root" <<'PY'
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

source = Path(sys.argv[1])
records = [json.loads(line) for line in gzip.open(source, "rt")]
reports = {}
for name, lower, upper, window in (
    ("mix32", 32768, 65536, 131072),
    ("mix64", 65536, 131072, 131072),
    ("mix128", 131072, 262144, 262144),
):
    sessions = defaultdict(list)
    for record in records:
        context = int(record.get("input_tokens_total") or 0)
        output = int(record.get("output_tokens") or 0)
        prefix = int(record.get("prefix_tokens") or 0)
        append = int(record.get("newly_append_tokens") or 0)
        if (record.get("provider") in ("claude", "codex") and lower <= context <= upper
                and output >= 1 and prefix >= 0 and append > 0 and context + output <= window):
            assert context == prefix + append
            sessions[record["session_id"]].append(record)
    eligible = [session for session, entries in sessions.items() if len(entries) >= 2]
    assert len(eligible) >= 48, (name, len(eligible))
    selector = random.Random(20261014)
    selected = sorted(eligible, key=lambda session: selector.random() ** (1 / len(sessions[session])), reverse=True)[:48]
    chosen = []
    for session in selected:
        entries = sorted(sessions[session], key=lambda record: record["round_index"])
        offset = selector.randint(0, len(entries) - 2)
        chosen.extend(entries[offset:offset + 2])
    reports[name] = {
        "eligible_sessions": len(eligible), "programs": 48, "rounds": len(chosen),
        "min_context": min(record["input_tokens_total"] for record in chosen),
        "max_context": max(record["input_tokens_total"] for record in chosen),
        "output_tokens": sum(record["output_tokens"] for record in chosen),
        "records_with_timing": sum(bool(record.get("timing_events")) for record in chosen),
        "round_manifest": [(record["session_id"], record["round_index"]) for record in chosen],
    }
assert all(report["records_with_timing"] == 0 for report in reports.values())
result = {"trace_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "scope": "finite gap-free length/session kernel stress; not timestamp-faithful or steady-state capacity",
          "ranges": reports}
(Path(sys.argv[2]) / "workload_preflight.json").write_text(json.dumps(result, indent=2) + "\n")
print("B1_KERNEL_PREFLIGHT_PASS", json.dumps({name: {key: value for key, value in report.items() if key != "round_manifest"} for name, report in reports.items()}), flush=True)
PY
for range_spec in 'mix32 32768 65536 131072 4.0' 'mix64 65536 131072 131072 4.0' 'mix128 131072 262144 262144 8.0'; do
  set -- $range_spec
  range_name=$1
  export AGENT_MIN_CTX=$2 AGENT_MAX_CTX=$3 CONTEXT_LEN=$4 YARN_FACTOR=$5
  export BENCH_TRACE_MAX_TOTAL_TOKENS=$CONTEXT_LEN
  for concurrency in 24 48; do
    export BENCH_SESSION_CONCURRENCY=$concurrency
    point=$root/${range_name}_c${concurrency}
    mkdir "$point"
    reference=$point/BASE_1
    for variant in BASE_1 CHAIN BOTH BASE_2; do
      export SGLANG_FUSED_CHAIN_TREE=0 SGLANG_FUSED_VERIFY_EPILOGUE=0
      if [ "$variant" = CHAIN ] || [ "$variant" = BOTH ]; then export SGLANG_FUSED_CHAIN_TREE=1; fi
      if [ "$variant" = BOTH ]; then export SGLANG_FUSED_VERIFY_EPILOGUE=1; fi
      output=$point/$variant
      mkdir "$output"
      echo "B1_KERNEL_ARM_START $range_name concurrency=$concurrency variant=$variant $(date -Is)"
      timeout --kill-after=30 3600 "$workdir/venv/bin/python" -u drivers/kernel_concurrency_driver.py \
        --scenario kernel_concurrency_env --methods MTPMIX_SPLIT --outdir "$output" > "$output/driver.log" 2>&1
      compare=()
      if [ "$variant" != BASE_1 ]; then compare=(--reference "$reference"); fi
      "$workdir/venv/bin/python" "$overlay/audit_kernel_concurrency.py" "$output" \
        --concurrency "$concurrency" --chain "$SGLANG_FUSED_CHAIN_TREE" --verify "$SGLANG_FUSED_VERIFY_EPILOGUE" \
        "${compare[@]}" > "$output/audit.log" 2>&1
      echo "B1_KERNEL_ARM_DONE $range_name concurrency=$concurrency variant=$variant $(date -Is)"
    done
  done
done
echo "B1_KERNEL_CONCURRENCY_DONE $(date -Is)"
