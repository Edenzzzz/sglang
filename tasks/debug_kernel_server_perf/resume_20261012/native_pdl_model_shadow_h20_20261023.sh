#!/bin/bash
set -euo pipefail
ROOT=${ROOT:-/dev/shm/w/native_pdl_model_shadow_20261023}
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test ! -d "$ROOT/python"
cp -a /dev/shm/w/up/python "$ROOT/python"
/dev/shm/w/venv/bin/python "$ROOT/apply_native_pdl_fa3_backend_20261023.py" \
  "$ROOT/python/sglang/srt/layers/attention/flashattention_backend.py"
/dev/shm/w/venv/bin/python "$ROOT/apply_native_pdl_model_shadow_20261023.py" \
  "$ROOT/python/sglang/srt/speculative/eagle_worker_v2.py"
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export PYTHONPATH=$ROOT/python:$ROOT:/dev/shm/w/fi
export HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_fa3_pdl_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_q32_20261023/native_fa2_q32/native_fa2_q32.so
export SGLANG_NATIVE_PDL=1 SGLANG_NATIVE_PDL_VERIFY=1
export SGLANG_MIXED_SPEC_MIN_CTX=8192 SGLANG_MIXED_SPEC_MIN_QUEUE=4
export NATIVE_PDL_MODEL_SHADOW=$ROOT/shadow.jsonl
export SRV_ARGS="--attention-backend fa3 --chunked-prefill-size 512 --disable-cuda-graph"
cd "$ROOT"
/dev/shm/w/venv/bin/python -u srv_correctness.py MTP_MIX_NATIVE_PDL_SHADOW "$ROOT/results"
/dev/shm/w/venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

rows = [json.loads(line) for line in Path(os.environ["NATIVE_PDL_MODEL_SHADOW"]).read_text().splitlines()]
assert rows and all(
    row["finite"] and row["baseline_repeat_exact"] and row["native_serial_exact"]
    and row["committed_sample_exact"] and row["committed_prefill_exact"] for row in rows
)
accepted = all(
    row["verify_requests_with_changed_emission"] == 0
    and row["prefill_requests_with_changed_emission"] == 0 for row in rows
)
print("NATIVE_PDL_MODEL_SHADOW_REPORT_COMPLETE", json.dumps(dict(
    forwards=len(rows), logit_rows=sum(row["rows"] for row in rows),
    argmax_flips=sum(row["argmax_flips"] for row in rows),
    prefill_argmax_flips=sum(row["prefill_argmax_flips"] for row in rows),
    worst_relative=max(row["worst_relative"] for row in rows),
    forwards_over_relative_0p02=sum(not row["model_relative_within_0p02"] for row in rows),
    verify_requests_with_changed_emission=sum(row["verify_requests_with_changed_emission"] for row in rows),
    prefill_requests_with_changed_emission=sum(row["prefill_requests_with_changed_emission"] for row in rows),
    maximum_kl=max(row["max_kl_ref_to_candidate"] for row in rows),
    same_input_emissions_equal=accepted,
)), flush=True)
raise SystemExit(0 if accepted else 3)
PY
