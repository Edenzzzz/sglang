#!/bin/bash
set -euo pipefail
ROOT=${ROOT:-/dev/shm/w/captured_mixed_20261023}
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test ! -d "$ROOT/python"
cp -a /dev/shm/w/up/python "$ROOT/python"
/dev/shm/w/venv/bin/python "$ROOT/apply_capture_native_fa3_20261023.py" \
  "$ROOT/python/sglang/srt/layers/attention/flashattention_backend.py"
/dev/shm/w/venv/bin/python - "$ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
original = (root / "mn_bench.sh").read_text()
original = original.replace("$W/drivers/serving/fanout_agents.py", str(root / "fanout_agents.py"))
(root / "mn_bench_capture.sh").write_text(original)
PY
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export PYTHONPATH=$ROOT/python:$ROOT:/dev/shm/w/fi
export HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_fa3_pdl_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_q32_20261023/native_fa2_q32/native_fa2_q32.so
cd "$ROOT"
/dev/shm/w/venv/bin/python -u capture_serving_workloads_20261023.py "$ROOT" ${CAPTURE_STAGE_ARGS:-}
