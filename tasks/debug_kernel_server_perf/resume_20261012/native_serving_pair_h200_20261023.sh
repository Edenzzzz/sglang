#!/bin/bash
set -euo pipefail
ROOT=${ROOT:-/dev/shm/w/native_serving_pair_h200_20261023}
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test ! -d "$ROOT/python"
cp -a /dev/shm/w/up/python "$ROOT/python"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export PYTHONPATH=$ROOT/python:$ROOT:/dev/shm/w/fi
export HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=${NATIVE_PDL_FA2_LIBRARY:-/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so}
export NATIVE_PDL_FA3_LIBRARY=${NATIVE_PDL_FA3_LIBRARY:-/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so}
export NATIVE_PDL_FA2_FP32_LIBRARY=${NATIVE_PDL_FA2_FP32_LIBRARY:-/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so}
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=${NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY:-/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so}
if [ "${SERVING_TRACE_FEATURES:-0}" = 1 ]; then
  /dev/shm/w/venv/bin/python "$ROOT/apply_steptrace_features.py" "$ROOT/python/sglang/srt"
fi
/dev/shm/w/venv/bin/python "$ROOT/apply_native_pdl_fa3_backend_20261023.py" "$ROOT/python/sglang/srt/layers/attention/flashattention_backend.py"
/dev/shm/w/venv/bin/python - "$ROOT" <<'PY'
from pathlib import Path
import sys

from native_serving_protocol_20261023 import patch_client_source, replace_once

root = Path(sys.argv[1])
client = root / "python/sglang/benchmark/serving.py"
client.write_text(patch_client_source(client.read_text()))
(client.parent / "workload_protocol.py").write_text((root / "workload_protocol.py").read_text())
script = (root / "mn_bench.sh").read_text()
script = replace_once(script, 'export PYTHONPATH=$W/python:$W/fi', 'export PYTHONPATH=${SERVING_PYTHONPATH:?}')
script = replace_once(script, 'mkdir -p $BENCH_BARRIER_DIR && touch $BENCH_BARRIER_DIR/go', 'test "$n" -ge 2 || exit 2\nwc -c < "$SERVING_SERVER_LOG" > "$OUT/server_log_offset"\nmkdir -p $BENCH_BARRIER_DIR && touch $BENCH_BARRIER_DIR/go')
(root / "mn_bench_pair.sh").write_text(script)
PY
cd "$ROOT"
/dev/shm/w/venv/bin/python -u run_native_serving_pair_20261023.py "$ROOT" ${SERVING_PAIR_ARGS:-}
