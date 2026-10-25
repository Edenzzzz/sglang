#!/bin/bash
# Serving-protocol pieces of the capture / pair source tree (as native_serving_pair_h200_20261023.sh makes them):
# patched benchmark client, workload_protocol beside it, mn_bench_pair.sh. Claude, 2026-10-25.
set -euo pipefail
S=${1:-/dev/shm/w/qwen25_v3_common_screen_h200_20261023}
cp /dev/shm/w/drivers/serving/workload_protocol.py $S/
cd $S
/dev/shm/w/venv/bin/python - "$S" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from native_serving_protocol_20261023 import patch_client_source, replace_once
root = Path(sys.argv[1])
client = root / "python/sglang/benchmark/serving.py"
src = client.read_text()
if "workload_protocol" not in src:
    client.write_text(patch_client_source(src))
(client.parent / "workload_protocol.py").write_text((root / "workload_protocol.py").read_text())
script = (root / "mn_bench.sh").read_text()
script = replace_once(script, 'export PYTHONPATH=$W/python:$W/fi', 'export PYTHONPATH=${SERVING_PYTHONPATH:?}')
script = replace_once(script, 'mkdir -p $BENCH_BARRIER_DIR && touch $BENCH_BARRIER_DIR/go', 'test "$n" -ge 2 || exit 2\nwc -c < "$SERVING_SERVER_LOG" > "$OUT/server_log_offset"\nmkdir -p $BENCH_BARRIER_DIR && touch $BENCH_BARRIER_DIR/go')
(root / "mn_bench_pair.sh").write_text(script)
print("SRC_FIXED")
PY
