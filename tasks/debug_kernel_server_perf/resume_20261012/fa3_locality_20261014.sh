#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
output_dir=$workdir/fa3_locality_20261014
trap 'status=$?; printf "FA3_LOCALITY_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
cd "$workdir"
grep -q '^FA3_NCU_V2_DONE ' fa3_ncu_v2_20261014.log
grep -q '^FA3_NCU_V2_EXIT status=0 ' fa3_ncu_v2_20261014.log
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[r]un_mixed_workloads|[f]used_reference|[t]ma_(multicast|warp)_reference' >/dev/null; then
  echo "GPU predecessor still running; refusing duplicate launch"
  exit 1
fi
test ! -e "$output_dir"
mkdir "$output_dir"
sha256sum -c tma_multicast_overlay_20261013/candidate.sha256
export PYTHONPATH=$workdir/up_tma_multicast/python HF_HOME=$workdir/hf
export SGLANG_FUSED_FI_ROOT=$workdir/fi_tma_multicast
export SGLANG_FUSED_MERGED_DIR=$workdir/merged_tma_gather TORCH_EXTENSIONS_DIR=$workdir/torch_ext_tma_gather
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export SGLANG_FUSED_FA3_CONSUMER_REGS=216 FUSED_WORK_BUDGET_SPLIT=1
export FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1 FUSED_FA2_KV_CHUNK=auto4096
export FUSED_FA3_DEDICATED_PRIMARY=1 FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7 FA3_STAGES=2 SM_SPLIT=112
export FUSED_PAGED_TMA_BUILD=0 FUSED_CLUSTER=1 FLASHINFER_FUSED_CLUSTER=1
export FUSED_FA3_TMA_TRACE=0 FUSED_FA3_TMA_MULTICAST=0 TIME_AB=1 TIME_ITERS=100 SEED=31
unset FUSED_NOSPLIT FUSED_KV_SHARE_BUILD FLASHINFER_FUSED_SYMMETRIC FLASHINFER_FUSED_SKIP_FA3 FA3_SKIP_KV_LOAD
unset NATIVE_PDL_REQUIRE_OVERLAP NATIVE_PDL_TRACE FA3_PAGED_TMA
driver=$workdir/drivers/tma_multicast_reference_20261013.py
verify8=$(printf '4:16384,%.0s' {1..8})
verify46=$(printf '4:49152,%.0s' {1..46})
fa3_8=$(printf '1024:32768,%.0s' {1..8})
shapes=("fa3_8x1024|FA3_REQUESTS=$fa3_8 FA2_REQUESTS=,"
        "prefill2048_v8|FA3_REQUESTS=2048:65536 FA2_REQUESTS=$verify8"
        "append512_v46|FA3_REQUESTS=512:49152 FA2_REQUESTS=$verify46"
        "varied_odd|FA3_REQUESTS=513:4097,17:32769,32:65537 FA2_REQUESTS=4:8193,8:32769")
for shape in "${shapes[@]}"; do
  read -r -a fixture_env <<< "${shape#*|}"
  for repeat in 1 2; do
    plans=(0 1)
    if [ "$repeat" = 2 ]; then plans=(1 0); fi
    for paired in "${plans[@]}"; do
      export FUSED_FA3_CLUSTER_PAIR=$paired
      label=${shape%%|*}_paired${paired}_$repeat
      echo "START $label $(date -Is)"
      env "${fixture_env[@]}" timeout --kill-after=20 480 "$workdir/venv/bin/python" \
        -u "$driver" > "$output_dir/$label.log" 2>&1
      "$workdir/venv/bin/python" - "$output_dir/$label.log" "$paired" "$repeat" "${shape%%|*}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
log = path.read_text()
assert log.rstrip().endswith("PASS") and "Traceback" not in log
assert "FA3_PAGED_TMA " not in log
checks = re.findall(r"CHECK ([^ ]+) bad_heads=(\d+)/(\d+)", log)
assert len(checks) == 7 and all(int(bad) == 0 and int(total) > 0 for _, bad, total in checks)
fixture = json.loads(next(line.removeprefix("FIXTURE ") for line in log.splitlines() if line.startswith("FIXTURE ")))
if sys.argv[4] == "fa3_8x1024":
    assert fixture["native"] == []
timings = {}
for kernel in ("candidate", "stock"):
    matches = re.findall(r"^TIME " + kernel + r" graph=True ms=([0-9.]+)$", log, re.MULTILINE)
    assert len(matches) == 1 and math.isfinite(float(matches[0])) and float(matches[0]) > 0
    timings[kernel + "_ms"] = float(matches[0])
audit = {"label": path.stem, "shape": sys.argv[4], "paired": bool(int(sys.argv[2])),
         "repeat": int(sys.argv[3]), **timings, "fixture": fixture,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit), flush=True)
PY
      echo "END $label $(date -Is)"
    done
  done
done
echo "FA3_LOCALITY_DONE $(date -Is)"
