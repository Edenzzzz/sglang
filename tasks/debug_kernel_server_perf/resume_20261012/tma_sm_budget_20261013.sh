#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
output_dir=$workdir/tma_sm_budget_20261013
trap 'status=$?; printf "TMA_SM_BUDGET_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
cd "$workdir"
grep -q '^TMA_PAGED_SERVING_DONE ' tma_paged_serving_gate_20261013.log
grep -q '^TMA_PAGED_SERVING_EXIT status=0 ' tma_paged_serving_gate_20261013.log
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[r]un_mixed_workloads|[f]used_reference|[t]ma_multicast_reference' >/dev/null; then
  echo "GPU predecessor still running; refusing duplicate launch"
  exit 1
fi
test ! -e "$output_dir"
mkdir "$output_dir"
sha256sum -c tma_multicast_overlay_20261013/candidate.sha256
cp tma_multicast_overlay_20261013/candidate.sha256 "$output_dir/"
export PYTHONPATH=$workdir/up_tma_multicast/python HF_HOME=$workdir/hf
export SGLANG_FUSED_FI_ROOT=$workdir/fi_tma_multicast TORCH_CUDA_ARCH_LIST=9.0a
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 FUSED_FA3_DEDICATED_PRIMARY=1
export FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7 FA3_STAGES=2 FUSED_FA3_TMA_TRACE=0
export TIME_AB=1 TIME_ITERS=100 SEED=31
unset FUSED_NOSPLIT FUSED_KV_SHARE_BUILD FUSED_FA3_KV_SHARE FUSED_FA3_KV_SHARE_BULK
unset FLASHINFER_FUSED_SYMMETRIC FLASHINFER_FUSED_SKIP_FA3 FLASHINFER_FUSED_PLAN_DEBUG FA3_SKIP_KV_LOAD TIME_FORCE
unset FA3_PAGED_TMA NATIVE_PDL_REQUIRE_OVERLAP NATIVE_PDL_TRACE
driver=$workdir/drivers/tma_multicast_reference_20261013.py
select_arm() {
  export FUSED_FA3_TMA_MULTICAST=0
  if [ "$1" = GATHER ]; then
    export SGLANG_FUSED_MERGED_DIR=$workdir/merged_tma_gather TORCH_EXTENSIONS_DIR=$workdir/torch_ext_tma_gather
    export FUSED_PAGED_TMA_BUILD=0 FUSED_CLUSTER=1 FLASHINFER_FUSED_CLUSTER=1 FUSED_FA3_CLUSTER_PAIR=0
  else
    export SGLANG_FUSED_MERGED_DIR=$workdir/merged_tma_multicast TORCH_EXTENSIONS_DIR=$workdir/torch_ext_tma_multicast
    export FUSED_PAGED_TMA_BUILD=1 FUSED_CLUSTER=2 FLASHINFER_FUSED_CLUSTER=2 FUSED_FA3_CLUSTER_PAIR=1
    if [ "$1" = MULTICAST ]; then export FUSED_FA3_TMA_MULTICAST=1; fi
  fi
}
verify8=$(printf '4:16384,%.0s' {1..8})
verify46=$(printf '4:49152,%.0s' {1..46})
shapes=("fresh512_v46|FA3_REQUESTS=512:512 FA2_REQUESTS=$verify46"
        "append512_v46|FA3_REQUESTS=512:49152 FA2_REQUESTS=$verify46"
        "prefill2048_v8|FA3_REQUESTS=2048:65536 FA2_REQUESTS=$verify8")
for shape in "${shapes[@]}"; do
  read -r -a fixture_env <<< "${shape#*|}"
  for repeat in 1 2; do
    budgets=(96 112 120)
    arms=(GATHER TMA MULTICAST)
    if [ "$repeat" = 2 ]; then budgets=(120 112 96); arms=(MULTICAST TMA GATHER); fi
    for budget in "${budgets[@]}"; do
      export SM_SPLIT=$budget
      for arm in "${arms[@]}"; do
        select_arm "$arm"
        label=${shape%%|*}_sm${budget}_${arm}_$repeat
        echo "START $label $(date -Is)"
        env | LC_ALL=C sort | grep -E '^(PYTHONPATH=|SGLANG_FUSED_|FUSED_|FLASHINFER_|TIME_|SM_SPLIT=|SEED=)' \
          > "$output_dir/$label.environment.txt"
        env "${fixture_env[@]}" timeout --kill-after=20 480 "$workdir/venv/bin/python" \
          -u "$driver" > "$output_dir/$label.log" 2>&1
        "$workdir/venv/bin/python" - "$output_dir/$label.log" "$arm" "$repeat" "$budget" "${shape%%|*}" <<'PY'
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
timings = {}
for kernel in ("candidate", "stock"):
    matches = re.findall(r"^TIME " + kernel + r" graph=True ms=([0-9.]+)$", log, re.MULTILINE)
    assert len(matches) == 1 and math.isfinite(float(matches[0])) and float(matches[0]) > 0
    timings[kernel + "_ms"] = float(matches[0])
audit = {"label": path.stem, "shape": sys.argv[5], "arm": sys.argv[2],
         "repeat": int(sys.argv[3]), "sm_split": int(sys.argv[4]), **timings,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit), flush=True)
PY
        echo "END $label $(date -Is)"
      done
    done
  done
done
echo "TMA_SM_BUDGET_DONE $(date -Is)"
