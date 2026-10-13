#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
output_dir=$workdir/fa3_kv_geometry_v2_kernel_20261014
trap 'status=$?; printf "FA3_KV_GEOMETRY_V2_KERNEL_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
cd "$workdir"
grep -q '^FA3_LOCALITY_DONE ' fa3_locality_20261014.log
grep -q '^FA3_LOCALITY_EXIT status=0 ' fa3_locality_20261014.log
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[r]un_mixed_workloads|[f]used_reference|[t]ma_.*_reference|[f]a3_kv_geometry_reference' >/dev/null; then
  echo "GPU predecessor still running; refusing duplicate launch"
  exit 1
fi
sha256sum -c fa3_kv_geometry_v2_overlay_20261014/baseline.sha256
test ! -e fi_fa3_kv_geometry_v2
test ! -e up_fa3_kv_geometry_v2
mkdir "$output_dir" up_fa3_kv_geometry_v2
cp -a fi_tma_multicast fi_fa3_kv_geometry_v2
cp -a up_tma_paged_serving/python up_fa3_kv_geometry_v2/python
cp -a fa3_kv_geometry_v2_overlay_20261014/port07/. fi_fa3_kv_geometry_v2/
cp fa3_kv_geometry_v2_overlay_20261014/up_racefix/python/sglang/srt/layers/attention/fused_fa3_loader.py \
  up_fa3_kv_geometry_v2/python/sglang/srt/layers/attention/fused_fa3_loader.py
cp drivers/tma_multicast_reference_20261013.py drivers/fa3_kv_geometry_v2_reference_20261014.py
sha256sum -c fa3_kv_geometry_v2_overlay_20261014/candidate.sha256
cp fa3_kv_geometry_v2_overlay_20261014/baseline.sha256 fa3_kv_geometry_v2_overlay_20261014/candidate.sha256 "$output_dir/"
export PYTHONPATH=$workdir/up_fa3_kv_geometry_v2/python HF_HOME=$workdir/hf
export SGLANG_FUSED_FI_ROOT=$workdir/fi_fa3_kv_geometry_v2 TORCH_CUDA_ARCH_LIST=9.0a
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 FUSED_FA3_DEDICATED_PRIMARY=1
export FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7 FA3_STAGES=2 SM_SPLIT=112
export FUSED_PAGED_TMA_BUILD=0 FUSED_CLUSTER=1 FLASHINFER_FUSED_CLUSTER=1 FUSED_FA3_CLUSTER_PAIR=0
export SGLANG_FUSED_FA3_CONSUMER_REGS=216
export TIME_ITERS=100 SEED=31
unset FUSED_NOSPLIT FUSED_KV_SHARE_BUILD FUSED_FA3_KV_SHARE FUSED_FA3_KV_SHARE_BULK
unset FLASHINFER_FUSED_SYMMETRIC FLASHINFER_FUSED_SKIP_FA3 FLASHINFER_FUSED_PLAN_DEBUG FA3_SKIP_KV_LOAD TIME_FORCE
unset FA3_PAGED_TMA NATIVE_PDL_REQUIRE_OVERLAP NATIVE_PDL_TRACE
driver=$workdir/drivers/fa3_kv_geometry_v2_reference_20261014.py
select_variant() {
  export FUSED_FA3_TMA_TRACE=0 FUSED_FA3_TMA_MULTICAST=0 FUSED_FA3_CLUSTER_PAIR=0
  export FA3_CTA_KV=$1
  export SGLANG_FUSED_MERGED_DIR=$workdir/merged_fa3_kv_v2_$1
  export TORCH_EXTENSIONS_DIR=$workdir/torch_ext_fa3_kv_v2_$1
}
verify8=$(printf '4:8192,%.0s' {1..8})
verify16=$(printf '4:16384,%.0s' {1..8})
verify46=$(printf '4:49152,%.0s' {1..46})
fa3_8=$(printf '1024:32768,%.0s' {1..8})
for variant in 96 128; do
  select_variant "$variant"
  echo "BUILD_START $variant $(date -Is)"
  timeout --kill-after=30 1500 "$workdir/venv/bin/python" -u -c \
    'from sglang.srt.layers.attention.fused_fa3_loader import load_fused_fa3; print(load_fused_fa3(verbose=True, cta_q=128))' \
    > "$output_dir/build_$variant.log" 2>&1
  echo "BUILD_DONE $variant $(date -Is)"
  for fixture in shared tail varied multiwork; do
    for paired in 0 1; do
      export FUSED_FA3_CLUSTER_PAIR=$paired FUSED_FA3_TMA_TRACE=0
      fixture_env=(FA3_REQUESTS=512:65536 FA2_REQUESTS="$verify8" NATIVE_PDL_REQUIRE_OVERLAP=1)
      if [ "$fixture" = tail ]; then
        fixture_env=(FA3_REQUESTS=17:95,17:96,17:97,32:127,32:128,32:129,32:191,32:192,32:193,17:257,32:385 FA2_REQUESTS=4:257,8:513 FUSED_NOSPLIT=1 NATIVE_PDL_REQUIRE_OVERLAP=0)
      elif [ "$fixture" = varied ]; then
        fixture_env=(FA3_REQUESTS=513:4097,17:32769,32:65537 FA2_REQUESTS=4:8193,8:32769 NATIVE_PDL_REQUIRE_OVERLAP=0)
      elif [ "$fixture" = multiwork ]; then
        fixture_env=(FA3_REQUESTS="$fa3_8" FA2_REQUESTS=, NATIVE_PDL_REQUIRE_OVERLAP=0)
      fi
      label=gate_${variant}_${fixture}_paired$paired
      echo "START $label $(date -Is)"
      TIME_AB=0 env "${fixture_env[@]}" timeout --kill-after=20 360 "$workdir/venv/bin/python" \
        -u "$driver" > "$output_dir/$label.log" 2>&1
      "$workdir/venv/bin/python" - "$output_dir/$label.log" "$paired" "$fixture" <<'PY'
from pathlib import Path
import re
import sys

log = Path(sys.argv[1]).read_text()
assert log.rstrip().endswith("PASS") and "Traceback" not in log
checks = re.findall(r"CHECK ([^ ]+) bad_heads=(\d+)/(\d+)", log)
assert len(checks) >= 5 and all(int(bad) == 0 and int(total) > 0 for _, bad, total in checks)
assert "FA3_PAGED_TMA " not in log
if sys.argv[3] == "shared":
    assert log.count("TMA_PDL_OVERLAP graph=") == 2
if sys.argv[3] == "varied":
    plans = re.findall(r"TMA_PRIMARY_PLAN splits=(\d+) merge_rows=(\d+)", log)
    assert plans and all(int(splits) > 1 and int(merges) > 0 for splits, merges in plans)
print("FA3_KV_GEOMETRY_V2_GATE", Path(sys.argv[1]).stem, "checks", len(checks), flush=True)
PY
    done
  done
done
shapes=("fresh512_v46|FA3_REQUESTS=512:512 FA2_REQUESTS=$verify46"
        "prefill2048_v8|FA3_REQUESTS=2048:65536 FA2_REQUESTS=$verify16"
        "append512_v46|FA3_REQUESTS=512:49152 FA2_REQUESTS=$verify46"
        "fa3_8x1024|FA3_REQUESTS=$fa3_8 FA2_REQUESTS=,"
        "varied_odd|FA3_REQUESTS=513:4097,17:32769,32:65537 FA2_REQUESTS=4:8193,8:32769")
for shape in "${shapes[@]}"; do
  read -r -a fixture_env <<< "${shape#*|}"
  for repeat in 1 2; do
    variants=(128 96)
    modes=(0 1)
    if [ "$repeat" = 2 ]; then variants=(96 128); modes=(1 0); fi
    for variant in "${variants[@]}"; do
      select_variant "$variant"
      for paired in "${modes[@]}"; do
        export FUSED_FA3_CLUSTER_PAIR=$paired
        label=${shape%%|*}_${variant}_paired${paired}_$repeat
        echo "START $label $(date -Is)"
        env | LC_ALL=C sort | grep -E '^(PYTHONPATH=|SGLANG_FUSED_|FUSED_|FLASHINFER_|TIME_|SM_SPLIT=|SEED=)' \
          > "$output_dir/$label.environment.txt"
        TIME_AB=1 env "${fixture_env[@]}" timeout --kill-after=20 480 "$workdir/venv/bin/python" \
          -u "$driver" > "$output_dir/$label.log" 2>&1
        "$workdir/venv/bin/python" - "$output_dir/$label.log" "$variant" "$repeat" "$paired" "${shape%%|*}" <<'PY'
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
audit = {"label": path.stem, "shape": sys.argv[5], "tile_kv": int(sys.argv[2]),
         "repeat": int(sys.argv[3]), "paired": bool(int(sys.argv[4])), **timings,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit), flush=True)
PY
        echo "END $label $(date -Is)"
      done
    done
  done
done
echo "FA3_KV_GEOMETRY_V2_KERNEL_DONE $(date -Is)"


