#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
output_dir=$workdir/tma_multicast_kernel_20261013
predecessor_pid=546077
predecessor_log=$workdir/native_pdl_combined_serving_20261013.log
trap 'status=$?; printf "TMA_MULTICAST_KERNEL_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
echo "WAIT_COMBINED_SERVING $(date -Is)"
while kill -0 "$predecessor_pid" 2>/dev/null; do
  predecessor_state=$(ps -p "$predecessor_pid" -o stat= || true)
  if [[ "$predecessor_state" == Z* ]]; then break; fi
  if ! predecessor_command=$(tr '\0' ' ' < "/proc/$predecessor_pid/cmdline"); then break; fi
  [[ "$predecessor_command" == "bash native_pdl_combined_serving_20261013.sh "* || "$predecessor_command" == "bash /dev/shm/w/native_pdl_combined_serving_20261013.sh "* ]]
  sleep 15
done
grep '^NATIVE_PDL_COMBINED_SERVING_EXIT status=' "$predecessor_log"
grep -q '^NATIVE_PDL_COMBINED_DONE ' "$workdir/native_pdl_combined_validate_20261013.log"
grep -q '^NATIVE_PDL_COMBINED_EXIT status=0 ' "$workdir/native_pdl_combined_validate_20261013.log"
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference|[t]ma_multicast_reference' >/dev/null; then
  echo "GPU predecessor still running; refusing duplicate launch"
  exit 1
fi
cd "$workdir"
sha256sum -c tma_multicast_overlay_20261013/baseline.sha256
test ! -e fi_tma_multicast
test ! -e up_tma_multicast
mkdir "$output_dir" up_tma_multicast
cp -a fi_native_pdl_combined fi_tma_multicast
cp -a up_native_pdl_combined/python up_tma_multicast/python
cp -a tma_multicast_overlay_20261013/port07/. fi_tma_multicast/
cp tma_multicast_overlay_20261013/up_racefix/python/sglang/srt/layers/attention/fused_fa3_loader.py \
  up_tma_multicast/python/sglang/srt/layers/attention/fused_fa3_loader.py
cp tma_multicast_overlay_20261013/tma_multicast_reference.py drivers/tma_multicast_reference_20261013.py
sha256sum -c tma_multicast_overlay_20261013/candidate.sha256
cp tma_multicast_overlay_20261013/baseline.sha256 tma_multicast_overlay_20261013/candidate.sha256 "$output_dir/"
export PYTHONPATH=$workdir/up_tma_multicast/python HF_HOME=$workdir/hf
export SGLANG_FUSED_FI_ROOT=$workdir/fi_tma_multicast TORCH_CUDA_ARCH_LIST=9.0a
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 FUSED_FA3_DEDICATED_PRIMARY=1
export SM_SPLIT=112 FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7 FA3_STAGES=2
export TIME_ITERS=100 SEED=31
unset FUSED_NOSPLIT FUSED_KV_SHARE_BUILD FUSED_FA3_KV_SHARE FUSED_FA3_KV_SHARE_BULK
unset FLASHINFER_FUSED_SYMMETRIC FLASHINFER_FUSED_SKIP_FA3 FLASHINFER_FUSED_PLAN_DEBUG FA3_SKIP_KV_LOAD TIME_FORCE
unset FA3_PAGED_TMA NATIVE_PDL_REQUIRE_OVERLAP NATIVE_PDL_TRACE
driver=$workdir/drivers/tma_multicast_reference_20261013.py
select_arm() {
  export FUSED_FA3_TMA_MULTICAST=0 FUSED_FA3_TMA_TRACE=0
  if [ "$1" = GATHER ]; then
    export SGLANG_FUSED_MERGED_DIR=$workdir/merged_tma_gather TORCH_EXTENSIONS_DIR=$workdir/torch_ext_tma_gather
    export FUSED_PAGED_TMA_BUILD=0 FUSED_CLUSTER=1 FLASHINFER_FUSED_CLUSTER=1 FUSED_FA3_CLUSTER_PAIR=0
  else
    export SGLANG_FUSED_MERGED_DIR=$workdir/merged_tma_multicast TORCH_EXTENSIONS_DIR=$workdir/torch_ext_tma_multicast
    export FUSED_PAGED_TMA_BUILD=1 FUSED_CLUSTER=2 FLASHINFER_FUSED_CLUSTER=2 FUSED_FA3_CLUSTER_PAIR=1
    if [ "$1" = MULTICAST ]; then export FUSED_FA3_TMA_MULTICAST=1; fi
  fi
}
for arm in GATHER TMA; do
  select_arm "$arm"
  echo "BUILD_START $arm $(date -Is)"
  timeout --kill-after=30 1500 "$workdir/venv/bin/python" -u -c \
    'from sglang.srt.layers.attention.fused_fa3_loader import load_fused_fa3; print(load_fused_fa3(verbose=True, cta_q=128))' \
    > "$output_dir/build_$arm.log" 2>&1
  echo "BUILD_DONE $arm $(date -Is)"
done
verify8=$(printf '4:8192,%.0s' {1..8})
verify16=$(printf '4:16384,%.0s' {1..8})
verify46=$(printf '4:49152,%.0s' {1..46})
fa3_8=$(printf '1024:32768,%.0s' {1..8})
for fixture in shared tail varied; do
  for arm in GATHER TMA MULTICAST; do
    select_arm "$arm"
    export FUSED_FA3_TMA_TRACE=1
    fixture_env=(FA3_REQUESTS=512:4096 FA2_REQUESTS="$verify8" NATIVE_PDL_REQUIRE_OVERLAP=1)
    if [ "$fixture" = tail ]; then
      fixture_env=(FA3_REQUESTS=1:129 FA2_REQUESTS=4:257 FUSED_NOSPLIT=1 NATIVE_PDL_REQUIRE_OVERLAP=0)
    elif [ "$fixture" = varied ]; then
      fixture_env=(FA3_REQUESTS=513:4097,17:32769,32:65537 FA2_REQUESTS=4:8193,8:32769 NATIVE_PDL_REQUIRE_OVERLAP=0)
    fi
    label=gate_${fixture}_$arm
    echo "START $label $(date -Is)"
    TIME_AB=0 env "${fixture_env[@]}" timeout --kill-after=20 360 "$workdir/venv/bin/python" \
      -u "$driver" > "$output_dir/$label.log" 2>&1
    "$workdir/venv/bin/python" - "$output_dir/$label.log" "$arm" "$fixture" <<'PY'
from pathlib import Path
import re
import sys

log = Path(sys.argv[1]).read_text()
assert log.rstrip().endswith("PASS") and "Traceback" not in log
checks = re.findall(r"CHECK ([^ ]+) bad_heads=(\d+)/(\d+)", log)
assert len(checks) >= 5 and all(int(bad) == 0 and int(total) > 0 for _, bad, total in checks)
assert all(f"candidate_graph_{repeat}" in log for repeat in range(3))
counters = re.findall(r"FA3_PAGED_TMA load_tiles=(\d+) multicast_tiles=(\d+) tail_tiles=(\d+)", log)
if sys.argv[2] == "GATHER":
    assert not counters
else:
    assert counters and max(int(loads) for loads, _, _ in counters) > 0
    if sys.argv[2] == "TMA":
        assert all(int(multicast) == 0 for _, multicast, _ in counters)
    elif sys.argv[3] == "shared":
        assert max(int(multicast) for _, multicast, _ in counters) > 0
    if sys.argv[3] == "tail":
        assert max(int(tails) for _, _, tails in counters) > 0
if sys.argv[3] == "shared":
    assert log.count("TMA_PDL_OVERLAP graph=") == 2
print("TMA_GATE", sys.argv[2:], "checks", len(checks), "counters", counters, flush=True)
PY
  done
done
shapes=("mixed_fresh|FA3_REQUESTS=512:512 FA2_REQUESTS=4:8192,4:32769"
        "prefill2048_v8|FA3_REQUESTS=2048:65536 FA2_REQUESTS=$verify16"
        "append512_v46|FA3_REQUESTS=512:49152 FA2_REQUESTS=$verify46"
        "fa3_8x1024|FA3_REQUESTS=$fa3_8 FA2_REQUESTS="
        "fa3_2x256|FA3_REQUESTS=256:131072,256:131072 FA2_REQUESTS="
        "split_q32_v8|FA3_REQUESTS=32:65536 FA2_REQUESTS=$verify8"
        "varied_odd|FA3_REQUESTS=513:4097,17:32769,32:65537 FA2_REQUESTS=4:8193,8:32769"
        "fa2_only|FA3_REQUESTS= FA2_REQUESTS=$verify8")
for shape in "${shapes[@]}"; do
  read -r -a fixture_env <<< "${shape#*|}"
  for repeat in 1 2; do
    arms=(GATHER TMA MULTICAST)
    if [ "$repeat" = 2 ]; then arms=(MULTICAST TMA GATHER); fi
    for arm in "${arms[@]}"; do
      select_arm "$arm"
      label=${shape%%|*}_${arm}_$repeat
      echo "START $label $(date -Is)"
      env | LC_ALL=C sort | grep -E '^(PYTHONPATH=|SGLANG_FUSED_|FUSED_|FLASHINFER_|TIME_|SM_SPLIT=|SEED=)' \
        > "$output_dir/$label.environment.txt"
      TIME_AB=1 env "${fixture_env[@]}" timeout --kill-after=20 480 "$workdir/venv/bin/python" \
        -u "$driver" > "$output_dir/$label.log" 2>&1
      "$workdir/venv/bin/python" - "$output_dir/$label.log" "$arm" "$repeat" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
log = path.read_text()
assert log.rstrip().endswith("PASS") and "Traceback" not in log
assert "FA3_PAGED_TMA " not in log, "device trace is enabled during timing"
checks = re.findall(r"CHECK ([^ ]+) bad_heads=(\d+)/(\d+)", log)
assert len(checks) == 7 and all(int(bad) == 0 and int(total) > 0 for _, bad, total in checks)
timings = {}
for kernel in ("candidate", "stock"):
    matches = re.findall(r"^TIME " + kernel + r" graph=True ms=([0-9.]+)$", log, re.MULTILINE)
    assert len(matches) == 1 and math.isfinite(float(matches[0])) and float(matches[0]) > 0
    timings[kernel + "_ms"] = float(matches[0])
audit = {"label": path.stem, "arm": sys.argv[2], "repeat": int(sys.argv[3]),
         **timings, "checks": len(checks), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit), flush=True)
PY
      echo "END $label $(date -Is)"
    done
  done
done
"$workdir/venv/bin/python" - "$output_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
audits = [json.loads(path.read_text()) for path in sorted(root.glob("*.audit.json"))]
assert len(audits) == 48, len(audits)
(root / "execution_audits.json").write_text(json.dumps(audits, indent=2) + "\n")
PY
echo "TMA_MULTICAST_KERNEL_DONE $(date -Is)"
