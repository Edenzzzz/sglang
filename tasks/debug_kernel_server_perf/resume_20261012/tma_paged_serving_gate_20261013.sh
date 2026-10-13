#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
output_dir=$workdir/tma_paged_serving_gate_20261013
predecessor_pid=561859
predecessor_log=$workdir/tma_multicast_kernel_20261013.log
trap 'status=$?; printf "TMA_PAGED_SERVING_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
echo "WAIT_TMA_KERNEL $(date -Is)"
while kill -0 "$predecessor_pid" 2>/dev/null; do
  predecessor_state=$(ps -p "$predecessor_pid" -o stat= || true)
  if [[ "$predecessor_state" == Z* ]]; then break; fi
  if ! predecessor_command=$(tr '\0' ' ' < "/proc/$predecessor_pid/cmdline"); then break; fi
  [[ "$predecessor_command" == "bash tma_multicast_kernel_20261013.sh "* || "$predecessor_command" == "bash /dev/shm/w/tma_multicast_kernel_20261013.sh "* ]]
  sleep 15
done
grep -q '^TMA_MULTICAST_KERNEL_DONE ' "$predecessor_log"
grep -q '^TMA_MULTICAST_KERNEL_EXIT status=0 ' "$predecessor_log"
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[r]un_mixed_workloads|[f]used_reference|[t]ma_multicast_reference' >/dev/null; then
  echo "GPU predecessor still running; refusing duplicate launch"
  exit 1
fi
cd "$workdir"
sha256sum -c tma_multicast_overlay_20261013/candidate.sha256
sha256sum -c tma_paged_serving_overlay_20261013/baseline.sha256
test ! -e up_tma_paged_serving
test ! -e "$output_dir"
mkdir up_tma_paged_serving "$output_dir"
cp -a up_tma_multicast/python up_tma_paged_serving/python
cp tma_paged_serving_overlay_20261013/up_racefix/python/sglang/srt/layers/attention/flashinfer_backend.py \
  up_tma_paged_serving/python/sglang/srt/layers/attention/flashinfer_backend.py
cp tma_paged_serving_overlay_20261013/srv_correctness.py drivers/srv_correctness_tma_paged_20261013.py
sha256sum -c tma_paged_serving_overlay_20261013/candidate.sha256
cp tma_paged_serving_overlay_20261013/baseline.sha256 tma_paged_serving_overlay_20261013/candidate.sha256 "$output_dir/"
cp "$workdir/q4_gate_graphs_20261012/MTP_LPT_Q4G.json" "$output_dir/"
export PYTHONPATH=$workdir/up_tma_paged_serving/python HF_HOME=$workdir/hf MODEL=$workdir/qwen7b
export SGLANG_FUSED_FI_ROOT=$workdir/fi_tma_multicast
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 FUSED_FA3_DEDICATED_PRIMARY=1 FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7
export SGLANG_FUSED_SM_FA3=112 SGLANG_FUSED_SM_MARGIN=0 SGLANG_SPLIT_OVERLAP=0
export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3 BENCH_CHUNK=512 BENCH_PAGE_SIZE=128
export CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DECODE_TOKENS=1200
export CORRECTNESS_DATASET=$workdir/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export SGLANG_FUSED_DECODE_GUARD=0 SGLANG_FUSED_SPLIT_TOL=0.02 SGLANG_FUSED_VERIFY_SNAPSHOT=1
unset SGLANG_FUSED_KERNEL_EXEC SGLANG_FUSED_DISPATCH SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH
unset FUSED_NOSPLIT FUSED_KV_SHARE_BUILD FUSED_FA3_KV_SHARE FUSED_FA3_KV_SHARE_BULK
unset FLASHINFER_FUSED_SYMMETRIC FLASHINFER_FUSED_SKIP_FA3 FLASHINFER_FUSED_PLAN_DEBUG FA3_SKIP_KV_LOAD
unset FA3_PAGED_TMA NATIVE_PDL_REQUIRE_OVERLAP NATIVE_PDL_TRACE SGLANG_FUSED_VERIFY_EPILOGUE
driver=$workdir/drivers/srv_correctness_tma_paged_20261013.py
for variant in STOCK SPLIT GATHER TMA MULTICAST; do
  arm=MTP_LPT_P128
  export SGLANG_FUSED_NATIVE_PDL=0 SGLANG_FUSED_SPLIT_VERIFY=0
  export FUSED_PAGED_TMA_BUILD=0 FUSED_CLUSTER=1 FLASHINFER_FUSED_CLUSTER=1 FUSED_FA3_CLUSTER_PAIR=0
  export FUSED_FA3_TMA_MULTICAST=0 FUSED_FA3_TMA_TRACE=0
  export SGLANG_FUSED_MERGED_DIR=$workdir/merged_tma_gather TORCH_EXTENSIONS_DIR=$workdir/torch_ext_tma_gather
  if [ "$variant" = SPLIT ]; then
    arm=MTPMIX_SPLIT_P128
    export SGLANG_FUSED_SPLIT_VERIFY=1
  elif [ "$variant" != STOCK ]; then
    arm=MTPMIXFUSED_${variant}_P128
    export SGLANG_FUSED_NATIVE_PDL=1 SGLANG_FUSED_SPLIT_VERIFY=1
    if [ "$variant" != GATHER ]; then
      export FUSED_PAGED_TMA_BUILD=1 FUSED_CLUSTER=2 FLASHINFER_FUSED_CLUSTER=2 FUSED_FA3_CLUSTER_PAIR=1
      export FUSED_FA3_TMA_TRACE=1
      export SGLANG_FUSED_MERGED_DIR=$workdir/merged_tma_multicast TORCH_EXTENSIONS_DIR=$workdir/torch_ext_tma_multicast
      if [ "$variant" = MULTICAST ]; then export FUSED_FA3_TMA_MULTICAST=1; fi
    fi
  fi
  echo "START $variant $arm $(date -Is)"
  env | LC_ALL=C sort | grep -E '^(PYTHONPATH=|SGLANG_FUSED_|SGLANG_SPLIT_|FUSED_|FLASHINFER_|BENCH_|CORRECTNESS_|MTP_)' \
    > "$output_dir/$arm.environment.txt"
  timeout --kill-after=30 2400 "$workdir/venv/bin/python" -u "$driver" "$arm" "$output_dir" \
    > "$output_dir/$arm.run.log" 2>&1
  "$workdir/venv/bin/python" -u "$driver" --compare "$output_dir" MTP_LPT_Q4G "$arm" \
    > "$output_dir/$arm.compare.log" 2>&1
  if [ "$variant" != STOCK ]; then
    "$workdir/venv/bin/python" -u "$driver" --compare "$output_dir" MTP_LPT_P128 "$arm" \
      > "$output_dir/$arm.page128_compare.log" 2>&1
  fi
  "$workdir/venv/bin/python" - "$output_dir" "$arm" "$variant" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

root, arm, variant = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
path = root / f"server_{arm}.log"
log = path.read_text()
result = json.loads((root / f"{arm}.json").read_text())
assert result["_requested_page_size"] == 128
assert not result["_fallbacks"] and not result["_tracebacks"]
assert "Traceback" not in log and "FUSED_SPLIT_VERIFY FAIL" not in log
assert "cuda graph: True" in log
comparisons = [(root / f"{arm}.compare.log").read_text()]
if variant != "STOCK":
    comparisons.append((root / f"{arm}.page128_compare.log").read_text())
assert all("GATE PASS" in comparison for comparison in comparisons)
checks = re.findall(r"FUSED_KERNEL_VERIFY PASS worst_head=([0-9.]+) native_pdl=1", log)
paged = re.findall(r"NATIVE_PDL_PAGED_CALLS (\d+) page_size=128", log)
counters = re.findall(r"FA3_PAGED_TMA load_tiles=(\d+) multicast_tiles=(\d+) tail_tiles=(\d+)", log)
if variant == "SPLIT":
    assert result["_fused_split_calls"] > 0 and "FUSED_SPLIT_VERIFY ok" in log
elif variant != "STOCK":
    assert paged and max(map(int, paged)) > 0 and checks
    assert "NATIVE_PDL_COMPACT_CALLS" in log
    if variant == "GATHER":
        assert not counters
    else:
        assert counters and max(int(loads) for loads, _, _ in counters) > 0
        if variant == "MULTICAST":
            assert max(int(multicast) for _, multicast, _ in counters) > 0
        else:
            assert all(int(multicast) == 0 for _, multicast, _ in counters)
audit = {"arm": arm, "variant": variant, "requested_page_size": 128,
         "paged_calls": max(map(int, paged), default=0), "layer0_checks": len(checks),
         "layer0_worst": max(map(float, checks), default=None),
         "max_multicast_tiles": max((int(multicast) for _, multicast, _ in counters), default=0),
         "server_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
         "comparisons": comparisons}
(root / f"{arm}.audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit), flush=True)
PY
  echo "END $variant $(date -Is)"
done
echo "TMA_PAGED_SERVING_DONE $(date -Is)"
