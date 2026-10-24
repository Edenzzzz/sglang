#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
root=$workdir/b1_chain_shadow_20261014
overlay=$workdir/b1_chain_shadow_overlay_20261014
trap 'status=$?; printf "B1_CHAIN_SHADOW_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
grep -q '^B1_CHAIN_EPILOGUE_DONE ' "$workdir/b1_chain_epilogue_v4_20261014.log"
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[w]orkload_driver|[r]un_mixed_workloads' >/dev/null; then
  exit 1
fi
cd "$workdir"
sha256sum -c b1_chain_epilogue_overlay_v4_20261014/candidate.sha256
test ! -e up_b1_chain_shadow
test ! -e "$root"
mkdir up_b1_chain_shadow "$root"
cp -a up_b1_chain_epilogue_v4/python up_b1_chain_shadow/python
cp "$overlay/eagle_worker_common.py" up_b1_chain_shadow/python/sglang/srt/speculative/eagle_worker_common.py
sha256sum -c "$overlay/candidate.sha256"
cp "$overlay/candidate.sha256" "$root/"
source "$workdir/tree_env.sh"
export PYTHONPATH=$workdir/up_b1_chain_shadow/python HF_HOME=$workdir/hf MODEL=$workdir/qwen7b
export SGLANG_FUSED_FI_ROOT=$workdir/port07
export SGLANG_FUSED_MERGED_DIR=$workdir/merged_colaunch TORCH_EXTENSIONS_DIR=$workdir/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112 SGLANG_SPLIT_OVERLAP=0
export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
export CHAT_DATASET=$workdir/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH SGLANG_STEP_TRACE
unset SGLANG_EDF_SCHEDULING SGLANG_EDF_LATENCY_MODEL SGLANG_EDF_ACCEPT_MODE
unset SGLANG_FUSED_NATIVE_PDL SGLANG_FUSED_KERNEL_EXEC SGLANG_FUSED_DISPATCH SGLANG_FUSED_DECODE_GUARD
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_SNAPSHOT
unset SGLANG_TWO_EXEC SGLANG_MIXED_ATTENTION_KERNEL SGLANG_ALLOW_MIXED_CHUNK_SPEC SGLANG_MIXED_SPEC
driver=$workdir/drivers/srv_correctness_edf_20261014.py
for arm in MTPMIX_SPLIT_B1_SHADOW MTPMIXFUSED_B1_SHADOW; do
  echo "B1_CHAIN_SHADOW_START $arm $(date -Is)"
  CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET="$CHAT_DATASET" BENCH_CHUNK=512 \
    CORRECTNESS_DECODE_TOKENS=1200 SGLANG_FUSED_DECODE_GUARD=0 \
    SGLANG_FUSED_CHAIN_TREE=1 SGLANG_FUSED_VERIFY_EPILOGUE=1 \
    timeout --kill-after=30 1200 "$workdir/venv/bin/python" -u "$driver" "$arm" "$root" \
    > "$root/$arm.run.log" 2>&1
  "$workdir/venv/bin/python" - "$root/server_$arm.log" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
log = path.read_text()
assert "Traceback" not in log and "fallback" not in log.lower()
assert "cuda graph: True" in log and "FUSED_VERIFY_EPILOGUE active" in log
samples = [tuple(map(int, entry)) for entry in re.findall(
    r"FUSED_CHAIN_VALIDATED calls=(\d+) batch=(\d+) min_context=(\d+) max_context=(\d+)", log
)]
assert len(samples) >= 32 and max(entry[3] for entry in samples) >= 12000
audit = {"exact_metadata_checks": samples,
         "server_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print("B1_CHAIN_SHADOW_PASS", path.name, len(samples), flush=True)
PY
done
echo "B1_CHAIN_SHADOW_DONE $(date -Is)"
