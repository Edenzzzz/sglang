#!/bin/bash
# A4 tuning (2026-10-13): is the co-launched FA2 executor's low bandwidth a memory-level-parallelism
# limit? Sweep FA2 CTAs per SM (FUSED_FA2_EXEC_CTAS = 2/3/4 x 132) x FA2 KV chunk (auto/1024/4096)
# on the two serving fixtures; same reference gate and roofline arithmetic as the BW run.
# Original note -- B1 follow-up (2026-10-13): the serving profile puts 75% of kernel time in attention, dominated by
# memory-bound verify rows (stock BatchPrefill ~1.34 ms/call, co-launched FA2ExecutorKernel ~1.55
# ms/call at mix32). Measure achieved KV bandwidth at serving verify shapes (q=4, GQA 28/4, D=128):
# fused (co-launch, 64/32/16, auto chunk) vs unfused (FA3 + stock BatchPrefill) vs BatchAttention,
# against the H200 HBM roofline (4.8 TB/s). KV bytes per layer = rows * kv * 4 heads * 128 * 2 * 2 B.
# Every arm must pass the fp32 causal reference.
set -uo pipefail
W=/dev/shm/w
OUT=$W/fa2_chunk_floor_20261013; mkdir -p $OUT
# INHERIT_LOCK=1: called from a script that already holds the GPU lock (re-flocking would deadlock).
[ "${INHERIT_LOCK:-0}" = 1 ] || { exec 9>"$W/a2_kernel.lock"; flock 9; }
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/up_racefix/python SGLANG_FUSED_FI_ROOT=$W/port07 HF_HOME=$W/hf
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto HEADS=28,4 SM_SPLIT=112 FUSED_REFERENCE_WORKSPACE_BYTES=1073741824
export TIME_AB=1 TIME_CUDA_GRAPH=1 TIME_ITERS=50 TIME_WARMUP=10 FUSED_FA3_CTA_Q=128
unset FLASHINFER_FUSED_PLAN_DEBUG
# label|rows|kv|env
# Chunk-floor check: the A2 auto rule was validated only against chunks <= 2048; serving verify
# rows want 4096. Before giving auto a floor, time auto / 2048 / 4096 on the A2 fixtures (small KV,
# where a large chunk could leave SMs idle) and on the serving verify fixtures.
shapes=()
fx=("b_kv4096_r1_q8|1|4096|MODE=band1 BAND_SIDE=fa2 BAND_KV=4096 BAND_ROWS=1 BAND_QO=8"
    "b_kv16384_r1_q4|1|16384|MODE=band1 BAND_SIDE=fa2 BAND_KV=16384 BAND_ROWS=1 BAND_QO=4"
    "b_kv4096_r16_q8|16|4096|MODE=band1 BAND_SIDE=fa2 BAND_KV=4096 BAND_ROWS=16 BAND_QO=8"
    "b_kv16384_r16_q8|16|16384|MODE=band1 BAND_SIDE=fa2 BAND_KV=16384 BAND_ROWS=16 BAND_QO=8"
    "mixed_verify_q8|32|2048|MODE=mixed_verify VERIFY_QO=8"
    "mixed_verify_q4|32|2048|MODE=mixed_verify VERIFY_QO=4"
    "mix_decode32|32|2048|MODE=mix_decode DEC_ROWS=32"
    "v_r46_kv16k|46|16384|MODE=band1 BAND_SIDE=fa2 BAND_KV=16384 BAND_ROWS=46 BAND_QO=4"
    "v_r16_kv128k|16|131072|MODE=band1 BAND_SIDE=fa2 BAND_KV=131072 BAND_ROWS=16 BAND_QO=4"
    "v_r8_kv8k|8|8192|MODE=band1 BAND_SIDE=fa2 BAND_KV=8192 BAND_ROWS=8 BAND_QO=4")
for f in "${fx[@]}"; do
  IFS='|' read -r label rows kv envs <<< "$f"
  for chunk in auto 2048 4096; do shapes+=("${label}_k${chunk}|$rows|$kv|$envs FUSED_FA2_KV_CHUNK=$chunk"); done
done
for s in "${shapes[@]}"; do
  IFS='|' read -r label rows kv envs <<< "$s"
  env $envs timeout 1500 $W/venv/bin/python -u $W/drivers/fused_reference.py > $OUT/$label.log 2>&1
  rc=$?
  bad=$(grep -oE 'bad rows [0-9]+/[0-9]+' $OUT/$label.log | tail -1)
  python3 - "$label" "$rows" "$kv" "$rc" "$bad" $OUT/$label.log <<'PY'
import re, sys
label, rows, kv, rc, bad, log = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
txt = open(log).read()
gb = rows * kv * 4 * 128 * 2 * 2 / 1e9
out = []
for name, pat in [("fused", r"^\[time\] fused +=\s*([0-9.]+)"), ("unfused", r"^\[time\] unfused +=\s*([0-9.]+)"),
                  ("BatchAttention", r"BatchAttention \(flash[^=]*=\s*([0-9.]+)")]:
    m = re.search(pat, txt, re.M)
    if m:
        ms = float(m.group(1)); out.append(f"{name} {ms:.4f} ms = {gb / ms:.2f} TB/s ({100 * gb / ms / 4.8:.0f}% of 4.8)")
print(f"BW {label} rc={rc} [{bad}] KV {gb:.3f} GB/layer | " + " | ".join(out))
PY
done
echo FA2_CHUNK_FLOOR_DONE
