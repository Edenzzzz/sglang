#!/bin/bash
# FA2 split-KV chunk under the co-launch with classes 64/32/16 (the spill-free set). A fixed
# 1024-key chunk gives a single kv-4096 request only 16 FA2 items to spread over 264 CTAs.
set -uo pipefail
W=/dev/shm/w
OUT=$W/colaunch_chunk_20261012
mkdir -p "$OUT"
until grep -q COLAUNCH_TILES_DONE $W/colaunch_tiles_20261012.log 2>/dev/null; do sleep 30; done
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/fi_colaunch
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_t64_32_16
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export HEADS=28,4 SM_SPLIT=112 FUSED_REFERENCE_WORKSPACE_BYTES=1073741824
export TIME_AB=1 TIME_CUDA_GRAPH=1 TIME_ITERS=50 TIME_WARMUP=10 FUSED_FA3_CTA_Q=128
unset FLASHINFER_FUSED_PLAN_DEBUG
shapes=(
  "b_kv4096_r1_q8|MODE=band1 BAND_SIDE=fa2 BAND_KV=4096 BAND_ROWS=1 BAND_QO=8"
  "b_kv16384_r1_q4|MODE=band1 BAND_SIDE=fa2 BAND_KV=16384 BAND_ROWS=1 BAND_QO=4"
  "b_kv4096_r16_q8|MODE=band1 BAND_SIDE=fa2 BAND_KV=4096 BAND_ROWS=16 BAND_QO=8"
  "b_kv16384_r16_q8|MODE=band1 BAND_SIDE=fa2 BAND_KV=16384 BAND_ROWS=16 BAND_QO=8"
  "mixed_verify_q8|MODE=mixed_verify VERIFY_QO=8"
  "mixed_verify_q4|MODE=mixed_verify VERIFY_QO=4"
  "mix_decode32|MODE=mix_decode DEC_ROWS=32"
)
for chunk in 256 512 1024 2048; do
  for entry in "${shapes[@]}"; do
    label=c${chunk}_${entry%%|*}; envs=${entry#*|}
    env $envs FUSED_FA2_KV_CHUNK=$chunk timeout 900 $W/venv/bin/python -u $W/drivers/fused_reference.py > "$OUT/$label.log" 2>&1
    rc=$?
    echo "CK $label rc=$rc [$(grep -oE 'bad rows [0-9]+/[0-9]+|NON-FINITE|REPLAY CORRECTNESS: bad_heads=[0-9]+/[0-9]+' $OUT/$label.log | tr '\n' ' ')] $(grep -oE '^\[time\] (fused|unfused) += [0-9.]+|BatchAttention \(flash[^=]*= [0-9.]+' $OUT/$label.log | sed 's/(flashinfer.*from)//' | tr '\n' ' ')"
  done
done
echo COLAUNCH_CHUNK_DONE
