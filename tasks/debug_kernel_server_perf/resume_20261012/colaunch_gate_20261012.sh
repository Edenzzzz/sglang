#!/bin/bash
# A: FA2 roles out of the fused kernel. FLASHINFER_FUSED_COLAUNCH=1 runs the FA3 grid alone on
# sm_split CTAs and launches FA2ExecutorKernel (same runners, same queue, native 128x2) as its
# programmatic dependent. Built in an isolated tree + extension cache so the serving tree's
# build is untouched. Every arm prints the fp32 causal-reference verdict before any timing.
set -uo pipefail
W=/dev/shm/w
OUT=$W/colaunch_gate_20261012
mkdir -p "$OUT"
until grep -q ROUND2_DONE $W/q4_gate_and_round2_20261012.log 2>/dev/null; do sleep 30; done
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/fi_colaunch
export TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch
export SGLANG_FUSED_FA2_TILES=48,32,16 FUSED_FA2_FINE_TAIL=1
export FUSED_CONST_PERS_PARAMS=1 FUSED_WORK_BUDGET_SPLIT=1
export FLASHINFER_FUSED_BACKFILL=1 FUSED_FA2_KV_CHUNK=1024
export HEADS=28,4 SM_SPLIT=112 FUSED_REFERENCE_WORKSPACE_BYTES=1073741824
export TIME_AB=1 TIME_CUDA_GRAPH=1 TIME_ITERS=50 TIME_WARMUP=10 FUSED_FA3_CTA_Q=128
unset FLASHINFER_FUSED_PLAN_DEBUG
sha256sum $W/fi_colaunch/include/flashinfer/attention/persistent_fa3_fused.cuh
run() {  # label, env assignments...
  local label=$1; shift
  env "$@" timeout 900 $W/venv/bin/python -u $W/drivers/fused_reference.py > "$OUT/$label.log" 2>&1
  local rc=$?
  echo "CL $label rc=$rc [$(grep -oE 'bad rows [0-9]+/[0-9]+|NON-FINITE|REPLAY CORRECTNESS: bad_heads=[0-9]+/[0-9]+' $OUT/$label.log | tr '\n' ' ')] $(grep -oE '^\[time\] (fused|unfused) += [0-9.]+|BatchAttention \(flash[^=]*= [0-9.]+' $OUT/$label.log | sed 's/(flashinfer.*from)//' | tr '\n' ' ')"
}
shapes=(
  "b_fa2_kv4096_r1_q8|MODE=band1 BAND_SIDE=fa2 BAND_KV=4096 BAND_ROWS=1 BAND_QO=8"
  "b_fa2_kv1024_r1_q1|MODE=band1 BAND_SIDE=fa2 BAND_KV=1024 BAND_ROWS=1 BAND_QO=1"
  "b_fa2_kv4096_r16_q4|MODE=band1 BAND_SIDE=fa2 BAND_KV=4096 BAND_ROWS=16 BAND_QO=4"
  "b_fa2_kv16384_r16_q8|MODE=band1 BAND_SIDE=fa2 BAND_KV=16384 BAND_ROWS=16 BAND_QO=8"
  "b_fa3_kv4096_r1_q8|MODE=band1 BAND_SIDE=fa3 BAND_KV=4096 BAND_ROWS=1 BAND_QO=8"
  "mixed_verify_q8|MODE=mixed_verify VERIFY_QO=8"
  "mixed_verify_q4|MODE=mixed_verify VERIFY_QO=4"
  "mix_decode32|MODE=mix_decode DEC_ROWS=32"
  "mixed|MODE=mixed"
  "fa3_split|MODE=fa3_split"
)
for entry in "${shapes[@]}"; do
  label=${entry%%|*}; envs=${entry#*|}
  run "${label}_cl0" $envs FLASHINFER_FUSED_COLAUNCH=0
  run "${label}_cl1" $envs FLASHINFER_FUSED_COLAUNCH=1
done
# Race screen: the co-launch is new concurrency, so repeat the mixed shapes.
for i in 1 2 3 4 5; do
  run "rep${i}_mixed_verify_q8_cl1" MODE=mixed_verify VERIFY_QO=8 FLASHINFER_FUSED_COLAUNCH=1
  run "rep${i}_mix_decode32_cl1" MODE=mix_decode DEC_ROWS=32 FLASHINFER_FUSED_COLAUNCH=1
done
for so in $(ls $TORCH_EXTENSIONS_DIR/*/*/*.so 2>/dev/null); do
  echo "== $so"
  /usr/local/cuda/bin/cuobjdump -res-usage $so 2>/dev/null | grep -A1 -E "Function .*(FusedFA3PersistentKernel|FA2ExecutorKernel)" \
    | grep -oE "Function _ZN10flashinfer[0-9]+[A-Za-z0-9]{0,30}|REG:[0-9]+|STACK:[0-9]+" | paste -sd' ' | sed 's/Function/\nF/g'
done
echo COLAUNCH_GATE_DONE
