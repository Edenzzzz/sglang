#!/bin/bash
# Co-launch + classes 64/32/16 + FUSED_FA2_KV_CHUNK=auto: (1) kernel gate of the auto chunk against
# the fixed chunks, incl. shapes outside the fit; (2) serving-output gate at q4; (3) realistic A/B
# (agent_chat_rated, chat 24/s + codex 0.1/s): fused co-launch x2 rounds vs the unfused 4-mode arm.
set -uo pipefail
W=/dev/shm/w
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/fi_colaunch HF_HOME=$W/hf
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export SGLANG_FUSED_SM_FA3=112
unset FLASHINFER_FUSED_PLAN_DEBUG SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE
sha256sum $W/fi_colaunch/include/flashinfer/attention/{fused_plan.cuh,persistent_fa3_fused.cuh}

# (1) kernel gate
OUT=$W/colaunch_auto2_20261012; mkdir -p $OUT
shapes=(
  "b_kv4096_r1_q8|MODE=band1 BAND_SIDE=fa2 BAND_KV=4096 BAND_ROWS=1 BAND_QO=8"
  "b_kv16384_r1_q4|MODE=band1 BAND_SIDE=fa2 BAND_KV=16384 BAND_ROWS=1 BAND_QO=4"
  "b_kv4096_r16_q8|MODE=band1 BAND_SIDE=fa2 BAND_KV=4096 BAND_ROWS=16 BAND_QO=8"
  "b_kv16384_r16_q8|MODE=band1 BAND_SIDE=fa2 BAND_KV=16384 BAND_ROWS=16 BAND_QO=8"
  "b_kv32768_r1_q4|MODE=band1 BAND_SIDE=fa2 BAND_KV=32768 BAND_ROWS=1 BAND_QO=4"
  "b_kv8192_r32_q4|MODE=band1 BAND_SIDE=fa2 BAND_KV=8192 BAND_ROWS=32 BAND_QO=4"
  "mixed_verify_q8|MODE=mixed_verify VERIFY_QO=8"
  "mixed_verify_q4|MODE=mixed_verify VERIFY_QO=4"
  "mixed_verify_q4_r32_kv8192|MODE=mixed_verify VERIFY_QO=4 VERIFY_ROWS=32 VERIFY_KV=8192"
  "mix_decode32|MODE=mix_decode DEC_ROWS=32"
)
for chunk in auto; do
  for entry in "${shapes[@]}"; do
    label=c${chunk}_${entry%%|*}; envs=${entry#*|}
    env $envs FUSED_FA2_KV_CHUNK=$chunk HEADS=28,4 SM_SPLIT=112 FUSED_REFERENCE_WORKSPACE_BYTES=1073741824 \
      TIME_AB=1 TIME_CUDA_GRAPH=1 TIME_ITERS=50 TIME_WARMUP=10 FUSED_FA3_CTA_Q=128 \
      timeout 900 $W/venv/bin/python -u $W/drivers/fused_reference.py > "$OUT/$label.log" 2>&1
    rc=$?
    echo "AU $label rc=$rc [$(grep -oE 'bad rows [0-9]+/[0-9]+|NON-FINITE|REPLAY CORRECTNESS: bad_heads=[0-9]+/[0-9]+' $OUT/$label.log | tr '\n' ' ')] $(grep -oE '^\[time\] (fused|unfused) += [0-9.]+|BatchAttention \(flash[^=]*= [0-9.]+' $OUT/$label.log | sed 's/(flashinfer.*from)//' | tr '\n' ' ')"
  done
done
echo AUTO_KERNEL_DONE
if grep -lE "bad rows [1-9]|NON-FINITE|bad_heads=[1-9]" $OUT/*.log >/dev/null 2>&1 || grep -L "bad rows 0/" $OUT/*.log | grep -q .; then
  echo "KERNEL GATE FAILED -- stopping before serving"; exit 1; fi
export FUSED_FA2_KV_CHUNK=auto

# (2) q4 serving-output gate, reusing the 10:18 stock/unfused baselines
OUT=$W/q4_gate_colaunch2_20261012; mkdir -p $OUT
cp $W/q4_gate_20261012/MTP_LPT_Q4.json $W/q4_gate_20261012/MTPMIX_LPT_Q4.json $OUT/
DRIVER=$W/validated_upstream/tasks/fused_specdec_workload/drivers/srv_correctness.py
( export CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET=$W/datasets/ShareGPT.original.json
  export BENCH_CHUNK=512 CORRECTNESS_DECODE_TOKENS=1200 SGLANG_FUSED_DECODE_GUARD=0 DISABLE_CUDA_GRAPH=1
  export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
  echo "GATE_START MTPMIXFUSED_CL_Q4 $(date +%T)"
  $W/venv/bin/python -u "$DRIVER" MTPMIXFUSED_CL_Q4 "$OUT" 2>&1 | tail -3
  $W/venv/bin/python -u "$DRIVER" --compare "$OUT" MTP_LPT_Q4 MTPMIXFUSED_CL_Q4 2>&1 || exit 1
  $W/venv/bin/python -u "$DRIVER" --compare "$OUT" MTPMIX_LPT_Q4 MTPMIXFUSED_CL_Q4 2>&1 || exit 1 )
gate=$?
echo "CL_GATE_DONE status=$gate"
[ $gate = 0 ] || { echo "OUTPUT GATE FAILED -- stopping before serving"; exit 1; }

# (3) realistic A/B
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/datasets/ShareGPT.original.json CHAT_RATE=24 CODEX_RATE=0.1 DISABLE_CUDA_GRAPH=1
cd $W/validated_upstream/tasks/realistic_workload_analysis
for r in 1 2; do
  OUT=$W/colaunch_agent_chat_rated24_r${r}_v2_20261012; mkdir -p $OUT
  arms="MTPMIXFUSED"; [ $r = 1 ] && arms="MTPMIXFUSED MTPMIX_LPT"
  for arm in $arms; do
    echo "START r$r $arm $(date +%T)"
    $W/venv/bin/python -u run_mixed_workloads.py --scenario agent_chat_rated --methods "$arm" --outdir "$OUT" 2>&1 | tail -8
    echo "END r$r $arm rc=$? $(date +%T)"
  done
done
echo COLAUNCH_SERVING_DONE
