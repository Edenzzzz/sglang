#!/bin/bash
# Production config: CUDA graphs ON. (1) q4 serving-output gate for stock MTP, unfused 4-mode and
# fused co-launch (64/32/16, auto FA2 chunk); stop on any failure. (2) realistic A/B with graphs on.
set -uo pipefail
W=/dev/shm/w
until grep -q RESTORE2_DONE $W/restore.log 2>/dev/null; do sleep 30; done
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/port07 HF_HOME=$W/hf
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112
unset FLASHINFER_FUSED_PLAN_DEBUG SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE DISABLE_CUDA_GRAPH

# Fresh-pod checks before any arm: the fork tree is what imports, RoPE theta is 1e6, and the
# rebuilt co-launch kernel reproduces the old pod's exact verdicts and timings.
F=$(PYTHONPATH=$W/validated_upstream/python $W/venv/bin/python -c "from sglang.srt.layers.attention import fused_fa3_loader as L; print(L.__file__)" 2>&1 | tail -1)
echo "LOADER $F"; case "$F" in /dev/shm/w/up/*|/dev/shm/w/validated_upstream/*) ;; *) echo "TREE CHECK FAILED"; exit 1;; esac
echo "ROPE $(CUDA_VISIBLE_DEVICES= SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 $W/venv/bin/python $W/ropecheck.py 2>&1 | grep -o 'get_rope_config: ([0-9.]*')"
PRE=$W/precheck_20261012; mkdir -p $PRE
for entry in "b_kv4096_r1_q8|MODE=band1 BAND_SIDE=fa2 BAND_KV=4096 BAND_ROWS=1 BAND_QO=8" \
             "mixed_verify_q8|MODE=mixed_verify VERIFY_QO=8" "mix_decode32|MODE=mix_decode DEC_ROWS=32"; do
  label=${entry%%|*}; envs=${entry#*|}
  env $envs HEADS=28,4 SM_SPLIT=112 FUSED_REFERENCE_WORKSPACE_BYTES=1073741824 TIME_AB=1 \
    TIME_CUDA_GRAPH=1 TIME_ITERS=50 TIME_WARMUP=10 FUSED_FA3_CTA_Q=128 \
    timeout 1500 $W/venv/bin/python -u $W/drivers/fused_reference.py > $PRE/$label.log 2>&1
  echo "PRE $label rc=$? [$(grep -oE 'bad rows [0-9]+/[0-9]+|REPLAY CORRECTNESS: bad_heads=[0-9]+/[0-9]+' $PRE/$label.log | tr '\n' ' ')] $(grep -oE '^\[time\] fused += [0-9.]+|BatchAttention \(flash[^=]*= [0-9.]+' $PRE/$label.log | sed 's/(flashinfer.*from)//' | tr '\n' ' ')"
done
if grep -lE "bad rows [1-9]|bad_heads=[1-9]" $PRE/*.log >/dev/null 2>&1 || grep -L "bad rows 0/" $PRE/*.log | grep -q .; then
  echo "PRECHECK FAILED"; exit 1; fi
echo PRECHECK_DONE

OUT=$W/q4_gate_graphs_20261012; mkdir -p $OUT
DRIVER=$W/validated_upstream/tasks/fused_specdec_workload/drivers/srv_correctness.py
( export CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
  export BENCH_CHUNK=512 CORRECTNESS_DECODE_TOKENS=1200 SGLANG_FUSED_DECODE_GUARD=0
  export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
  for arm in MTP_LPT_Q4G MTPMIX_LPT_Q4G MTPMIXFUSED_CL_Q4G; do
    echo "GATE_START $arm $(date +%T)"
    $W/venv/bin/python -u "$DRIVER" "$arm" "$OUT" 2>&1 | tail -3
    grep -c "cuda graph: True" /tmp/comp_$arm.jsonl 2>/dev/null
  done
  $W/venv/bin/python -u "$DRIVER" --compare "$OUT" MTP_LPT_Q4G MTPMIX_LPT_Q4G 2>&1 || exit 1
  $W/venv/bin/python -u "$DRIVER" --compare "$OUT" MTP_LPT_Q4G MTPMIXFUSED_CL_Q4G 2>&1 || exit 1
  $W/venv/bin/python -u "$DRIVER" --compare "$OUT" MTPMIX_LPT_Q4G MTPMIXFUSED_CL_Q4G 2>&1 || exit 1 )
gate=$?
echo "G_GATE_DONE status=$gate"
[ $gate = 0 ] || { echo "OUTPUT GATE FAILED -- stopping before serving"; exit 1; }

export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json CHAT_RATE=24 CODEX_RATE=0.1
cd $W/validated_upstream/tasks/realistic_workload_analysis
OUT=$W/graphs_agent_chat_rated24_20261012; mkdir -p $OUT
for arm in MTP_LPT MTPMIXFUSED MTPMIX_LPT; do
  echo "START $arm $(date +%T)"
  $W/venv/bin/python -u run_mixed_workloads.py --scenario agent_chat_rated --methods "$arm" --outdir "$OUT" 2>&1 | tail -8
  echo "END $arm rc=$? $(date +%T) graphs=$(grep -c 'cuda graph: True' $OUT/server_$arm.log) nographs=$(grep -c 'cuda graph: False' $OUT/server_$arm.log)"
done
echo GRAPHS_DONE
