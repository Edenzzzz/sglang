#!/bin/bash
# Serving-output gate on the re-forked H200 (d6a80befdd4e4da5) for the best 4-mode MTP serving config of
# mn_h200_fourmode_20261018.sh: KERNEL=mtpsplit (FA3 on 112 SMs + FA2 verify rows), EDF + chunk preemption, uncached
# deadlines, deadline chunk 1024/2048:6000, chunk 2048, mix gate 8K ctx / 4 queued. srv_correctness arms: MTP (stock
# MTP reference), MTP_REP (noise floor), MTP_C2048 (null case: stock MTP at the config's chunk size), MTP_MIX_SPLIT_EDF
# (the config); each vs MTP. Claude, 2026-10-18.
# Checkpoint-every variant: the config arm + SGLANG_CHECKPOINT_EVERY_TOKENS=8192, vs the MTP reference already in $out.
set -uo pipefail
W=/dev/shm/w
CUDART=$(dirname "$(find $(readlink -f $W/venv) -name 'libcudart.so.12*' 2>/dev/null | head -1)")
export PYTHONPATH=$W/python:$W/fi HF_HOME=$W/hf FI_ROOT=$W/fi TORCH_CUDA_ARCH_LIST=9.0a CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$CUDART:${LD_LIBRARY_PATH:-}
out=$W/gate_h200_fourmode_20261018
D=$W/tasks/fused_specdec_workload/drivers/srv_correctness.py
P=$W/venv/bin/python
( export SGLANG_MIXED_ATTENTION_KERNEL=persistent SGLANG_TWO_EXEC=1 SGLANG_FUSED_FA3_SM_SPLIT=1
  export FLASHINFER_SM_PARTITION=1 UP_STAGES=2 SGLANG_FUSED_FA3_SRC=$W/fi/tests/fused_fa3_persistent_ext.cu
  export SGLANG_FUSED_FI_ROOT=$W/port07 SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1
  export FUSED_CONST_PERS_PARAMS=1 FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1
  export FLASHINFER_FUSED_COLAUNCH=1 FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112
  export SGLANG_SPLIT_OVERLAP=0 SGLANG_FUSED_DECODE_GUARD=0
  export SGLANG_EDF_PREEMPT_CHUNK=1 SGLANG_EDF_RESERVE_EARLIER=1 SGLANG_EDF_PREEMPT_MAX_TOKENS=1000000000
  export SGLANG_EDF_UNCACHED_DEADLINE=1 SGLANG_EDF_DEADLINE_CHUNK=1024:2048:6000
  export SGLANG_MIXED_SPEC_MIN_CTX=8192 SGLANG_MIXED_SPEC_MIN_QUEUE=4 SGLANG_CHECKPOINT_EVERY_TOKENS=8192
  SRV_ARGS="--chunked-prefill-size 2048" $P $D MTP_MIX_SPLIT_CK_EDF $out 2>&1 | tail -2 )
for arm in MTP_MIX_SPLIT_CK_EDF; do echo "== $arm vs MTP"; $P $D --compare $out MTP $arm 2>&1; done
L=$out/server_MTP_MIX_SPLIT_CK_EDF.log
echo "GATE_H200 ckpt_skipped=$(grep CHECKPOINT_EVERY $L | tail -1 | grep -o 'skipped=[0-9]*') split=$(grep FUSED_SPLIT_CALLS $L | tail -1 | grep -o 'CALLS [0-9]*') tracebacks=$(grep -c Traceback $L) priority=$(grep -c enable_priority_scheduling.:.True $L) fused_build=$(grep -ci 'fused_fa3\|load_fused' $L) mixed_spec=$(grep -c 'MIXED_SPEC steps' $L)"
echo GATE_H200_DONE
