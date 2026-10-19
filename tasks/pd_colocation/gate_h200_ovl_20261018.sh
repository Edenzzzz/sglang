#!/bin/bash
# Serving-output gate on the re-forked H200 (dadf94f0e1da7238) for the FA2-overlapped split arms of
# mn_h200_nospec_20261018.sh. No MTP: LPT (stock chunked prefill + mixed chunk, no spec), LPT_REP (noise floor),
# SPLIT_OVL_EDF (ours: FA3/FA2 split with FA2 overlapped, EDF + chunk preemption, uncached deadlines, deadline chunk
# 2048/8192, chunk 8192), each vs LPT. MTP: MTP_MIX_SPLIT_OVL_EDF (4-mode step, overlapped split, the best MTP setting)
# vs the MTP reference of gate_h200_fourmode_20261018 (copied in). Claude, 2026-10-18.
set -uo pipefail
W=/dev/shm/w
CUDART=$(dirname "$(find $(readlink -f $W/venv) -name 'libcudart.so.12*' 2>/dev/null | head -1)")
export PYTHONPATH=$W/python:$W/fi HF_HOME=$W/hf FI_ROOT=$W/fi TORCH_CUDA_ARCH_LIST=9.0a CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$CUDART:${LD_LIBRARY_PATH:-}
out=$W/gate_h200_ovl_20261018; rm -rf $out; mkdir -p $out
cp $W/gate_h200_fourmode_20261018/MTP.json $out/
D=$W/tasks/fused_specdec_workload/drivers/srv_correctness.py
P=$W/venv/bin/python
split_env() {
  export SGLANG_MIXED_ATTENTION_KERNEL=persistent SGLANG_TWO_EXEC=1 SGLANG_FUSED_FA3_SM_SPLIT=1
  export FLASHINFER_SM_PARTITION=1 UP_STAGES=2 SGLANG_FUSED_FA3_SRC=$W/fi/tests/fused_fa3_persistent_ext.cu
  export SGLANG_FUSED_FI_ROOT=$W/port07 SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1
  export FUSED_CONST_PERS_PARAMS=1 FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1
  export FLASHINFER_FUSED_COLAUNCH=1 FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112
  export SGLANG_SPLIT_OVERLAP=1 SGLANG_FUSED_DECODE_GUARD=0
  export SGLANG_EDF_PREEMPT_CHUNK=1 SGLANG_EDF_RESERVE_EARLIER=1 SGLANG_EDF_PREEMPT_MAX_TOKENS=1000000000
  export SGLANG_EDF_UNCACHED_DEADLINE=1
}
$P $D LPT $out 2>&1 | tail -1
$P $D LPT_REP $out 2>&1 | tail -1
( split_env; export SGLANG_EDF_DEADLINE_CHUNK=2048:8192:6000
  SRV_ARGS="--chunked-prefill-size 8192" $P $D SPLIT_OVL_EDF $out 2>&1 | tail -1 )
( split_env; export SGLANG_EDF_DEADLINE_CHUNK=1024:2048:6000 SGLANG_MIXED_SPEC_MIN_CTX=8192 SGLANG_MIXED_SPEC_MIN_QUEUE=4
  SRV_ARGS="--chunked-prefill-size 2048" $P $D MTP_MIX_SPLIT_OVL_EDF $out 2>&1 | tail -1 )
for arm in LPT_REP SPLIT_OVL_EDF; do echo "== $arm vs LPT"; $P $D --compare $out LPT $arm 2>&1 | grep -E "vs |GATE"; done
echo "== MTP_MIX_SPLIT_OVL_EDF vs MTP"; $P $D --compare $out MTP MTP_MIX_SPLIT_OVL_EDF 2>&1 | grep -E "vs |GATE"
for a in SPLIT_OVL_EDF MTP_MIX_SPLIT_OVL_EDF; do L=$out/server_$a.log
  echo "GATE_H200 $a split=$(grep FUSED_SPLIT_CALLS $L | tail -1 | grep -o 'CALLS [0-9]*.*overlapped=[0-9]*') tracebacks=$(grep -c Traceback $L) fallbacks=$(grep -ci fallback $L)"; done
echo GATE_H200_DONE
