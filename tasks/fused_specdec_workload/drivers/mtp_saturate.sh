#!/bin/bash
# Capacity A/B: agent_chat_saturate (both tenants at infinite rate), 3 interleaved rounds.
# 4-mode MTP serving A/B: 3 interleaved rounds of
#   MTP_LPT (stock sglang spec decoding), MTPMIX_LPT (4-mode step, FA2), MTPMIX_SPLIT.
# Every arm must report 0 fallbacks; the MIXED_SPEC counter proves the 4-mode step ran.
set -u
W=/dev/shm/w
cd $W
[ -e $W/.venv ] || ln -s $W/venv $W/.venv
. $W/venv/bin/activate
export PYTHONPATH=$W/python:$W/fi HF_HOME=$W/hf FI_ROOT=$W/fi TORCH_CUDA_ARCH_LIST=9.0a
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export UP_STAGES=2 MMA_PV_RS=1 SCHED_LPT=0
rm -rf ~/.cache/flashinfer/90a/cached_ops
cd $W/tasks/realistic_workload_analysis
for r in 1 2 3; do
  for m in MTP_LPT MTPMIX_LPT MTPMIX_SPLIT; do
    o=$W/mtpsat/r${r}_$m
    rm -rf $o
    python run_mixed_workloads.py --scenario agent_chat_saturate --methods $m --outdir $o > $W/mtprep_r${r}_$m.log 2>&1
    L=$o/server_$m.log
    acc=$(grep -oE "accept len: [0-9.]+" $L | awk '{s+=$3;n++} END{if(n) printf "%.2f",s/n; else print "NA"}')
    echo "r$r $m: fallbacks=$(grep -ci fallback $L) tracebacks=$(grep -c Traceback $L) jsons=$(ls $o/*.json 2>/dev/null | wc -l) accept_len=$acc mixed=[$(grep -oE 'MIXED_SPEC steps=[0-9]+' $L | tail -1)]"
  done
done
python3 $W/repsum.py --root $W/mtpsat --base MTP_LPT
echo MTPREPSDONE
