#!/bin/bash
# Serving correctness gate on one H20 GPU before any MTP multi-node arm: greedy outputs of the
# 4-mode split vs stock MTP, with a second stock run as the noise floor
# (tasks/fused_specdec_workload/drivers/srv_correctness.py). Prints GATE PASS/FAIL lines.
set -u
W=/dev/shm/w
export PYTHONPATH=$W/python:$W/fi HF_HOME=$W/hf FI_ROOT=$W/fi TORCH_CUDA_ARCH_LIST=9.0a CUDA_VISIBLE_DEVICES=0
ulimit -n $(ulimit -Hn)
O=$W/mn/gate; rm -rf $O; mkdir -p $O
G=$W/tasks/fused_specdec_workload/drivers/srv_correctness.py
for a in MTP_LPT MTP_LPT2 MTPMIX_SPLIT; do
  $W/venv/bin/python $G $a $O > $O/$a.log 2>&1
  echo "$a: $(tail -1 $O/$a.log | cut -c1-150) fallbacks=$(grep -ci fallback $O/server_$a.log 2>/dev/null)"
done
$W/venv/bin/python $G --compare $O MTP_LPT MTP_LPT2
$W/venv/bin/python $G --compare $O MTP_LPT MTPMIX_SPLIT
echo H20GATEDONE
