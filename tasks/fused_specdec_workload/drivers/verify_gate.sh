#!/bin/bash
# Resolve the gate's weak margin: is the 4-mode split's early divergence from stock MTP a real
# numerical fault, or batch-composition reduction order? SGLANG_SPLIT_VERIFY=1 recomputes every
# row of every split step in fp32 from the paged cache and logs the worst relative error per side.
set -u
W=/dev/shm/w
export PYTHONPATH=$W/python:$W/fi HF_HOME=$W/hf FI_ROOT=$W/fi TORCH_CUDA_ARCH_LIST=9.0a CUDA_VISIBLE_DEVICES=0
ulimit -n $(ulimit -Hn)
O=$W/mn/verify; rm -rf $O; mkdir -p $O
export SGLANG_SPLIT_VERIFY=1
$W/venv/bin/python $W/tasks/fused_specdec_workload/drivers/srv_correctness.py MTPMIX_SPLIT $O > $O/run.log 2>&1
L=$O/server_MTPMIX_SPLIT.log
echo "mixed steps: $(grep -oE 'MIXED_SPEC steps=[0-9]+' $L | tail -1)"
echo "verify lines: $(grep -c SPLIT_VERIFY $L)"
grep -oE "fa3_max_rel=[0-9.e+-]+|fa2_max_rel=[0-9.e+-]+" $L | sort | uniq -c | sort -rn | head -6
echo "--- worst 3 rows:"
grep "SPLIT_VERIFY rows" $L | tail -3 | cut -c1-400
echo VERIFYDONE
