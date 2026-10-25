#!/bin/bash
# Plan shape + FA2 balance of the 96 H200 captures (plan_balance_20261025.py): default plan on the NP timeline build
# (splits, items, span vs balance bounds), then the dsmem plan's chunk inflation at C = 2 / 4 / 8 / 16 (stats only).
# Waits for the DSMEM run to free the GPU. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
until grep -q DSMEM_JIT_DONE a6_dsmem.log 2>/dev/null; do sleep 60; done
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3
B=/opt/tiger/w/trace_sampled_h200_20261025
SETS="mix32_two mix32_four mix128_two mix128_four"
unset UNIFIED_MERGE_MODE
for d in $SETS; do
  UNIFIED_SO=/opt/tiger/w/torch_ext_unified_h200_np_tl/unified_dispatch_ext/unified_dispatch_ext.so \
    timeout 1800 /dev/shm/w/venv/bin/python -u plan_balance_20261025.py $B/$d/captures > pb_default_$d.log 2>&1
  echo "default $d: $(tail -1 pb_default_$d.log)"
done
for c in 2 4 8 16; do
  for d in $SETS; do
    UNIFIED_MERGE_MODE=dsmem UNIFIED_DSMEM_CLUSTER=$c PLAN_BALANCE_TIMELINE=0 \
    UNIFIED_SO=/opt/tiger/w/torch_ext_unified_h200_np_dsmem/unified_dispatch_ext/unified_dispatch_ext.so \
      timeout 1800 /dev/shm/w/venv/bin/python -u plan_balance_20261025.py $B/$d/captures > pb_dsmem_c${c}_$d.log 2>&1
    echo "dsmem C=$c $d: $(tail -1 pb_dsmem_c${c}_$d.log)"
  done
done
echo PLAN_BALANCE_DONE
