#!/bin/bash
# Build FA2-executor variants of the unified extension (2-class) for the occupancy sweep. Claude, 2026-10-24.
cd /dev/shm/w/ub
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a PYTHONPATH=/dev/shm/w/up/python:/dev/shm/w/fi MAX_JOBS=8 VERBOSE=1 UNIFIED_FA2_CLASSES=2
for v in ${VARIANTS:-"kv4 UNIFIED_FA2_MMA_KV=4" "wq1 UNIFIED_WQ32=1"}; do
  set -- $v; name=$1; shift
  rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
  env "$@" TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
  echo "== $name $(grep -E 'UNIFIED_BUILD_OK' build_$name.log | head -1 | cut -c1-40)"; grep -A3 "entry function '_Z16UnifiedFA2Kernel" build_$name.log | grep -E 'spill|registers'
done
echo VARIANTS_DONE
