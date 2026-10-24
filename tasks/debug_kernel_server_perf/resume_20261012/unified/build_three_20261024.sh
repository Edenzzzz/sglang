#!/bin/bash
# Build the three FA2 executors of the unified plan (A6), checking every switch on the nvcc line. Claude, 2026-10-24.
cd /dev/shm/w/ub
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=${MAX_JOBS:-4} VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1
export PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
for v in "np_last UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=0" "np2 UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1" "fa3items UNIFIED_FA2_NP=0"; do
  set -- $v; name=$1; shift
  rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
  env "$@" TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name /dev/shm/w/venv/bin/python -u \
    build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
  ok=$(grep -c UNIFIED_BUILD_OK build_$name.log)
  for want in $(echo "$@" | grep -oE 'UNIFIED_[A-Z0-9_]+=[0-9]+'); do
    grep -q -- "-D$want" build_$name.log || { echo "FLAG_MISSING $name $want"; ok=0; }
  done
  echo "build $name ok=$ok"; grep -E " error" build_$name.log | head -3
done
echo BUILD3_DONE
