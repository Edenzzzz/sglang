#!/bin/bash
# Ready-queue merge hang hunt (A6, H20): rebuild with a watchdog that prints the queue state and traps, then the full
# FP32 + graph check of mix32_two (where the first run stalled after 5 captures). Claude, 2026-10-25.
cd /dev/shm/w/ub
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
name=pers_ready
grep -q UNIFIED_READY_STUCK port07_unified/tests/unified_dispatch_ext.cu || { echo STALE; exit 1; }
rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
UNIFIED_FA2_NP=0 UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
grep -q UNIFIED_BUILD_OK build_$name.log || { echo BUILD_FAILED; exit 1; }
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry UNIFIED_MERGE_MODE=ready UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 timeout 1200 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py /opt/tiger/w/trace_sampled_h20_20261024/mix32_two/captures/*.pt > check_pready_dbg.log 2>&1
echo "rc=$? PASS $(grep -c ' PASS' check_pready_dbg.log)"
grep -m5 -E "STUCK|Error|error" check_pready_dbg.log | cut -c1-200
echo DBG_DONE
