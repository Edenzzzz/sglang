#!/bin/bash
# A6 with a 64-row FA2 class in the NP executor (v3's BatchPrefill 64-row tile, 4q x 1kv, NUM_MMA_KV 8), so rows of
# 33-64 packed queries (GLM-4.5-Air's 4-token verify rows pack 48) go to FA2 as in v3 instead of FA3. Plans with 64-row
# items launch a separate instantiation. GLM stage 1 (43 captures) and a Qwen2.5-7B regression check, FP32 + 20 graph
# replays on every capture. Claude, 2026-10-25.
set -u
cd /dev/shm/w/ub
grep -q "bool HAS64" port07_unified/tests/unified_dispatch_ext.cu || { echo STALE; exit 1; }
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8 VERBOSE=1 UNIFIED_FA2_CLASSES=3 UNIFIED_FA2_DYNAMIC=1 UNIFIED_TILES=64,32,16 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
name=np64
rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED; exit 1; }
for w in UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 UNIFIED_FA2_W0=64 UNIFIED_FA2_CLASSES=3; do grep nvcc build_$name.log | grep -q -- "-D$w" || { echo FLAG_MISSING $w; exit 1; }; done
echo "built: $(grep -A2 'Compiling entry function .*UnifiedFA2NPKernel' build_$name.log | grep -E 'registers|spill' | tr '\n' ' ' | cut -c1-300)"
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=65 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
for spec in "glm /opt/tiger/w/trace_sampled_glm_h20_20261025 /opt/tiger/w/unified_glm_h20_20261025 mix32_two mix32_four" \
            "qwen /opt/tiger/w/trace_sampled_h20_20261024 /opt/tiger/w/unified_h20_20261024 mix32_two mix32_four mix128_two mix128_four"; do
  set -- $spec; model=$1; B=$2; OUT=$3; shift 3
  for d in "$@"; do
    UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 timeout 2400 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_np64_${model}_$d.log 2>&1
    echo "check $model $d: FAIL $(grep -v GRAPH_FAILS check_np64_${model}_$d.log | grep -c FAIL), PASS $(grep -c ' PASS' check_np64_${model}_$d.log), items $(grep -o 'items48/32/16/fa3=[0-9/]*' check_np64_${model}_$d.log | head -2 | tr '\n' ' ')"
    /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json --unified-only --append --label u_np64 2>&1 | grep -E "FAIL|rror|Traceback"
  done
  for b in arm:native_pdl_geometry_v3 default; do
    /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_np64 --baseline $b $OUT/*/kernel_results.json | sed -E "s/ \|.*//; s/^/$model /"
  done
done
echo NP64_DONE
