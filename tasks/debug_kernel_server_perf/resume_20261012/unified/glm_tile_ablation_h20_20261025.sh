#!/bin/bash
# GLM-4.5-Air rows of 33-64 packed queries (4-token verify rows pack 48): which executor and tile (A6, H20 pod 4, rank-0
# attention of the 43 trace-sampled TP4 captures). Arms on the same captures:
#   fa3_128  rows on FA3, q tile 128 (upstream's tile for head dim 128)    UNIFIED_FA3_MIN=33, stock FA3 built BLOCK_M=128
#   fa3_64   rows on FA3, q tile 64 (one consumer warpgroup)                UNIFIED_FA3_MIN=33, stock FA3 built BLOCK_M=64
#   fa2_64   rows on FA2's 64-row class (v3's BatchPrefill tile)             UNIFIED_FA3_MIN=65, build W0=64
#   fa2_48   rows on FA2's 48-row class (1q x 4kv, 3 MMAs per warp)          UNIFIED_FA3_MIN=49, build W0=48
# (1) the verify rows alone (rows_route_ablation_20261025.py), (2) FP32 + 20 graph replays and stage 1 vs native v3 for
# each arm. In stage 1 the fa3_64 build also runs the prefill rows at q tile 64 (one FA3 launch, one tile size).
# Claude, 2026-10-25.
set -u
W=/dev/shm/w; U=$W/ub
mkdir -p $U && cd $U
[ -f bench_captured_mixed_20261023.py ] || tar xzf $W/glmabl/ub_scripts.tgz -C $U
cp $W/glmabl/repo/*.py $U/
[ -d $U/port07_unified ] || tar xzf $W/glmabl/port07_unified.tgz -C $U
grep -q "using NPK48" $U/port07_unified/tests/unified_dispatch_ext.cu || { echo STALE; exit 1; }
[ -d /opt/tiger/w/trace_sampled_glm_h20_20261025 ] || tar xf $W/glmabl/glm_traces.tar -C /opt/tiger/w
mkdir -p $W/fa3bm && cd $W/fa3bm && { [ -d flash-attention ] || tar xzf $W/glmabl/flash_attention.tgz; } && { [ -d stock_fa3_unsplit_work_20261014 ] || tar xzf $W/glmabl/fa3bm.tgz; }
grep -q FLASHINFER_FA3_BLOCK_M $W/fa3bm/stock_fa3_unsplit_work_20261014/port07/include/flashinfer/attention/hopper/fa3_upstream.cuh || { echo STALE_FA3; exit 1; }
cd $U
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16
for bm in 128 64; do
  [ -f /opt/tiger/w/torch_ext_fa3_bm$bm/stock_fa3_pdl/stock_fa3_pdl.so ] && continue
  PYTHONPATH=$W/up/python:$W/fa3bm/stock_fa3_unsplit_work_20261014 STOCK_FLASH_ATTENTION=$W/fa3bm/flash-attention STOCK_FA3_UNSPLIT_SPECIALIZE=1 \
    STOCK_FA3_MERGED=/opt/tiger/w/stock_fa3_merged_bm$bm STOCK_FA3_EXTRA="-DFLASHINFER_FA3_BLOCK_M=$bm" TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_fa3_bm$bm \
    $W/venv/bin/python -u $W/fa3bm/stock_fa3_unsplit_work_20261014/load_stock_fa3_variant.py > build_fa3_bm$bm.log 2>&1
  [ -f /opt/tiger/w/torch_ext_fa3_bm$bm/stock_fa3_pdl/stock_fa3_pdl.so ] || { grep -E " error|Error" build_fa3_bm$bm.log | head -5; echo BUILD_FAILED fa3_bm$bm; exit 1; }
  grep -q -- "-DFLASHINFER_FA3_BLOCK_M=$bm" build_fa3_bm$bm.log || { echo FLAG_MISSING fa3_bm$bm; exit 1; }
  echo "built fa3 bm$bm"
done
export VERBOSE=1 UNIFIED_FA2_CLASSES=3 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=$U:$W/up/python:$W/fi
for w0 in 64 48; do
  name=glm_np$w0
  [ -f /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so ] && continue
  UNIFIED_TILES=$w0,32,16 UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name \
    $W/venv/bin/python -u build_unified_h200_20261024.py $U/port07_unified > build_$name.log 2>&1
  grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED $name; exit 1; }
  for f in UNIFIED_FA2_W0=$w0 UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 UNIFIED_FA2_CLASSES=3; do grep nvcc build_$name.log | grep -q -- "-D$f" || { echo FLAG_MISSING $name $f; exit 1; }; done
  echo "built $name: $(grep -A2 'Compiling entry function .*UnifiedFA2NPKernel' build_$name.log | grep -E 'registers|spill' | tr '\n' ' ' | cut -c1-260)"
done
A=/opt/tiger/w/a6lib
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=$A/native_fa2_q32.so NATIVE_PDL_FA2_FP32_LIBRARY=$A/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=$A/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_SPLIT_POLICY=geometry_fa3
B=/opt/tiger/w/trace_sampled_glm_h20_20261025; OUT=/opt/tiger/w/unified_glm_h20_20261025
# arm: label so fa3lib fa3_min
ARMS="fa3_128:glm_np64:bm128:33 fa3_64:glm_np64:bm64:33 fa2_64:glm_np64:bm128:65 fa2_48:glm_np48:bm128:49"
for a in $ARMS; do
  IFS=: read -r label so bm fmin <<< "$a"
  export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$so/unified_dispatch_ext/unified_dispatch_ext.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_fa3_bm$bm/stock_fa3_pdl/stock_fa3_pdl.so
  rf=$fmin; [ $label = fa3_128 ] || [ $label = fa3_64 ] && rf=1; [ $label = fa2_64 ] || [ $label = fa2_48 ] && rf=1073741824
  ROUTE_FA3_MIN=$rf timeout 2400 $W/venv/bin/python -u rows_route_ablation_20261025.py $label $B/mix32_four/captures $B/mix32_two/captures > route_$label.log 2>&1
  echo "rows $label: $(grep -E 'ROUTE_TOTAL|Error' route_$label.log | tail -1)"
  for d in mix32_two mix32_four; do
    UNIFIED_FA3_MIN=$fmin UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 timeout 2400 $W/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_tile_${label}_$d.log 2>&1
    echo "check $label $d: FAIL $(grep -v GRAPH_FAILS check_tile_${label}_$d.log | grep -c FAIL), PASS $(grep -c ' PASS' check_tile_${label}_$d.log)"
    UNIFIED_FA3_MIN=$fmin $W/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json --unified-only --append --label u_tile_$label 2>&1 | grep -E "FAIL|rror|Traceback"
  done
  echo "stage1 $label vs v3:"; $W/venv/bin/python analyze_trace_sampled_20261023.py u_tile_$label --baseline arm:native_pdl_geometry_v3 $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
done
echo GLM_TILE_DONE
