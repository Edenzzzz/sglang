#!/bin/bash
# H20 pod 4: serving A6 library from port07_srv (same source md5 as the H200 serving build: fused_plan.cuh ac0ce78c83a1,
# unified_dispatch_ext.cu a7e28a43983e), np64 configuration; FP32 + 20 graph replays on the 43 GLM-4.5-Air captures with
# FA3_MIN=65, stage 1 vs native v3, then the library staged as /dev/shm/w/a6srv/unified_dispatch_ext.so. Waits for the
# Llama stage-2 run on this pod. Claude, 2026-10-26.
set -u
until grep -qE 'A6_STAGE2_LLAMA_DONE|A6_NOT_ACTIVE|Traceback' /dev/shm/w/a6_stage2_llama_over_nodc.log 2>/dev/null; do sleep 60; done
sleep 30
cd /dev/shm/w/ub
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16 VERBOSE=1 UNIFIED_FA2_CLASSES=3 UNIFIED_FA2_DYNAMIC=1 UNIFIED_TILES=64,32,16 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
name=srv_np64_20261026
rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name \
  /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_srv > build_$name.log 2>&1
grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED; exit 1; }
for w in UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 UNIFIED_FA2_W0=64 UNIFIED_FA2_CLASSES=3; do grep nvcc build_$name.log | grep -q -- "-D$w" || { echo FLAG_MISSING $w; exit 1; }; done
SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
echo "built $name md5=$(md5sum $SO | cut -c1-12)"
A=/opt/tiger/w/a6lib
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=$A/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=$A/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=$A/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=$A/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=65 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_SO=$SO
B=/opt/tiger/w/trace_sampled_glm_h20_20261025; OUT=/opt/tiger/w/srv_gate_glm_20261026; mkdir -p $OUT
ok=1
for d in $(cd $B && ls -d */ | tr -d /); do
  [ -d $B/$d/captures ] || continue
  UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 timeout 2400 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_srv_glm_$d.log 2>&1
  rc=$?; f=$(grep -v GRAPH_FAILS check_srv_glm_$d.log | grep -c FAIL)
  echo "check GLM $d rc=$rc: FAIL $f, PASS $(grep -c ' PASS' check_srv_glm_$d.log)"
  [ $rc != 0 ] || [ $f != 0 ] && ok=0
  mkdir -p $OUT/$d
  /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json --label u_srv 2>&1 | grep -E "FAIL|rror|Traceback"
done
[ $ok = 1 ] || { echo SRV_NOT_EXACT; exit 1; }
echo "GLM stage 1, serving lib vs native v3:"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_srv --baseline arm:native_pdl_geometry_v3 $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
mkdir -p /dev/shm/w/a6srv && cp $SO /dev/shm/w/a6srv/unified_dispatch_ext.so && echo "staged md5=$(md5sum /dev/shm/w/a6srv/unified_dispatch_ext.so | cut -c1-12)"
echo SRV_BUILD_DONE
