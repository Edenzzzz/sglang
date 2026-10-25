#!/bin/bash
# Serving A6 library, 2026-10-26 (flashinfer port07 at 24c0545+: tile-rounded FA2 chunk default, decode-only
# instantiation opt-in, DSMEM v2 opt-in), the np64 configuration the H20 pods run (3 FA2 classes 64 / 32 / 16, NP
# executor with the in-launch merge, stock FA3). FP32 + 20 graph replays with the serving settings (FA3_MIN=65) on the
# 96 Qwen captures and the synthetic sets, stage 1 tile vs power-of-two rounding in this build, then a tarball for the
# pods. Claude, 2026-10-26.
set -u
cd /dev/shm/w/ub
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16 VERBOSE=1 UNIFIED_FA2_CLASSES=3 UNIFIED_FA2_DYNAMIC=1 UNIFIED_TILES=64,32,16 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
name=serving_np64_20261026
rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name \
  /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED; exit 1; }
for w in UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 UNIFIED_FA2_W0=64 UNIFIED_FA2_CLASSES=3; do grep nvcc build_$name.log | grep -q -- "-D$w" || { echo FLAG_MISSING $w; exit 1; }; done
SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
echo "built $name md5=$(md5sum $SO | cut -c1-12) src=$(md5sum port07_unified/include/flashinfer/attention/fused_plan.cuh port07_unified/tests/unified_dispatch_ext.cu | cut -c1-12 | tr '\n' ' ')"
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=65 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_SO=$SO
unset UNIFIED_MERGE_MODE UNIFIED_FA2_ORDER UNIFIED_PDL UNIFIED_NP_ONLY16 UNIFIED_FA2_CHUNK_ROUND UNIFIED_DSMEM_CAP_CHUNK
B=/opt/tiger/w/trace_sampled_h200_20261025; R=/opt/tiger/w/unified_h200_20261025; S=/opt/tiger/w/synth_meld_20261025
ok=1
for d in $B/mix32_two $B/mix32_four $B/mix128_two $B/mix128_four $S/sharegpt_two $S/sharegpt_four $S/longtail_four $S/sharedlong_four; do
  UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 timeout 2400 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $d/captures/*.pt > check_$name.$(basename $d).log 2>&1
  rc=$?; f=$(grep -v GRAPH_FAILS check_$name.$(basename $d).log | grep -c FAIL)
  echo "check $(basename $d) rc=$rc: FAIL $f, PASS $(grep -c ' PASS' check_$name.$(basename $d).log)"
  [ $rc != 0 ] || [ $f != 0 ] && ok=0
done
[ $ok = 1 ] || { echo SERVING_LIB_NOT_EXACT; exit 1; }
for d in mix32_two mix32_four mix128_two mix128_four; do
  for v in pow2 tile; do
    UNIFIED_FA2_CHUNK_ROUND=$v /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $R/$d/kernel_results.json --unified-only --append --label u_srv_$v 2>&1 | grep -E "FAIL|rror|Traceback"
  done
done
echo "serving lib, tile vs pow2:"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_srv_tile --baseline arm:u_srv_pow2 $R/*/kernel_results.json | sed -E 's/ \|.*//'
echo "serving lib vs native v3:"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_srv_tile --baseline arm:native_pdl_geometry_v3 $R/*/kernel_results.json | sed -E 's/ \|.*//'
mkdir -p /dev/shm/w/a6srv && cp $SO /dev/shm/w/a6srv/unified_dispatch_ext.so && cd /dev/shm/w/a6srv && tar czf /dev/shm/w/a6srv_20261026.tgz unified_dispatch_ext.so && echo "tarball md5=$(md5sum /dev/shm/w/a6srv_20261026.tgz | cut -c1-32)"
echo SERVING_LIB_DONE
