#!/bin/bash
# FA3 multicast step 2 (A5.5): KV exchange in stock flash-attention (fa3_kv_exchange.patch) -- 2-CTA clusters on paired
# m-blocks, each CTA gathers half of every K / V tile (head-dim columns) and bulk-copies it into the peer over DSMEM;
# cluster transaction pipelines; pair-aware varlen scheduler. Three stock_fa3_pdl builds from one tree: control, exchange
# with the smem-layout check, exchange. A6 (NP executor) FP32 + 20 graph replays on all 96 H200 captures with each
# exchange build (timeouts catch a deadlock), then stage 1 exchange vs control vs native v3. Claude, 2026-10-25.
set -u
W=/dev/shm/w; X=$W/fa3x
cd $W/ub
grep -q KvExchange $X/flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp || { echo STALE; exit 1; }
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16
for v in ${FA3X_BUILDS:-ctrl xchk xchg}; do
  case $v in
    ctrl) extra="" ;;
    xchk) extra="-DFLASHINFER_FA3_CLUSTER_M=2 -DFLASHINFER_FA3_KV_EXCHANGE=1 -DFLASHINFER_FA3_KV_EXCHANGE_CHECK=1" ;;
    xchg) extra="-DFLASHINFER_FA3_CLUSTER_M=2 -DFLASHINFER_FA3_KV_EXCHANGE=1" ;;
    xnop) extra="-DFLASHINFER_FA3_CLUSTER_M=2 -DFLASHINFER_FA3_KV_EXCHANGE=1 -DFLASHINFER_FA3_KV_EXCHANGE_NOPUSH=1" ;;
  esac
  rm -rf /opt/tiger/w/torch_ext_fa3x_$v/stock_fa3_pdl
  PYTHONPATH=$W/up/python:$X/stock_fa3_unsplit_work_20261014 STOCK_FLASH_ATTENTION=$X/flash-attention STOCK_FA3_UNSPLIT_SPECIALIZE=1 \
    STOCK_FA3_MERGED=/opt/tiger/w/stock_fa3_merged_fa3x_$v STOCK_FA3_EXTRA="$extra" TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_fa3x_$v \
    $W/venv/bin/python -u $X/stock_fa3_unsplit_work_20261014/load_stock_fa3_variant.py > build_fa3x_$v.log 2>&1
  [ -f /opt/tiger/w/torch_ext_fa3x_$v/stock_fa3_pdl/stock_fa3_pdl.so ] || { grep -E " error|error:" build_fa3x_$v.log | head -8; echo BUILD_FAILED $v; exit 1; }
  for f in $extra; do grep -q -- "$f" build_fa3x_$v.log || { echo FLAG_MISSING $v $f; exit 1; }; done
  echo "built fa3x $v"
done
export PYTHONPATH=$W/ub:$W/up/python:$W/fi CUDA_VISIBLE_DEVICES=0
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_SO=/opt/tiger/w/torch_ext_unified_h200_np_u/unified_dispatch_ext/unified_dispatch_ext.so
B=/opt/tiger/w/trace_sampled_h200_20261025; OUT=/opt/tiger/w/unified_h200_20261025
SETS="mix32_two mix32_four mix128_two mix128_four"
for v in ${FA3X_CHECKS:-xchk xchg}; do
  for d in $SETS; do
    NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_fa3x_$v/stock_fa3_pdl/stock_fa3_pdl.so UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 \
      timeout 900 $W/venv/bin/python -u check_unified_capture_20261024.py $B/$d/captures/*.pt > check_fa3x_${v}_$d.log 2>&1
    rc=$?
    echo "check $v $d rc=$rc: FAIL $(grep -v GRAPH_FAILS check_fa3x_${v}_$d.log | grep -c FAIL), PASS $(grep -c ' PASS' check_fa3x_${v}_$d.log), layout_bad $(grep -c LAYOUT_BAD check_fa3x_${v}_$d.log) $(grep -m1 -E 'Error|error' check_fa3x_${v}_$d.log | cut -c1-160)"
    [ $rc = 124 ] && { echo "HANG $v $d"; break 2; }
  done
done
for v in ${FA3X_BENCH:-ctrl xchg}; do
  for d in $SETS; do
    NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_fa3x_$v/stock_fa3_pdl/stock_fa3_pdl.so timeout 1800 \
      $W/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $OUT/$d/kernel_results.json --unified-only --append --label u_fa3x_$v 2>&1 | grep -E "FAIL|rror|Traceback"
  done
done
echo "control vs the serving A6 (same build, rebuilt FA3):"; $W/venv/bin/python analyze_trace_sampled_20261023.py u_fa3x_ctrl --baseline arm:u_h200_np_u $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
echo "exchange vs control:"; $W/venv/bin/python analyze_trace_sampled_20261023.py u_fa3x_xchg --baseline arm:u_fa3x_ctrl $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
echo "exchange vs native v3:"; $W/venv/bin/python analyze_trace_sampled_20261023.py u_fa3x_xchg --baseline arm:native_pdl_geometry_v3 $OUT/*/kernel_results.json | sed -E 's/ \|.*//'
echo "pairing without the exchange vs control:"; $W/venv/bin/python analyze_trace_sampled_20261023.py u_fa3x_xnop --baseline arm:u_fa3x_ctrl $OUT/*/kernel_results.json 2>&1 | sed -E 's/ \|.*//' | tail -4
echo FA3X_DONE
