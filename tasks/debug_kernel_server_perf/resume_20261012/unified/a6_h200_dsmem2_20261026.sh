#!/bin/bash
# DSMEM-cluster reduction, second design (flashinfer: split count from the normal chunk policy; groups of <= C members
# reduce inside their cluster after one cluster barrier, larger groups are reduced by their last member through the
# global counter, no waits), H200: FP32 + 20 graph replays on captures and synthetic sets, then stage 1 vs the ticketed
# merge items in the same build, on captures and synthetic sets, C = 4 / 8 / 16. Claude, 2026-10-26.
set -u
cd /dev/shm/w/ub
grep -q 'UNIFIED_DSMEM_CAP_CHUNK' port07_unified/include/flashinfer/attention/fused_plan.cuh || { echo STALE; exit 1; }
grep -q 's_last_member' port07_unified/tests/unified_dispatch_ext.cu || { echo STALE; exit 1; }
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16 VERBOSE=1 UNIFIED_FA2_CLASSES=2 UNIFIED_FA2_DYNAMIC=1 PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi
name=h200_np_dsmem2
rm -rf /opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext
UNIFIED_FA2_NP=1 UNIFIED_NP_MERGE=1 UNIFIED_FA3_STOCK=1 TORCH_EXTENSIONS_DIR=/opt/tiger/w/torch_ext_unified_$name \
  /dev/shm/w/venv/bin/python -u build_unified_h200_20261024.py /dev/shm/w/ub/port07_unified > build_$name.log 2>&1
grep -q UNIFIED_BUILD_OK build_$name.log || { grep " error" build_$name.log | head; echo BUILD_FAILED; exit 1; }
echo "built $name: cluster kernel $(grep -A2 'Function properties for _Z25UnifiedFA2NPClusterKernelILb0E' build_$name.log | grep spill | head -1)"
export CUDA_VISIBLE_DEVICES=0 NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3
unset UNIFIED_MERGE_MODE UNIFIED_FA2_ORDER UNIFIED_PDL UNIFIED_NP_ONLY16 UNIFIED_FA2_CHUNK_ROUND UNIFIED_DSMEM_CAP_CHUNK
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_$name/unified_dispatch_ext/unified_dispatch_ext.so
S=/opt/tiger/w/synth_meld_20261025; OUT=/opt/tiger/w/synth_meld_results_20261025
B=/opt/tiger/w/trace_sampled_h200_20261025; R=/opt/tiger/w/unified_h200_20261025
SYN="sharegpt_two sharegpt_four longtail_four longtail_two math_two sharedprefix_two"
ok=1
for c in 4 8; do
  for d in $B/mix32_two $B/mix32_four $B/mix128_two $B/mix128_four $(for x in $SYN; do echo $S/$x; done); do
    UNIFIED_MERGE_MODE=dsmem UNIFIED_DSMEM_CLUSTER=$c UNIFIED_GRAPH=1 UNIFIED_REPLAYS=20 timeout 2400 /dev/shm/w/venv/bin/python -u check_unified_capture_20261024.py $d/captures/*.pt > check_dsmem2_c${c}_$(basename $d).log 2>&1
    rc=$?; f=$(grep -v GRAPH_FAILS check_dsmem2_c${c}_$(basename $d).log | grep -c FAIL)
    echo "check C=$c $(basename $d) rc=$rc: FAIL $f, PASS $(grep -c ' PASS' check_dsmem2_c${c}_$(basename $d).log) $(grep -m1 -E 'Error|error|STUCK' check_dsmem2_c${c}_$(basename $d).log | cut -c1-150)"
    [ $rc != 0 ] || [ $f != 0 ] && ok=0
  done
done
[ $ok = 1 ] || { echo DSMEM2_NOT_EXACT; exit 1; }
for d in mix32_two mix32_four mix128_two mix128_four; do
  /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $R/$d/kernel_results.json --unified-only --append --label u_d2_ticket 2>&1 | grep -E "FAIL|rror|Traceback"
  for c in 4 8 16; do
    UNIFIED_MERGE_MODE=dsmem UNIFIED_DSMEM_CLUSTER=$c /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $R/$d/kernel_results.json --unified-only --append --label u_d2_c$c 2>&1 | grep -E "FAIL|rror|Traceback"
  done
done
for c in 4 8 16; do echo "captures C=$c vs ticketed (same build):"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_d2_c$c --baseline arm:u_d2_ticket $R/*/kernel_results.json | sed -E 's/ \|.*//'; done
for d in $(cd $S && ls -d */ | tr -d /); do
  /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $S/$d/captures $OUT/$d/kernel_results.json --unified-only --append --label u_d2_ticket > d2_t_$d.log 2>&1
  for c in 4 8 16; do
    UNIFIED_MERGE_MODE=dsmem UNIFIED_DSMEM_CLUSTER=$c /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $S/$d/captures $OUT/$d/kernel_results.json --unified-only --append --label u_d2_c$c > d2_c${c}_$d.log 2>&1
  done
done
/dev/shm/w/venv/bin/python - <<'PY'
import json,glob,collections
for p in sorted(glob.glob('/opt/tiger/w/synth_meld_results_20261025/*/kernel_results.json')):
  r=json.load(open(p)); t=collections.defaultdict(lambda: collections.defaultdict(list))
  for x in r['records']: t[x['capture']][x['arm']].append(x['milliseconds'])
  row=[]
  for c in sorted(t):
    m={a:sum(v)/len(v) for a,v in t[c].items()}
    if 'u_d2_ticket' in m: row.append(f"{c[:6]} " + "/".join(f"{m['u_d2_ticket']/m[a]:.3f}" for a in ('u_d2_c4','u_d2_c8','u_d2_c16') if a in m))
  print("dsmem2 C=4/8/16 vs ticketed", p.split('/')[-2], ' | '.join(row))
PY
echo DSMEM2_DONE
