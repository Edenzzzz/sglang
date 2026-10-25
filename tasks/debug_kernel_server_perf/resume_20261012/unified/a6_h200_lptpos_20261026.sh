#!/bin/bash
# FA2 order lptpos (longest first, equal-cost chunks by KV position) vs LPT, same build (h200_np_cr, built by the
# chunk-rounding run), H200: synthetic sets incl. the 48K shared prefix + the 96 captures. kvpos alone helped only a
# long shared prefix with split rows (1.08x) and lost 0.71-0.88x on split ShareGPT batches. Claude, 2026-10-26.
set -u
cd /dev/shm/w/ub
until grep -q CHUNK_ROUND_DONE a6_chunk_round.log 2>/dev/null; do sleep 60; done
grep -q 'lptpos' port07_unified/include/flashinfer/attention/fused_plan.cuh || { echo STALE; exit 1; }
grep -q lptpos /opt/tiger/w/torch_ext_unified_h200_np_cr/unified_dispatch_ext/unified_dispatch_ext.so || { echo SO_WITHOUT_LPTPOS; exit 1; }
export PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi CUDA_VISIBLE_DEVICES=0
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3
unset UNIFIED_MERGE_MODE UNIFIED_FA2_ORDER UNIFIED_PDL UNIFIED_NP_ONLY16 UNIFIED_FA2_CHUNK_ROUND
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_h200_np_cr/unified_dispatch_ext/unified_dispatch_ext.so
S=/opt/tiger/w/synth_meld_20261025; OUT=/opt/tiger/w/synth_meld_results_20261025
B=/opt/tiger/w/trace_sampled_h200_20261025; R=/opt/tiger/w/unified_h200_20261025
for d in $(cd $S && ls -d */ | tr -d /); do
  UNIFIED_FA2_ORDER=lptpos timeout 3600 /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $S/$d/captures $OUT/$d/kernel_results.json --unified-only --append --label u_lptpos > lptpos_$d.log 2>&1
  echo "$d: $(grep -c UNIFIED_BENCH_FAIL lptpos_$d.log) fails"
done
for d in mix32_two mix32_four mix128_two mix128_four; do
  UNIFIED_FA2_ORDER=lptpos /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $B/$d/captures $R/$d/kernel_results.json --unified-only --append --label u_lptpos 2>&1 | grep -E "FAIL|rror|Traceback"
done
echo "captures, lptpos vs LPT (same build):"; /dev/shm/w/venv/bin/python analyze_trace_sampled_20261023.py u_lptpos --baseline arm:u_cr_pow2 $R/*/kernel_results.json | sed -E 's/ \|.*//'
sed -i "s/'u_np_lpt2','u_np_kvpos'/'u_np_lpt2','u_np_kvpos','u_lptpos'/" kv_cmp.py
/dev/shm/w/venv/bin/python - <<'PY'
import json,glob,collections
for p in sorted(glob.glob('/opt/tiger/w/synth_meld_results_20261025/*/kernel_results.json')):
  r=json.load(open(p)); t=collections.defaultdict(lambda: collections.defaultdict(list))
  for x in r['records']: t[x['capture']][x['arm']].append(x['milliseconds'])
  row=[]
  for c in sorted(t):
    m={a:sum(v)/len(v) for a,v in t[c].items()}
    if 'u_cr_pow2' in m and 'u_lptpos' in m: row.append(f"{c[:6]} {m['u_cr_pow2']/m['u_lptpos']:.3f}")
  print("lptpos vs LPT", p.split('/')[-2], ' | '.join(row))
PY
echo LPTPOS_DONE
