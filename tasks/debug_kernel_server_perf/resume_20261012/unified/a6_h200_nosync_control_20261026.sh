#!/bin/bash
# Control for the four-mode merge-free-bound anomaly: the per-item timeline (single run) shows no-sync and the real
# merge equal on four-mode steps, while the graph-replay bench measured no-sync 0.71-0.86x. Here both builds run the
# bench's graph replay with the same UNIFIED_TIMING_BOUND=1, interleaved per set, on the same steps. Gap persists ->
# a no-sync / graph-replay artifact; gap vanishes -> drift between runs taken hours apart. Claude, 2026-10-26.
set -u
cd /dev/shm/w/ub
until grep -q LPTPOS_DONE a6_lptpos.log 2>/dev/null; do sleep 60; done
export PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/fi CUDA_VISIBLE_DEVICES=0
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_STOCK=1 UNIFIED_FA3_MIN=33 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_TIMING_BOUND=1
unset UNIFIED_MERGE_MODE UNIFIED_FA2_ORDER UNIFIED_PDL UNIFIED_NP_ONLY16 UNIFIED_FA2_CHUNK_ROUND
S=/opt/tiger/w/synth_meld_20261025; C=/opt/tiger/w/nosync_control_20261026; rm -rf $C; mkdir -p $C
for d in sharegpt_four longtail_four sharegpt_two; do
  for rep in 1 2; do
    for b in np_sync np_nosync; do
      UNIFIED_SO=/opt/tiger/w/torch_ext_unified_h200_$b/unified_dispatch_ext/unified_dispatch_ext.so \
        /dev/shm/w/venv/bin/python -u bench_unified_trace_20261024.py $S/$d/captures $C/$d.json --unified-only --append --label ${b}_r$rep 2>&1 | grep -E "rror|Traceback"
    done
  done
  /dev/shm/w/venv/bin/python - $C/$d.json <<'PY'
import json,sys,collections
r=json.load(open(sys.argv[1])); t=collections.defaultdict(lambda: collections.defaultdict(list))
for x in r['records']: t[x['capture']][x['arm']].append(x['milliseconds'])
for c in sorted(t):
    m={a:sum(v)/len(v) for a,v in t[c].items()}
    print(sys.argv[1].split('/')[-1], c[:6], ' '.join(f"{a}={m[a]:.4f}" for a in sorted(m)),
          f"sync/nosync r1={m['np_sync_r1']/m['np_nosync_r1']:.3f} r2={m['np_sync_r2']/m['np_nosync_r2']:.3f}")
PY
done
echo NOSYNC_CONTROL_DONE
