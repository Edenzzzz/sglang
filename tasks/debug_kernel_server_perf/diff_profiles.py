"""Diff the LPT vs DMC nsys captures: GPU busy fraction, API blocking, and
where the idle time sits."""
import sqlite3
import sys


def analyze(db):
    c = sqlite3.connect(db)
    out = {}
    # capture window from first to last GPU activity
    span = c.execute("""SELECT min(start), max(end) FROM (
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
        UNION ALL SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY)""").fetchone()
    window = span[1] - span[0]
    out["window_s"] = window / 1e9

    # GPU busy: merge kernel intervals
    rows = c.execute("""SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
                        ORDER BY start""").fetchall()
    busy = 0
    cur_s, cur_e = None, None
    for s, e in rows:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    out["gpu_busy_pct"] = busy / window * 100
    out["n_kernels"] = len(rows)

    # API blocking
    api = c.execute("""SELECT s.value, count(*), sum(r.end-r.start)/1e9, max(r.end-r.start)/1e6
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id
        WHERE s.value LIKE 'cuda%' GROUP BY s.value
        ORDER BY sum(r.end-r.start) DESC LIMIT 6""").fetchall()
    out["api"] = api

    # named kernel groups
    kg = c.execute("""SELECT
        CASE WHEN s.value LIKE '%PersistentKernelTemplate%' THEN 'persistent'
             WHEN s.value LIKE '%BatchPrefillWithPagedKVCacheKernel%' THEN 'bp_paged'
             WHEN s.value LIKE '%PrefillWithKVCacheKernel%' THEN 'ragged'
             WHEN s.value LIKE '%BatchDecode%' THEN 'decode'
             ELSE 'other' END grp,
        count(*), sum(k.end-k.start)/1e9, avg(k.end-k.start)/1e6
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
        GROUP BY grp ORDER BY 3 DESC""").fetchall()
    out["kgroups"] = kg
    return out


for name in ("LPT", "DMC_CG"):
    db = f"nsys_{name}.sqlite"
    try:
        r = analyze(db)
    except Exception as e:
        print(f"{name}: {e}")
        continue
    print(f"\n=== {name}: window {r['window_s']:.1f}s, GPU busy {r['gpu_busy_pct']:.1f}%, "
          f"{r['n_kernels']} kernels")
    print("  kernel groups (count, total_s, avg_ms):")
    for g, n, tot, avg in r["kgroups"]:
        print(f"    {g:11} {n:7} {tot:8.2f}s {avg:8.3f}ms")
    print("  top API blocking:")
    for nm, n, tot, mx in r["api"]:
        print(f"    {nm:28} {n:7} calls {tot:8.2f}s  max {mx:8.1f}ms")
