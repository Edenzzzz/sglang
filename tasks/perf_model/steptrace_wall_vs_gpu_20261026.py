"""Per step type: count, GPU ms (STEPTRACE gpu=), wall ms (next launch - this launch), tokens, decode rows, from
SGLANG_STEP_TRACE server logs of run_native_serving_pair arms. Wall > GPU means the host held the next launch.
  python steptrace_wall_vs_gpu_20261026.py <arm dir> [...]   Claude, 2026-10-26."""
import re, sys, statistics, collections
for a in sys.argv[1:]:
    off = int(open(f"{a}/requests/server_log_offset").read())
    f = open(f"{a}/server.log", errors="ignore"); f.seek(off)
    rows = []
    for line in f:
        if "STEPTRACE " not in line or "gpu=" not in line: continue
        m = re.search(r"STEPTRACE (\w+) et=(\d+) bs=(\d+) np=(\d+) ptok=(\d+) pf=(\d+) nd=(\d+) .* t=([0-9.]+) gpu=([0-9.]+)", line)
        if m: rows.append((m[1], int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[7]), float(m[8]), float(m[9])))
    rows.sort(key=lambda r: r[6])
    by = collections.defaultdict(list)
    for i, r in enumerate(rows[:-1]):
        wall = (rows[i + 1][6] - r[6]) * 1e3
        kind = r[0] + ("+pf" if r[3] else "")
        by[kind].append((r[7], wall, r[1], r[5]))
    span = (rows[-1][6] - rows[0][6]) if rows else 0
    print(f"== {a.split('/')[-1]}: steps={len(rows)} span={span:.1f}s gpu_sum={sum(r[7] for r in rows)/1e3:.1f}s")
    for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        g = [x[0] for x in v]; w = [x[1] for x in v]
        print(f"   {k:14s} n={len(v):5d} gpu mean={statistics.mean(g):6.2f}ms p50={statistics.median(g):6.2f}  wall mean={statistics.mean(w):6.2f}ms p50={statistics.median(w):6.2f}  tokens={statistics.mean(x[2] for x in v):6.0f} nd={statistics.mean(x[3] for x in v):5.1f}  sum_wall={sum(w)/1e3:6.1f}s")
