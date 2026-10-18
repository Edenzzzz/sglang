"""GPU busy union vs wall in a torch-profiler chrome trace, plus idle gaps > 50 us and the kernels on each side.
  python gpu_gaps.py <trace.json[.gz]>"""
import collections, gzip, json, sys

p = sys.argv[1]
t = json.load(gzip.open(p) if p.endswith(".gz") else open(p))
ev = [e for e in t["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
ev.sort(key=lambda e: e["ts"])
busy = 0.0
gaps = []
end = ev[0]["ts"]
prev = ev[0]
for e in ev:
    if e["ts"] > end:
        gaps.append((e["ts"] - end, prev["name"][:70], e["name"][:70]))
    if e["ts"] + e["dur"] > end:
        busy += e["ts"] + e["dur"] - max(e["ts"], end)
        end = e["ts"] + e["dur"]
        prev = e
wall = end - ev[0]["ts"]
big = [g for g in gaps if g[0] > 50]
print(f"{p}: kernels={len(ev)} wall={wall/1e3:.1f}ms busy={busy/1e3:.1f}ms ({busy/wall*100:.1f}%) "
      f"gaps>50us n={len(big)} total={sum(g[0] for g in big)/1e3:.1f}ms all_gaps={sum(g[0] for g in gaps)/1e3:.1f}ms")
agg = collections.defaultdict(lambda: [0, 0.0])
for d, a, b in big:
    k = (a, b)
    agg[k][0] += 1; agg[k][1] += d
for (a, b), (n, d) in sorted(agg.items(), key=lambda x: -x[1][1])[:15]:
    print(f"  {d/1e3:8.1f}ms n={n:4d}  after {a}  ->  before {b}")
kn = collections.defaultdict(float)
for e in ev:
    kn[e["name"][:60]] += e["dur"]
print("top kernels by time:")
for k, d in sorted(kn.items(), key=lambda x: -x[1])[:12]:
    print(f"  {d/1e3:8.1f}ms {d/busy*100:5.1f}%  {k}")
