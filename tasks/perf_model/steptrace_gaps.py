"""Host vs GPU time per serving step from STEPTRACE logs: wall span, GPU-busy sum, and the interval between consecutive
step timestamps (t=) against each step's GPU time, by step type -- how much of a decode row's per-token time is GPU work.
  python steptrace_gaps.py worker.log [worker.log ...]
Claude, 2026-10-21.
"""
import re, sys
import numpy as np
PAT = re.compile(r"STEPTRACE (\w+) et=\d+ bs=(\d+) np=\d+ ptok=(\d+) pf=\d+ nd=\d+ dkv=\d+ kv=\d+ mx=(\d+).*? t=([0-9.]+) gpu=([0-9.]+)")
for path in sys.argv[1:]:
    rs = [(m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5)), float(m.group(6)))
          for m in map(PAT.search, open(path, errors="replace")) if m]
    t = np.array([r[4] for r in rs]); g = np.array([r[5] for r in rs])
    dt = np.diff(t) * (1e3 if t.max() < 1e8 else 1e3)
    busy = dt < 200  # inside a busy period (gaps > 200 ms are idle server, no work queued)
    print(f"== {path.split('/')[-2][-40:]}: {len(rs)} steps, wall {t[-1] - t[0]:.0f} s, GPU sum {g.sum() / 1e3:.0f} s")
    print(f"   busy-period intervals: n={busy.sum()} mean={dt[busy].mean():.2f} ms, GPU mean {g[1:][busy].mean():.2f} ms, "
          f"host gap mean {(dt[busy] - g[1:][busy]).mean():.2f} ms")
    for kind, mx in (("DECODE", 0), ("EXTEND", 1)):
        sel = np.array([(r[0] == kind and (r[3] > 0) == mx) for r in rs[1:]]) & busy
        if sel.sum():
            print(f"   {kind:6s} mixed={mx}: n={sel.sum()} interval mean={dt[sel].mean():.2f} p50={np.median(dt[sel]):.2f} "
                  f"GPU mean={g[1:][sel].mean():.2f} bs mean={np.mean([rs[i+1][1] for i in np.where(sel)[0]]):.1f}")
