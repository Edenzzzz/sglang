import json, sys, numpy as np
for f in sys.argv[1:]:
    d = json.load(open(f)); t = np.array(d["ttfts"]) * 1000
    unc = np.array(d["input_lens"]) - np.array(d.get("cached_tokens") or [0]*len(t))
    itl = [x*1000 for r in d["itls"] for x in r]
    line = f"{f.split('/')[-2]:10s} mean={t.mean():.0f} p50={np.median(t):.0f} itl={np.mean(itl):.1f} p99itl={np.percentile(itl,99):.0f}"
    for lo, hi in [(0, 4096), (4096, 32768), (32768, 1e9)]:
        m = (unc >= lo) & (unc < hi); line += f" | [{lo//1024}K+) n={m.sum()} {t[m].mean():.0f}"
    print(line)
