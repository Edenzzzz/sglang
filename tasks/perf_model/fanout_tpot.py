import json, sys, glob
import numpy as np
for path in sorted(glob.glob("/dev/shm/w/mn/*fanout*/fanout.json")):
    d = json.load(open(path))
    rs = [r for r in d["records"] if r.get("ttft") is not None and r.get("e2e") and r.get("out_tokens", 0) > 1]
    tpot = np.array([(r["e2e"] - r["ttft"]) / (r["out_tokens"] - 1) * 1e3 for r in rs])
    itl = np.array([r["itl_ms"] for r in rs if r.get("itl_ms")])
    print(f"{path.split('/')[-2][:58]:58s} n={len(rs):5d} TPOT mean={tpot.mean():6.2f}ms p50={np.median(tpot):6.2f} p99={np.percentile(tpot,99):7.2f} | chunk-ITL mean={itl.mean():6.2f}")
