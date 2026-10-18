import json, glob
import numpy as np
SLOS = [(10.0, 80.0), (10.0, 50.0), (5.0, 50.0), (20.0, 30.0)]
for path in sorted(glob.glob("/dev/shm/w/mn/*fanout*text*/fanout.json")):
    d = json.load(open(path)); wall = d["wall_s"]
    rs = [r for r in d["records"] if r.get("ttft") is not None and r.get("out_tokens", 0) > 1]
    ok = []
    for ttft_s, tpot_ms in SLOS:
        good = [r for r in rs if r["ttft"] <= ttft_s and (r["e2e"] - r["ttft"]) / (r["out_tokens"] - 1) * 1e3 <= tpot_ms]
        ok.append(f"{ttft_s:g}s/{tpot_ms:g}ms: {len(good)/max(len(rs),1)*100:5.1f}% {sum(r['out_tokens'] for r in good)/wall:6.1f}tok/s")
    print(f"{path.split('/')[-2][:46]:46s} n={len(rs):5d} | " + " | ".join(ok))
