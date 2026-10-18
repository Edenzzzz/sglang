import json, glob, collections, os
import numpy as np
for d in sorted(glob.glob("/dev/shm/w/mn/*fanout_text*")):
    fj = os.path.join(d, "fanout.json")
    if not os.path.exists(fj):
        continue
    j = json.load(open(fj)); rs = [r for r in j["records"] if r.get("cached") is not None]
    inp = sum(r["input_len"] for r in rs); unc = sum(r["input_len"] - r["cached"] for r in rs)
    line = f"{os.path.basename(d)[:44]:44s} req={len(rs):5d} input={inp/1e6:7.1f}M uncached(re-prefill)={unc/1e6:6.2f}M ({unc/inp*100:4.1f}%)"
    rl = os.path.join(d, "router_decisions.jsonl")
    if os.path.exists(rl):
        ws = collections.defaultdict(set); n = 0
        for l in open(rl):
            try:
                x = json.loads(l)
            except Exception:
                continue
            n += 1
            key = x.get("prefix_key") or x.get("hash") or x.get("prompt_head")
            w = x.get("worker") or x.get("chosen") or x.get("dst")
            if key is not None and w is not None:
                ws[str(key)[:64]].add(str(w))
        if n:
            keys = list(x.keys())
            line += f" | router decisions={n} keys={keys[:8]}"
            if ws:
                c = [len(v) for v in ws.values()]
                line += f" distinct-workers/prefix mean={np.mean(c):.2f} max={max(c)}"
    print(line)
