import json, glob, os
K = ("mean_ttft_ms", "p99_ttft_ms", "mean_itl_ms", "p99_itl_ms", "output_throughput")
def load(f):
    return json.loads([l for l in open(f) if l.strip()][-1])
print(f"{'rate':>4} {'stream':6} {'arm':13} " + " ".join(f"{k:>16}" for k in K))
for rate in (4, 8, 12, 16):
    base = {}
    for arm in ("MTP_LPT", "MTPMIX_LPT", "MTPMIX_SPLIT"):
        for s in ("codex", "chat"):
            f = f"/dev/shm/w/mtpload/rate{rate}/r1_{arm}/{arm}_{s}.json"
            if not os.path.exists(f):
                continue
            d = load(f)
            if arm == "MTP_LPT":
                base[s] = d
            row = []
            for k in K:
                v = d.get(k) or 0.0
                b = base.get(s, {}).get(k) or 0.0
                row.append(f"{v:9.1f}" + (f"({(v / b - 1) * 100:+5.0f}%)" if arm != "MTP_LPT" and b else "       "))
            print(f"{rate:>4} {s:6} {arm:13} " + " ".join(row))
