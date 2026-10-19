"""Per-tenant scoring of single-node serving arms from bench_serving --output-details files: each tenant's SLO attainment
(share of requests meeting both a TTFT and a per-request TPOT target) and SLO goodput (output tok/s of those requests).
Per-request TPOT = sum(itls) / (output_len - 1): tokens, not stream chunks, so MTP arms compare fairly.
  python bench_per_tenant.py DIR [DIR ...]     (each DIR holds agent.json and chat.json)
Claude, 2026-10-19.
"""
import json
import sys

import numpy as np

SLOS = {"agent": [(10.0, 80.0), (10.0, 30.0), (5.0, 30.0)], "chat": [(0.5, 20.0), (1.0, 30.0), (2.0, 50.0)]}


def score(path, slos):
    c = json.load(open(path))
    ok = [i for i, e in enumerate(c["errors"]) if not e and c["output_lens"][i] > 1]
    ttft = np.array([c["ttfts"][i] for i in ok])
    tpot = np.array([sum(c["itls"][i]) / (c["output_lens"][i] - 1) * 1e3 for i in ok])
    out = np.array([c["output_lens"][i] for i in ok])
    cells = []
    for t, m in slos:
        good = (ttft <= t) & (tpot <= m)
        cells.append(f"{t:g}s/{m:g}ms={good.mean() * 100:5.1f}%/{out[good].sum() / c['duration']:6.1f}tok/s")
    return (f"n={len(ok)} TTFT mean={ttft.mean():6.2f}s TPOT p50={np.median(tpot):5.1f} p90={np.percentile(tpot, 90):5.1f}"
            f" | " + " ".join(cells))


for d in sys.argv[1:]:
    name = d.rstrip("/").split("/")[-1]
    for tenant in ("agent", "chat"):
        print(f"{name[:52]:52s} {tenant:5s} {score(f'{d}/{tenant}.json', SLOS[tenant])}")
