"""Per-tenant scoring of the 8-GPU fan-out runs: chat tenant on SLO attainment (TTFT and per-request TPOT targets, from
bench_serving's per-request arrays), agent tenant on SLO goodput (as fanout_slo_goodput.py). Per-request TPOT =
sum(itls) / (output_len - 1), which counts tokens rather than stream chunks, so MTP arms compare fairly.
  python fanout_per_tenant.py DIR [DIR ...]      (each DIR holds chat.json and fanout.json)
Claude, 2026-10-19.
"""
import json
import sys

import numpy as np

CHAT_SLOS = [(0.5, 20.0), (1.0, 20.0), (1.0, 30.0), (2.0, 50.0)]
AGENT_SLOS = [(10.0, 80.0), (20.0, 30.0)]

for d in sys.argv[1:]:
    c = json.load(open(f"{d}/chat.json"))
    ok = [i for i, e in enumerate(c["errors"]) if not e and c["output_lens"][i] > 1]
    ttft = np.array([c["ttfts"][i] for i in ok])
    tpot = np.array([sum(c["itls"][i]) / (c["output_lens"][i] - 1) * 1e3 for i in ok])
    chat = " ".join(f"{t:g}s/{m:g}ms={np.mean((ttft <= t) & (tpot <= m)) * 100:5.1f}%" for t, m in CHAT_SLOS)
    f = json.load(open(f"{d}/fanout.json"))
    rs = [r for r in f["records"] if r.get("ttft") is not None and r.get("out_tokens", 0) > 1]
    agent = []
    for t, m in AGENT_SLOS:
        good = [r for r in rs if r["ttft"] <= t and (r["e2e"] - r["ttft"]) / (r["out_tokens"] - 1) * 1e3 <= m]
        agent.append(f"{t:g}s/{m:g}ms={sum(r['out_tokens'] for r in good) / f['wall_s']:5.0f}tok/s")
    print(f"{d.rstrip('/').split('/')[-1][:44]:44s} chat n={len(ok)} TPOT p50={np.median(tpot):5.1f} "
          f"p90={np.percentile(tpot, 90):5.1f} | {chat} | agent {' '.join(agent)}")
