"""Per-tenant scoring of the 8-GPU fan-out runs: chat tenant on SLO attainment (TTFT and per-request TPOT targets, from
bench_serving's per-request arrays), agent tenant on SLO goodput (as fanout_slo_goodput.py). Per-request TPOT =
sum(itls) / (output_len - 1), which counts tokens rather than stream chunks, so MTP arms compare fairly.
  python fanout_per_tenant.py DIR [DIR ...]      (each DIR holds chat.json and fanout.json)
A second line per DIR ("PAPER") scores the paper's SLO matrix (2026-10-25, from JITServe / SLOs-Serve): chat TTFT 2 s
with mean TPOT 50 ms (tight) / 100 ms (loose); agent rounds TTFT 10 s, 20 s when the round has more than 200K uncached
tokens, with TPOT 80 / 100 ms; each at SLO scales 0.8 / 1.0 / 1.5 (attainment for chat, SLO goodput for agents).
It prints before the unchanged original line, which stays last (the stage-3 drivers parse it with tail -1).
Claude, 2026-10-19.
"""
import json
import sys

import numpy as np

CHAT_SLOS = [(0.5, 20.0), (1.0, 20.0), (1.0, 30.0), (2.0, 50.0)]
AGENT_SLOS = [(10.0, 80.0), (20.0, 30.0)]
PAPER_CHAT = {"tight": (2.0, 50.0), "loose": (2.0, 100.0)}
PAPER_AGENT = {"tight": 80.0, "loose": 100.0}
PAPER_SCALES = (0.8, 1.0, 1.5)
LONG_UNCACHED = 200_000


def agent_ttft_slo(r):
    uncached = int(r.get("input_len") or 0) - int(r.get("cached") or 0)
    return 20.0 if uncached > LONG_UNCACHED else 10.0

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
    paper = []
    for k in PAPER_SCALES:
        cells = []
        for name, (t, m) in PAPER_CHAT.items():
            cells.append(f"chat_{name}={np.mean((ttft <= t * k) & (tpot <= m * k)) * 100:5.1f}%")
        for name, m in PAPER_AGENT.items():
            good = [r for r in rs if r["ttft"] <= agent_ttft_slo(r) * k
                    and (r["e2e"] - r["ttft"]) / (r["out_tokens"] - 1) * 1e3 <= m * k]
            cells.append(f"agent_{name}={sum(r['out_tokens'] for r in good) / f['wall_s']:5.0f}tok/s"
                         f"({len(good) / max(1, len(rs)) * 100:4.1f}%)")
        paper.append(f"x{k:g}: " + " ".join(cells))
    print(f"PAPER {d.rstrip('/').split('/')[-1][:44]:44s} " + " | ".join(paper))
    print(f"{d.rstrip('/').split('/')[-1][:44]:44s} chat n={len(ok)} TPOT p50={np.median(tpot):5.1f} "
          f"p90={np.percentile(tpot, 90):5.1f} | {chat} | agent {' '.join(agent)}")
