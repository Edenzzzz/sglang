"""Per-request decode speed (tok/s/user) distributions of the fan-out stage-3 arms, for error bars: agent rounds
(out_tokens - 1) / (e2e - ttft) and chat requests (output_len - 1) / sum(itls), p1 / p10 / p50 / p90 / p99 and the
mean, per arm. Also writes <out.json> with the per-arm arrays for the figures.
  python fanout_tok_s_user_20261026.py <out.json> DIR [DIR ...]
Claude, 2026-10-26.
"""
import json
import sys

import numpy as np

out = {}
for d in sys.argv[2:]:
    name = d.rstrip("/").split("/")[-1]
    f = json.load(open(f"{d}/fanout.json"))
    a = [(r["out_tokens"] - 1) / (r["e2e"] - r["ttft"]) for r in f["records"]
         if r.get("ttft") is not None and r.get("out_tokens", 0) > 1 and r["e2e"] > r["ttft"]]
    c = json.load(open(f"{d}/chat.json"))
    ch = [(c["output_lens"][i] - 1) / sum(c["itls"][i]) for i, e in enumerate(c["errors"])
          if not e and c["output_lens"][i] > 1 and sum(c["itls"][i]) > 0]
    row = {}
    for tenant, v in (("agent", a), ("chat", ch)):
        v = np.array(v)
        q = np.percentile(v, [1, 10, 50, 90, 99]) if len(v) else [float("nan")] * 5
        row[tenant] = dict(n=int(len(v)), mean=float(v.mean()) if len(v) else float("nan"),
                           p1=float(q[0]), p10=float(q[1]), p50=float(q[2]), p90=float(q[3]), p99=float(q[4]),
                           values=[round(float(x), 2) for x in v])
        print(f"{name[:44]:44s} {tenant:5s} n={len(v):5d} mean={row[tenant]['mean']:7.1f} "
              f"p1={q[0]:7.1f} p10={q[1]:7.1f} p50={q[2]:7.1f} p90={q[3]:7.1f} p99={q[4]:7.1f}")
    out[name] = row
json.dump(out, open(sys.argv[1], "w"))
