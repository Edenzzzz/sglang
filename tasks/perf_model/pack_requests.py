"""Pack every arm's per-request records under an mn/ results root into one compressed .npz (float32): for each arm and
stream (chat, agent / fanout, spike streams), TTFT (s), per-request TPOT (ms, sum(itls) / (out - 1) or (e2e - ttft) /
(out - 1)), output tokens and start time when recorded. Enough to recompute every per-tenant attainment, goodput and
percentile in the task docs after the pods are gone.
  python pack_requests.py <mn root> <out.npz>
Claude, 2026-10-20.
"""
import glob, json, os, sys
import numpy as np

root, out = sys.argv[1], sys.argv[2]
arrs = {}
for d in sorted(glob.glob(f"{root}/*/")):
    arm = os.path.basename(d.rstrip("/"))
    for name in ("chat", "agent", "chat_spike", "fanout", "fanout_spike"):
        p = f"{d}/{name}.json"
        if not os.path.exists(p):
            continue
        try:
            j = json.load(open(p))
        except ValueError:
            continue
        if "records" in j:  # fanout driver
            rs = [r for r in j["records"] if r.get("ttft") is not None and r.get("out_tokens", 0) > 1]
            a = np.array([[r["ttft"], (r["e2e"] - r["ttft"]) / (r["out_tokens"] - 1) * 1e3, r["out_tokens"],
                           r.get("start", -1)] for r in rs], dtype=np.float32).reshape(-1, 4)
        else:  # sglang bench_serving
            ok = [i for i, e in enumerate(j.get("errors", [])) if not e and j["output_lens"][i] > 1]
            a = np.array([[j["ttfts"][i], sum(j["itls"][i]) / (j["output_lens"][i] - 1) * 1e3, j["output_lens"][i], -1]
                          for i in ok], dtype=np.float32).reshape(-1, 4)
        arrs[f"{arm}/{name}"] = a
np.savez_compressed(out, **arrs)
print(len(arrs), "streams", os.path.getsize(out), "bytes")
