"""Per-arm serving summary with spread for a kernel_concurrency_driver A/B root (runs on the pod):
  python summarize_ab.py <root> <arm> [<arm> ...]
Prints throughput, mean TTFT, mean / P99 ITL, tok/s/user p1/p10/p50/p90/p99 and acceptance length per stream."""
import glob, json, os, sys

import numpy as np

root, arms = sys.argv[1], sys.argv[2:]
for C in (8, 16):
    for arm in arms:
        d = f"{root}/cohort_c{C}/{arm}"
        for f in sorted(glob.glob(d + "/*.json")):
            b = os.path.basename(f)
            if "programs" in b or "validation" in b:
                continue
            try:
                j = json.loads(open(f).read().strip().splitlines()[-1])
            except Exception:
                continue
            if "ttfts" not in j:
                continue
            out = np.array(j["output_lens"], float)
            itls = j["itls"]
            dec = np.array([sum(x) for x in itls])
            ok = (out > 1) & (dec > 1e-3 * (out - 1))
            user = (out[ok] - 1) / dec[ok]
            p = np.percentile(user, [1, 10, 50, 90, 99]) if len(user) else [0] * 5
            tok = np.concatenate([np.array(x) for x in itls if len(x)])
            acc = j.get("accept_length")
            print(f"C={C} {arm:13s} {b[:-5]:28s} n={len(out):5d} tput={out.sum() / j['duration']:7.1f} "
                  f"ttft={np.mean(j['ttfts']):6.2f}s itl={tok.mean() * 1e3:6.1f}ms p99itl={np.percentile(tok, 99) * 1e3:6.0f} "
                  f"tok/s/user p1/p10/p50/p90/p99={'/'.join(f'{x:.0f}' for x in p)} accept={acc}")
