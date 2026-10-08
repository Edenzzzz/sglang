#!/usr/bin/env python3
"""Summarize a multi-tenant mixed-workload run: per tenant and node-wide.

Per-tenant numbers matter as much as the aggregate — the point of co-tenancy
is that one tenant's traffic changes the batch composition the other tenant is
served in, so a node-level throughput number alone hides who paid and who won.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def load(p):
    d = json.load(open(p))
    r = {
        "tput": (d["total_input_tokens"] + d["total_output_tokens"]) / d["duration"],
        "out_tput": d["output_throughput"],
        "mean_ttft": d["mean_ttft_ms"], "p99_ttft": d["p99_ttft_ms"],
        "mean_itl": d["mean_itl_ms"], "p99_itl": d["p99_itl_ms"],
        "n": d["completed"], "dur": d["duration"],
        "retok": d.get("total_output_tokens_retokenized", 0) / max(d["total_output_tokens"], 1),
    }
    if d.get("itls"):
        mx = [max(x) * 1000 for x in d["itls"] if x]
        if mx:
            r["reqmax_p99"] = float(np.percentile(mx, 99))
            r["worst"] = float(max(mx))
        # Sarathi-style SLO attainment: fraction of requests whose own P99
        # ITL meets the budget (default matches SGLANG_ITL_SLO_MS=100).
        import os
        slo = float(os.environ.get("ITL_SLO_MS", "100"))
        p99s = [np.percentile(x, 99) * 1000 for x in d["itls"] if len(x) >= 5]
        if p99s:
            r["itl_slo_attain"] = float(np.mean([v <= slo for v in p99s]))
    return r


def pct(b, a):
    return f"{b/a-1:+.2%}" if a else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--baseline", default="DMC_CG")
    ap.add_argument("--method", default="LPT")
    a = ap.parse_args()
    d = Path(a.outdir)

    # Method names contain underscores (DMC_CG), so strip the exact prefix
    # rather than splitting on the first separator.
    streams = sorted({f.name[len(a.baseline) + 1:].rsplit(".", 1)[0]
                      for f in d.glob(f"{a.baseline}_*.json")})
    if not streams:
        print(f"no {a.baseline}_*.json in {d}")
        return

    tot = {}
    for m in (a.baseline, a.method):
        print(f"\n=== {m} ===")
        tot[m] = {"toks": 0.0, "dur": 0.0}
        for s in streams:
            p = d / f"{m}_{s}.json"
            if not p.exists():
                print(f"  {s:8} MISSING")
                continue
            r = load(p)
            tot[m]["toks"] += r["tput"] * r["dur"]
            tot[m]["dur"] = max(tot[m]["dur"], r["dur"])
            extra = (f" reqmaxP99 {r['reqmax_p99']:7.0f}" if "reqmax_p99" in r else "")
            if "itl_slo_attain" in r:
                extra += f" sloAtt {r['itl_slo_attain']*100:5.1f}%"
            print(f"  {s:8} tput {r['tput']:7.0f}  TTFT {r['mean_ttft']:8.0f}/{r['p99_ttft']:8.0f}ms"
                  f"  ITL {r['mean_itl']:6.2f}/{r['p99_itl']:7.2f}ms{extra}"
                  f"  n={r['n']:5} retok {r['retok']*100:5.1f}%")

    print(f"\n=== {a.method} vs {a.baseline} ===")
    for s in streams:
        pa, pb = d / f"{a.baseline}_{s}.json", d / f"{a.method}_{s}.json"
        if not (pa.exists() and pb.exists()):
            continue
        x, y = load(pa), load(pb)
        line = (f"  {s:8} tput {pct(y['tput'], x['tput']):>8}"
                f"  meanTTFT {pct(y['mean_ttft'], x['mean_ttft']):>8}"
                f"  p99TTFT {pct(y['p99_ttft'], x['p99_ttft']):>8}"
                f"  meanITL {pct(y['mean_itl'], x['mean_itl']):>8}"
                f"  p99ITL {pct(y['p99_itl'], x['p99_itl']):>8}")
        if "reqmax_p99" in x and "reqmax_p99" in y:
            line += f"  reqmaxP99 {pct(y['reqmax_p99'], x['reqmax_p99']):>8}"
        print(line)

    if all(tot[m]["dur"] for m in tot):
        ta = tot[a.baseline]["toks"] / tot[a.baseline]["dur"]
        tb = tot[a.method]["toks"] / tot[a.method]["dur"]
        print(f"\n  node-wide throughput: {a.baseline} {ta:.0f} -> {a.method} {tb:.0f} tok/s "
              f"({pct(tb, ta)})")

    for m in (a.baseline, a.method):
        log = d / f"server_{m}.log"
        if log.exists():
            fm = [l for l in log.read_text(errors="ignore").splitlines() if "forward mix" in l]
            if fm:
                print(f"  {m} forward mix: {fm[-1].split('forward mix')[-1].strip()}")


if __name__ == "__main__":
    main()
