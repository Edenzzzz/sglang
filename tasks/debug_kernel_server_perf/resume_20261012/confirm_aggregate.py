"""Aggregate confirm_resv_20261015: per point and tenant, mean over reps of each arm, % vs the MTP mean,
and the rep spread (|r1-r2|/mean) as the measured noise. Usage: confirm_aggregate.py <root>"""
import json, os, sys
import numpy as np

ROOT = sys.argv[1]
ARMS = [("MTP", "MTP_LPT"), ("SPLIT_C3072", "MTPMIX_SPLIT"), ("EDF_RESV", "MTPMIX_SPLIT_EDFR"),
        ("EDF_RESV2", "MTPMIX_SPLIT_EDFR")]
POINTS = [("tracelab_c4", ["agent"]), ("tracelab_c8", ["agent"]), ("cohort_c16", ["agent", "chat"])]


def metrics(path):
    d = json.load(open(path))
    t = np.array(d["ttfts"]) * 1000
    itl = np.array([x * 1000 for r in d["itls"] for x in r])
    out = sum(d["output_lens"])
    dur = None
    for k in ("duration",):
        dur = d.get(k)
    return {"tput": out / dur if dur else float("nan"), "ttft": t.mean(), "ttft_p50": np.median(t),
            "itl": itl.mean(), "itl_p99": np.percentile(itl, 99)}


for point, tenants in POINTS:
    for tenant in tenants:
        rows = {}
        for label, method in ARMS:
            reps = []
            for rep in ("rep1", "rep2"):
                f = os.path.join(ROOT, rep, point, label, f"{method}_{tenant}.json")
                if os.path.exists(f):
                    reps.append(metrics(f))
            if reps:
                rows[label] = reps
        if "MTP" not in rows:
            continue
        base = {k: np.mean([r[k] for r in rows["MTP"]]) for k in rows["MTP"][0]}
        print(f"== {point} {tenant}")
        for label, reps in rows.items():
            m = {k: np.mean([r[k] for r in reps]) for k in reps[0]}
            spread = {k: (abs(reps[0][k] - reps[1][k]) / m[k] if len(reps) == 2 else float("nan")) for k in m}
            print(f"  {label:12s} n={len(reps)} " + " ".join(
                f"{k}={m[k]:.1f}({(m[k] / base[k] - 1) * 100:+.0f}%,±{spread[k] * 50:.0f}%)" for k in m))
