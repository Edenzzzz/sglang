#!/usr/bin/env python3
"""EDF step-model check: how well a lightweight step model predicts step time on the pdmux schedule vs the stock
(unfused) schedules, from STEPTRACE logs (SGLANG_STEP_TRACE=1; pdmux lines carry sg= and lay= from
apply_steptrace_pdmux.py).

Targets: gpu = the forward's GPU time; gap = launch-to-launch interval of consecutive steps of the same kind (decode
for pdmux, all steps for stock), restricted to back-to-back steps (< --gap-ms), i.e. what ITL sees.
Models (least squares; fit on the first half of the trace, MAPE / p90 APE on the second half):
  stock      ms = c0 + c_tok*ptok + c_attn*attn + c_bs*nd + c_kv*dkv           (fit_mtp_step.py's form)
  pdmux dec  ms = per-SM-group (c0 + c_bs*nd + c_kv*dkv)                        (decode on its own SM group)
  pdmux pre  ms = per-SM-group (c0 + lay*(c_tok*ptok + c_attn*attn))           (one split forward of `lay` layers)
with attn = (ptok*prefix/np + ptok^2/(2*np))/1e6.
Usage: fit_pdmux_step.py <server.log> [...]"""
import re
import sys

import numpy as np

PAT = re.compile(r"STEPTRACE (\w+) et=(\d+) bs=(\d+) np=(\d+) ptok=(\d+) pf=(\d+) nd=(\d+) dkv=(\d+) kv=(\d+) "
                 r"mx=(\d+)(?: sg=(\d+) lay=(\d+))?(?: acc=[0-9.]+)?(?: dacc=[0-9.]+)? t=([0-9.]+) gpu=([0-9.]+)")
GAP_MS = 200.0


def rows(path):
    out = []
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if not m:
            continue
        g = m.groups()
        r = dict(mode=g[0], np=int(g[3]), ptok=int(g[4]), pf=int(g[5]), nd=int(g[6]), dkv=int(g[7]),
                 sg=int(g[10]) if g[10] is not None else -1, lay=int(g[11]) if g[11] is not None else 0,
                 t=float(g[12]), gpu=float(g[13]))
        r["attn"] = (r["ptok"] * r["pf"] / max(r["np"], 1) + 0.5 * r["ptok"] ** 2 / max(r["np"], 1)) / 1e6
        out.append(r)
    return out


def add_gaps(rs):
    for a, b in zip(rs, rs[1:]):
        g = (b["t"] - a["t"]) * 1e3
        a["gap"] = g if 0 < g < GAP_MS else None
    if rs:
        rs[-1]["gap"] = None


def evaluate(name, rs, design, target, groups=None):
    rs = [r for r in rs if r.get(target) is not None and r[target] > 0]
    if len(rs) < 40:
        print(f"  {name:28s} {target:3s} n={len(rs)} (too few)")
        return
    half = len(rs) // 2
    tr, te = rs[:half], rs[half:]
    keys = sorted({groups(r) for r in rs}) if groups else [None]
    pred = {}
    coef_count = 0
    for k in keys:
        trk = [r for r in tr if (groups(r) if groups else None) == k]
        if len(trk) < 5:
            continue
        X = np.array([design(r) for r in trk]); y = np.array([r[target] for r in trk])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred[k] = c; coef_count += len(c)
    ape = []
    for r in te:
        k = groups(r) if groups else None
        if k not in pred:
            continue
        p = float(np.dot(design(r), pred[k]))
        ape.append(abs(p - r[target]) / r[target])
    ape = np.array(ape)
    ys = np.array([r[target] for r in te])
    print(f"  {name:28s} {target:3s} n={len(rs)} test={len(ape)} mean={ys.mean():.2f} ms cv={ys.std() / ys.mean():.2f} "
          f"MAPE={ape.mean():.1%} p90APE={np.percentile(ape, 90):.1%} coefs={coef_count}")


for path in sys.argv[1:]:
    rs = rows(path)
    print(path)
    if any(r["sg"] >= 0 for r in rs):
        dec = [r for r in rs if r["lay"] == 0]
        pre = [r for r in rs if r["lay"] > 0]
        add_gaps(dec)
        sg = lambda r: r["sg"]
        evaluate("pdmux decode", dec, lambda r: [1.0, r["nd"], r["dkv"] / 1e3], "gpu", sg)
        evaluate("pdmux decode", dec, lambda r: [1.0, r["nd"], r["dkv"] / 1e3], "gap", sg)
        evaluate("pdmux prefill split", pre, lambda r: [1.0, r["lay"] * r["ptok"] / 1e3, r["lay"] * r["attn"]], "gpu", sg)
    else:
        add_gaps(rs)
        d = lambda r: [1.0, r["ptok"] / 1e3, r["attn"], r["nd"], r["dkv"] / 1e3]
        evaluate("stock all steps", rs, d, "gpu")
        evaluate("stock all steps", rs, d, "gap")
        for target in ("gpu", "gap"):
            evaluate("stock decode-only steps", [r for r in rs if r["np"] == 0], d, target)
            evaluate("stock mixed steps", [r for r in rs if r["np"] > 0], d, target)
