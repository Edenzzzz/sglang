#!/usr/bin/env python3
"""S4 / goal point 3: does a lightweight step model predict what EDF needs on the 4-mode fused schedule better than on
stock MTP, and does a predicted acceptance length help predict TPOT?

From STEPTRACE logs (SGLANG_STEP_TRACE=1). Per step: features np, ptok, attn = (ptok*prefix/np + ptok^2/(2*np))/1e6,
nd (decode/verify rows), dkv (their KV, K tokens); acc = the server's running mean accepted tokens per verify (the
online signal EDF can use; an EWMA of it is what a deployment would feed). Targets:
  step   = the step's GPU time (ms)
  tpot   = launch-to-launch interval of back-to-back steps (< 200 ms) / accepted tokens per verify -- the per-token time
           a decoding request sees
Models: least squares, linear in [1, ptok, attn, nd, dkv] (stock form), fit on the first half, held-out MAPE / p90 APE
on the second half; the TPOT model is fit as interval and divided by (a) the EWMA acceptance or (b) a constant (the
fit half's mean acceptance) -- the "without acceptance input" baseline. Also a small MLP (sklearn, 2x32) when available.
Usage: fit_tpot_acc.py <server.log> [...]"""
import re
import sys

import numpy as np

PAT = re.compile(r"STEPTRACE (\w+) et=(\d+) bs=(\d+) np=(\d+) ptok=(\d+) pf=(\d+) nd=(\d+) dkv=(\d+) kv=(\d+) "
                 r"mx=(\d+)(?: sg=\d+ lay=\d+)? acc=([0-9.]+)(?: dacc=([0-9.]+))? t=([0-9.]+) gpu=([0-9.]+)")


def load(path):
    rs = []
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if m:
            g = m.groups()
            np_, ptok, pf, nd, dkv = int(g[3]), int(g[4]), int(g[5]), int(g[6]), int(g[7])
            attn = (ptok * pf / max(np_, 1) + 0.5 * ptok ** 2 / max(np_, 1)) / 1e6
            rs.append(dict(np=np_, ptok=ptok, attn=attn, nd=nd, dkv=dkv / 1e3, acc=float(g[10]),
                           dacc=float(g[11]) if g[11] is not None else None,
                           t=float(g[12]) * 1e3, gpu=float(g[13])))
    rs.sort(key=lambda r: r["t"])
    return rs


def mape(pred, y):
    e = np.abs(pred - y) / np.maximum(y, 1e-6)
    return 100 * e.mean(), 100 * np.percentile(e, 90)


def design(rs):
    return np.array([[1.0, r["ptok"] / 1e3, r["attn"], r["nd"], r["dkv"]] for r in rs])


def fit_eval(rs, target, label, kind):
    if len(rs) < 40:
        return
    h = len(rs) // 2
    X, y = design(rs), np.array([target(r) for r in rs])
    coef, *_ = np.linalg.lstsq(X[:h], y[:h], rcond=None)
    m, p = mape(X[h:] @ coef, y[h:])
    out = f"  {label:34s} {kind:7s} n={len(rs):6d} mean={y[h:].mean():7.2f} linear MAPE={m:5.1f}% p90={p:5.1f}%"
    try:
        from sklearn.neural_network import MLPRegressor
        mu, sd = X[:h].mean(0), X[:h].std(0) + 1e-9
        mlp = MLPRegressor((32, 32), max_iter=600, random_state=0).fit((X[:h] - mu) / sd, y[:h])
        m2, p2 = mape(mlp.predict((X[h:] - mu) / sd), y[h:])
        out += f" | MLP MAPE={m2:5.1f}% p90={p2:5.1f}%"
    except Exception:
        pass
    print(out)


for path in sys.argv[1:]:
    rs = load(path)
    print(path)
    kinds = {"decode-only": [r for r in rs if r["np"] == 0],
             "with prefill": [r for r in rs if r["np"] > 0 and r["nd"] > 0],
             "all": rs}
    for name, sub in kinds.items():
        fit_eval(sub, lambda r: r["gpu"], f"step GPU time, {name}", "gpu")
    # TPOT: interval to the next back-to-back step, per accepted token.
    pairs = [(a, b["t"] - a["t"]) for a, b in zip(rs, rs[1:]) if 0 < b["t"] - a["t"] < 200 and a["nd"] > 0]
    if len(pairs) > 40:
        h = len(pairs) // 2
        X = design([a for a, _ in pairs])
        iv = np.array([d for _, d in pairs])
        # Per-step acceptance when the trace has it (dacc, apply_steptrace_dacc.py), else the cumulative mean.
        per_step = pairs[0][0]["dacc"] is not None
        acc = np.array([(a["dacc"] or a["acc"]) if per_step else a["acc"] for a, _ in pairs])
        acc = np.maximum(acc, 1.0)
        # EWMA of the server's acceptance (alpha 0.05 per step), what an online predictor would hold.
        ew = np.empty_like(acc); e = acc[0]
        for i, x in enumerate(acc):
            e = 0.95 * e + 0.05 * x; ew[i] = e
        tpot = iv / acc
        coef, *_ = np.linalg.lstsq(X[:h], iv[:h], rcond=None)
        pred_iv = X[h:] @ coef
        m_acc, p_acc = mape(pred_iv / ew[h:], tpot[h:])
        m_const, p_const = mape(pred_iv / acc[:h].mean(), tpot[h:])
        print(f"  TPOT per accepted token           n={len(pairs):6d} mean={tpot[h:].mean():6.2f} ms"
              f"  with EWMA acceptance MAPE={m_acc:5.1f}% p90={p_acc:5.1f}%"
              f"  | constant acceptance MAPE={m_const:5.1f}% p90={p_const:5.1f}%"
              f"  ({'per-step' if per_step else 'cumulative'} acc range {acc.min():.2f}-{acc.max():.2f}, sd {acc.std():.2f})")
