#!/usr/bin/env python3
"""Fit the EDF step model for the speculative (MTP) server from a STEPTRACE log.

An MTP step is draft steps + one target forward (prefill chunk | verify rows) + draft extends,
which none of the non-speculative models describe. EDF needs, per admission decision, the step's
wall time as a function of the chunk it is about to admit:

    ms = c0 + c_tok*n_p + c_attn*(n_p*r_p + n_p^2/2)/1e6 + c_bs*bs + c_kv*sum_kv/1e3 + c_acc*a_hat

n_p/r_p are the prefill tokens and their average prefix, bs/sum_kv the verify rows and their KV.
`a_hat` is an optional online acceptance EWMA; old traces omit it and produce
the original five-coefficient model.
Target is the launch-to-launch interval (what ITL sees), restricted to back-to-back steps
(interval < --gap-ms) so idle arrival gaps do not enter. Fit on the first half of the trace,
report MAPE on the second half, and write the coefficients for SGLANG_EDF_LATENCY_MODEL=linear:<json>.

Usage: fit_mtp_step.py server.log --out mtp_step.json
"""
import argparse, json, re
import numpy as np

FEATS = ["et", "bs", "np", "ptok", "pf", "nd", "dkv", "kv", "mx"]
LINE = re.compile(
    r"STEPTRACE (\w+) "
    + " ".join(f"{f}=(-?\\d+)" for f in FEATS)
    + r"(?: acc=([0-9.]+))? t=([0-9.]+)"
)
NAMES = ["c0", "c_tok", "c_attn", "c_bs", "c_kv"]


def design(r, include_acc=False):
    np_, ptok, pf, nd, dkv = r["np"], r["ptok"], r["pf"], r["nd"], r["dkv"]
    rp = pf / np_ if np_ else 0.0
    result = [1.0, ptok, (ptok * rp + 0.5 * ptok * ptok / max(np_, 1)) / 1e6, nd, dkv / 1e3]
    if include_acc:
        result.append(float(r.get("acc") or 1.0))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap-ms", type=float, default=250.0)
    a = ap.parse_args()
    rows = []
    for m in LINE.finditer(open(a.log, errors="replace").read()):
        groups = m.groups()
        r = dict(zip(FEATS, map(int, groups[1:10])))
        r["mode"] = groups[0]
        r["acc"] = float(groups[10]) if groups[10] is not None else None
        r["t"] = float(groups[11])
        rows.append(r)
    has_acc = any(r["acc"] is not None for r in rows)
    X, y, modes = [], [], []
    for r, nxt in zip(rows, rows[1:]):
        dt = (nxt["t"] - r["t"]) * 1e3
        if 0 < dt < a.gap_ms:
            X.append(design(r, include_acc=has_acc)); y.append(dt); modes.append(r["mode"])
    X = np.array(X)
    y = np.array(y)
    h = len(y) // 2
    coef, *_ = np.linalg.lstsq(X[:h], y[:h], rcond=None)
    pred = X[h:] @ coef
    mape = float(np.mean(np.abs(pred - y[h:]) / y[h:]))
    const = float(np.mean(np.abs(np.mean(y[:h]) - y[h:]) / y[h:]))
    by = {}
    for md in sorted(set(modes)):
        idx = np.array([i for i in range(h, len(y)) if modes[i] == md], dtype=int)
        if len(idx):
            by[md] = {"n": int(len(idx)), "mape": float(np.mean(np.abs(X[idx] @ coef - y[idx]) / y[idx])),
                      "mean_ms": float(np.mean(y[idx]))}
    coef_full, *_ = np.linalg.lstsq(X, y, rcond=None)
    names = NAMES + (["c_acc"] if has_acc else [])
    res = {"coef": dict(zip(names, map(float, coef_full))), "n_steps": int(len(y)),
           "heldout_mape": mape, "heldout_mape_constant": const, "by_mode": by, "log": a.log}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
