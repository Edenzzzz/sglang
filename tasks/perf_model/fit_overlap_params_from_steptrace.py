"""Fit the analytic overlap cost model's per-GPU constants from real serving steps.

`managers/overlap_cost_model.py` predicts a forward step as

    T_d(layer) = a_d * decode_kv + c_d                 (decode attention, bandwidth-bound)
    T_p(layer) = a_p * (n*prefix + n^2/2) + c_p        (chunk attention, compute-bound)
    T_attn     = max(T_p, T_d) + beta * min(T_p, T_d)  (the fused kernel's overlap law)
    T_step     = c0 + c1 * (n + decode_bs) + L * T_attn

The shipped constants were fit on an H800 from microbenchmarks. This fits the same seven constants to STEPTRACE
lines of a real run (any GPU, any model), so the EDF chunk solver (edf_compute_max_chunk_size, which binary-searches
the largest chunk whose predicted step fits the tightest row's slack) can run with
SGLANG_EDF_LATENCY_MODEL=analytic and SGLANG_OVERLAP_COST_PARAMS=<out>.

Fit on the first half of the steps in time order, scored on the second half, per step type, on relative error.
  python fit_overlap_params_from_steptrace.py <steptrace log> [...] --layers 28 --out params.json
Claude, 2026-10-27.
"""
import argparse
import json
import re

import numpy as np
import torch

PAT = re.compile(r"STEPTRACE (\w+) et=(\d+) bs=(\d+) np=(\d+) ptok=(\d+) pf=(\d+) nd=(\d+) dkv=(\d+) kv=(\d+) "
                 r"mx=(\d+)(?: sg=\d+ lay=\d+)?(?: acc=([0-9.]+))?(?: dacc=[0-9.]+)? t=([0-9.]+) gpu=([0-9.]+)")


def load(paths):
    rows = []
    for path in paths:
        for line in open(path):
            m = PAT.search(line)
            if not m:
                continue
            g = m.groups()
            gpu = float(g[12])
            if gpu <= 0:
                continue
            rows.append((float(g[4]), float(g[5]), float(g[6]), float(g[7]), float(g[11]), gpu))
    rows.sort(key=lambda r: r[4])  # time order
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    args = parser.parse_args()
    rows = load(args.logs)
    if len(rows) < 200:
        raise SystemExit(f"only {len(rows)} usable steps")
    n_p = torch.tensor([r[0] for r in rows], dtype=torch.float64)
    pf = torch.tensor([r[1] for r in rows], dtype=torch.float64)
    nd = torch.tensor([r[2] for r in rows], dtype=torch.float64)
    dkv = torch.tensor([r[3] for r in rows], dtype=torch.float64)
    y = torch.tensor([r[5] for r in rows], dtype=torch.float64)
    work = n_p * pf + 0.5 * n_p * n_p
    half = len(rows) // 2
    L = float(args.layers)

    # Positive parameters through exp; beta in [0, 1] through a sigmoid.
    init = torch.tensor([np.log(6.5e-7), np.log(0.04), np.log(4.0e-8), np.log(0.2), 0.0, np.log(5.0), np.log(0.025)],
                        dtype=torch.float64, requires_grad=True)

    def predict(p, sl):
        a_d, c_d, a_p, c_p = p[0].exp(), p[1].exp(), p[2].exp(), p[3].exp()
        beta, c0, c1 = torch.sigmoid(p[4]), p[5].exp(), p[6].exp()
        t_d = torch.where(dkv[sl] > 0, a_d * dkv[sl] + c_d, torch.zeros_like(dkv[sl]))
        t_p = torch.where(n_p[sl] > 0, a_p * work[sl] + c_p, torch.zeros_like(n_p[sl]))
        both = (t_d > 0) & (t_p > 0)
        attn = torch.where(both, torch.maximum(t_p, t_d) + beta * torch.minimum(t_p, t_d), t_p + t_d)
        return c0 + c1 * (n_p[sl] + nd[sl]) + L * attn

    train = slice(0, half)
    opt = torch.optim.Adam([init], lr=0.05)
    for _ in range(args.steps):
        opt.zero_grad()
        loss = ((predict(init, train) - y[train]).abs() / y[train].clamp_min(1e-3)).mean()
        loss.backward()
        opt.step()
    p = init.detach()
    params = dict(a_d=float(p[0].exp()), c_d=float(p[1].exp()), a_p=float(p[2].exp()), c_p=float(p[3].exp()),
                  beta=float(torch.sigmoid(p[4])), c0=float(p[5].exp()), c1=float(p[6].exp()), n_layers=args.layers)
    test = slice(half, len(rows))
    pred = predict(p, test).numpy()
    truth = y[test].numpy()
    ape = np.abs(pred - truth) / truth
    kinds = np.array(["merged" if r[0] > 0 and r[2] > 0 else ("prefill" if r[0] > 0 else "decode")
                      for r in rows[half:]])
    report = dict(params=params, n_steps=len(rows), heldout={})
    for kind in ("all", "decode", "merged", "prefill"):
        sel = slice(None) if kind == "all" else (kinds == kind)
        if kind != "all" and not sel.any():
            continue
        report["heldout"][kind] = dict(n=int(len(ape[sel])), mape_percent=float(ape[sel].mean() * 100),
                                       p90_percent=float(np.percentile(ape[sel], 90) * 100))
    json.dump(params, open(args.out, "w"), indent=1)
    print(json.dumps(report, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
