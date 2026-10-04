#!/usr/bin/env python3
"""How predictable is a server step, fused kernel vs unfused, across model classes.

Goal point 3 claims the fused attention kernel makes EDF's step-cost prediction
*easier*. A 3-parameter line cannot represent this kernel's cost, which contains
a makespan term max(FA3 role, FA2 role) over a static SM partition, so a linear
fit measures the model class and not the kernel. This script therefore fits the
same trace with six classes, from that line up to a gated mixture of linear
experts -- a mixture of two experts can represent a max of two linear functions
exactly, which is the structure the kernel actually has.

Input: server logs carrying the enriched per-step trace (SGLANG_STEP_TRACE=1),

  STEPTRACE MIXED et=512 bs=10 np=1 ptok=503 pf=46589 nd=9 dkv=512 kv=47101 mx=503 t=...

Target is the launch-to-launch interval, which is what an EDF admission decision
has to predict, and which adds no synchronisation to the measured path.

Usage: fit_steptrace.py fused=tr2_s8.log baseline=tr2_base.log [--mode MIXED]
"""
import argparse, re, sys
import numpy as np

FEATS = ["et", "bs", "np", "ptok", "pf", "nd", "dkv", "kv", "mx"]
LINE = re.compile(r"STEPTRACE (\w+) " + " ".join(f"{f}=(-?\\d+)" for f in FEATS) + r" t=([0-9.]+)")


def parse(path):
    """Rows of (mode, features, dt_ms). dt is to the NEXT step of any mode."""
    recs = []
    for ln in open(path, errors="replace"):
        m = LINE.search(ln)
        if m:
            recs.append((m.group(1), [int(m.group(i + 2)) for i in range(len(FEATS))],
                         float(m.group(len(FEATS) + 2))))
    out = []
    for (mode, f, t), (_, _, t_next) in zip(recs, recs[1:]):
        out.append((mode, f, (t_next - t) * 1e3))
    return out


def select(rows, mode, lo=0.05, hi=1000.0):
    """Keep steps of one mode whose interval is a step and not an idle gap."""
    sel = [(f, dt) for (m, f, dt) in rows if (mode is None or m == mode) and lo <= dt <= hi]
    X = np.array([f for f, _ in sel], dtype=np.float64)
    y = np.array([dt for _, dt in sel], dtype=np.float64)
    return X, y


def metrics(y, p):
    e = np.abs(p - y) / np.maximum(y, 1e-9) * 100.0
    return float(e.mean()), float(np.percentile(e, 90))


# ---------------------------------------------------------------- model classes
def fit_linear(Xtr, ytr, Xte, cols=None):
    if cols is not None:
        Xtr, Xte = Xtr[:, cols], Xte[:, cols]
    A = np.hstack([np.ones((len(Xtr), 1)), Xtr])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.hstack([np.ones((len(Xte), 1)), Xte]) @ w


def fit_rf(Xtr, ytr, Xte):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def fit_gbt(Xtr, ytr, Xte):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=0)
    m.fit(Xtr, np.log1p(ytr))
    return np.expm1(m.predict(Xte))


def _torch_fit(Xtr, ytr, Xte, build, epochs, lr, log_target):
    import torch
    torch.manual_seed(0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    xe = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    tt = torch.tensor(np.log1p(ytr) if log_target else ytr, dtype=torch.float32)
    net = build(Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        pred = net(xt)
        # relative loss: MAPE is the reported metric, so optimise its smooth kin
        loss = ((pred - tt).abs() / tt.abs().clamp_min(1e-6)).mean()
        loss.backward()
        opt.step()
        sch.step()
    with torch.no_grad():
        p = net(xe).numpy()
    return np.expm1(p) if log_target else p


def fit_mlp(Xtr, ytr, Xte):
    """The project's perf-model architecture: [128, 64, 32], log1p target."""
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            layers, prev = [], d
            for h in (128, 64, 32):
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return _torch_fit(Xtr, ytr, Xte, MLP, 400, 2e-3, True)


def fit_moe(Xtr, ytr, Xte, n_experts=4):
    """Gated mixture of LINEAR experts: the class that contains max(a.x, b.x).

    Each expert is one affine cost model (one role saturating); the gate picks
    which role's makespan dominates. This is the smallest class that can
    represent the fused kernel's cost exactly, so it separates "hard to model"
    from "hard for a line".
    """
    import torch
    import torch.nn as nn

    class MoE(nn.Module):
        def __init__(self, d, k):
            super().__init__()
            self.experts = nn.Linear(d, k)
            self.gate = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, k))

        def forward(self, x):
            w = torch.softmax(self.gate(x), dim=-1)
            return (w * self.experts(x)).sum(-1)

    return _torch_fit(Xtr, ytr, Xte, lambda d: MoE(d, n_experts), 3000, 5e-3, False)


MODELS = {
    "linear(tokens,bs)": lambda a, b, c: fit_linear(a, b, c, cols=[0, 1]),
    "linear(all)": fit_linear,
    "random forest": fit_rf,
    "grad boosting": fit_gbt,
    "MLP [128,64,32]": fit_mlp,
    "MoE-4 linear": fit_moe,
}


def cv(X, y, fn, folds=5):
    """Out-of-fold predictions on shuffled folds, identical across arms."""
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(y))
    pred = np.zeros_like(y)
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te, assume_unique=False)
        pred[te] = fn(X[tr], y[tr], X[te])
    return metrics(y, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="name=path")
    ap.add_argument("--mode", default="MIXED", help="step mode to fit, or ALL")
    args = ap.parse_args()

    arms = {}
    for spec in args.arms:
        name, path = spec.split("=", 1)
        arms[name] = parse(path)

    mode = None if args.mode == "ALL" else args.mode
    data = {}
    for name, rows in arms.items():
        X, y = select(rows, mode)
        data[name] = (X, y)
        print(f"{name}: {len(y)} {args.mode} steps, "
              f"mean {y.mean():.2f} ms, p50 {np.median(y):.2f}, p90 {np.percentile(y, 90):.2f}")
    print()
    w = max(len(k) for k in MODELS) + 2
    print(f"{'model class':<{w}}" + "".join(f"{n+' MAPE':>16}{n+' p90':>15}" for n in data))
    for label, fn in MODELS.items():
        cells = ""
        for name, (X, y) in data.items():
            if len(y) < 50:
                cells += f"{'n/a':>16}{'':>15}"
                continue
            mape, p90 = cv(X, y, fn)
            cells += f"{mape:>15.2f}%{p90:>14.2f}%"
        print(f"{label:<{w}}" + cells)


if __name__ == "__main__":
    main()
