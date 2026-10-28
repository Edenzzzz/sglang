"""Fit and SAVE a step-time model on the features STEPTRACE actually carries, and report its decode-step BIAS.

Why this exists. Paper 5.2.2 credits an "MLP with acceptance" at 3.4-4.0% on merged steps, but no such artifact is
on disk: the three checkpoints under this directory all carry `['n_p','r_p','r_d','decode_bs','overlap']` from
train_perf_model_mlp.py, with `overlap` constant (a dead input), and fit_steptrace.py fits an MLP in memory and
discards it. So the figure is unreproducible and the router cannot be given the model.

What this fits: the five features STEPTRACE logs per step -- chunk tokens, cached prefix, decode rows, their KV
length, and the speculative acceptance EWMA -- against the step's measured GPU time.

What it reports, and why the second number is the point. The linear model's per-step error on decode steps is 29%
MAPE with a -13.0% SIGNED bias, and the bias is what matters: a request's TPOT is a MEAN over its decode steps, so
independent error shrinks as 1/sqrt(n) while bias passes straight through. A scalar calibration does not remove it
(held-out bias -13.3%), which says the error is load-dependent rather than an offset. So this script is judged on
whether a model with the same inputs but a nonlinear form reduces the decode-step BIAS on a chronologically
held-out split -- not on whether it reduces MAPE.

  python fit_step_mlp_20261028.py <steptrace log> [...] [--out tasks/perf_model/step_mlp_20261028.pt]
Claude, 2026-10-28.
"""
import argparse
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from chunk_predictor import load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("logs", nargs="+")
ap.add_argument("--out", default=str(__import__("pathlib").Path(__file__).resolve().parent / "step_mlp_20261028.pt"))
ap.add_argument("--epochs", type=int, default=400)
a = ap.parse_args()

rows = load(a.logs)          # already sorted by timestamp, so the split below is chronological
FEATS = ["ptok", "pf", "nd", "dkv", "acc"]
X = np.array([[float(r[f]) for f in FEATS] for r in rows], dtype=np.float32)
y = np.array([r["gpu"] for r in rows], dtype=np.float32)
is_dec = np.array([r["ptok"] == 0 and r["nd"] > 0 for r in rows])
cut = len(rows) // 2
print(f"{len(rows)} steps ({is_dec.sum()} decode-only); chronological split at {cut}")

# log1p on inputs and target: step time spans two orders of magnitude, and the linear model's own attn term is
# already quadratic in the chunk, so a raw-scale fit is dominated by the longest prefill steps.
Xl, yl = np.log1p(X), np.log1p(y)
mu, sd = Xl[:cut].mean(0), Xl[:cut].std(0) + 1e-8
Xn = (Xl - mu) / sd

torch.manual_seed(0)
net = nn.Sequential(nn.Linear(len(FEATS), 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
xt, yt = torch.tensor(Xn[:cut]), torch.tensor(yl[:cut]).unsqueeze(1)
for epoch in range(a.epochs):
    opt.zero_grad()
    loss = nn.functional.mse_loss(net(xt), yt)
    loss.backward()
    opt.step()

with torch.no_grad():
    pred = np.expm1(net(torch.tensor(Xn)).squeeze(1).numpy())

print(f"{'split':>10}{'rows':>9}{'MAPE':>9}{'bias':>9}{'median':>9}")
for name, sel in (("train all", np.arange(len(rows)) < cut), ("HELD OUT all", np.arange(len(rows)) >= cut),
                  ("train dec", (np.arange(len(rows)) < cut) & is_dec),
                  ("HELD OUT dec", (np.arange(len(rows)) >= cut) & is_dec)):
    e = (pred[sel] - y[sel]) / np.maximum(y[sel], 1e-6)
    print(f"{name:>10}{sel.sum():>9}{100*np.abs(e).mean():>8.1f}%{100*e.mean():>8.1f}%{100*np.median(e):>8.1f}%")

torch.save({"model_state": net.state_dict(), "X_mean": mu, "X_std": sd,
            "feature_names": FEATS, "log1p_inputs": True, "log1p_target": True,
            "trained_on": a.logs, "train_rows": cut}, a.out)
print(f"saved {a.out}")
