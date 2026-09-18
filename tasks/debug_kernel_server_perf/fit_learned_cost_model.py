#!/usr/bin/env python3
"""Head-to-head: can a learned model predict BatchPrefill's mixed-batch cost
as well as it predicts the persistent kernel's?

The analytic law (max + beta*min) fails for BatchPrefill (fitted beta=2.19).
That could mean BP is genuinely irregular, or merely that the functional form
is wrong. This fits the SAME model class on the SAME features to both targets
and compares held-out error: if BP is only "hard for that form", an MLP should
close the gap; if its cost surface has planner cliffs, BP stays worse.

Data: the gate sweeps (composition -> measured kernel ms for both kernels).
"""
import argparse
import json

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def load(paths, planner_feats=False, gqa=7):
    """planner_feats: add the variables fa2's planner actually branches on --
    the batch-average packed qo length (which picks the CTA tile) and the
    tile it implies. If BP's extra error is the planner's cliffs, these
    should recover it."""
    X, bp, pa = [], [], []
    for p in paths:
        for r in json.load(open(p)):
            kvs = r["dec_kvs"] if "dec_kvs" in r else [r["d_kv"] - 1] * r["d_bs"]
            D = float(sum(kvs))
            bs = len(kvs)
            conc = sum(k * k for k in kvs) / max(D, 1.0)
            n_p = float(sum(r["p_lens"]))
            work = float(sum(q * c for q, c in zip(r["p_lens"], r["p_prefix"])))
            f = [np.log1p(D), np.log1p(work), np.log1p(n_p), np.log1p(bs),
                 3.0 * D / (D + work + 1.0), np.log1p(conc)]
            if planner_feats:
                # FA2DetermineCtaTileQ branches on the batch-average packed
                # qo length: decode rows contribute gqa, each chunk q*gqa.
                nrows = bs + len(r["p_lens"])
                packed = (bs * gqa + sum(q * gqa for q in r["p_lens"])) / nrows
                tile = 128 if packed > 64 else (64 if packed > 16 else 16)
                f += [np.log1p(packed), float(tile), float(len(r["p_lens"])),
                      np.log1p(max(r["p_lens"]))]
            X.append(f)
            bp.append(r["bp_ms"])
            pa.append(min(r["pa_flip_ms"], r["pa_noflip_ms"]))
    return np.array(X), np.array(bp), np.array(pa)


def cv(model_fn, X, y, seed=0, folds=5):
    """K-fold CV in log space; report relative error on held-out points."""
    errs = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        sc = StandardScaler().fit(X[tr])
        m = model_fn()
        m.fit(sc.transform(X[tr]), np.log(y[tr]))
        pred = np.exp(m.predict(sc.transform(X[te])))
        errs[te] = np.abs(pred - y[te]) / y[te]
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[
        "tasks/debug_kernel_server_perf/gate_sweep_merged_qwen25_h800.json",
        "tasks/debug_kernel_server_perf/gate_sweep_blend_qwen25_h800.json"])
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    for planner in (False, True):
        run(a, planner)


def run(a, planner):
    X, bp, pa = load(a.data, planner_feats=planner)
    print(f"\n=== features: {'composition + PLANNER variables' if planner else 'composition only'} "
          f"({X.shape[1]}) , n={len(X)} ===")
    models = {
        "MLP (128,64,32)": lambda: MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=4000,
                                                random_state=0, early_stopping=False),
        "GBDT (400 trees)": lambda: GradientBoostingRegressor(n_estimators=400, random_state=0),
    }
    print(f"{'model':<18} {'target':<13} {'MAPE':>7} {'p95':>7} {'max':>8}")
    for name, fn in models.items():
        for tag, y in (("BatchPrefill", bp), ("persistent", pa)):
            e = np.concatenate([cv(fn, X, y, seed=s) for s in range(a.seeds)])
            print(f"{name:<18} {tag:<13} {e.mean()*100:6.1f}% {np.percentile(e,95)*100:6.1f}% "
                  f"{e.max()*100:7.1f}%")


if __name__ == "__main__":
    main()
