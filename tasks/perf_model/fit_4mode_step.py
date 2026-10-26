#!/usr/bin/env python3
"""Which step model predicts the 4-mode mixed step best? Candidates fit on STEPTRACE logs (SGLANG_STEP_TRACE=1), first
half of each trace in time order, scored on the second half: MAPE and p90 APE per step kind (decode = verify/decode
rows only, mixed = a prefill chunk beside decode rows, prefill = prefill rows only), for the forward's GPU time and for
the launch-to-launch gap of back-to-back steps (< 200 ms, what ITL sees).

Features per step (pdmux logs also carry sg = SM group and lay = layers in a split forward, used by mlp / gbt):
ptok (prefill tokens), pf (their summed prefix), np (prefill requests), nd (decode rows), dkv (decode
KV tokens), kv (total KV), with attn = (ptok * pf / np + ptok^2 / (2 np)) / 1e6 the causal-attention work of the chunk.
  lin5      ms = c0 + c_tok*ptok + c_attn*attn + c_bs*nd + c_kv*dkv            (pdmux's form, fit_pdmux_step.py)
  lin7      lin5 + c_np*np + c_mixed*[np>0 and nd>0]                           (per-request and mixed-step overhead)
  lin4mode  lin7 + c_draft*nd*S + c_ver*nd*T + c_vkv*dkv*T                     (all four modes: S draft steps and T
            verify tokens per decode row, from the log's speculative config; 0 / 0 without speculation)
  overlap   ms = c0 + max(Tp, Td) + beta*min(Tp, Td), Tp = c_tok*ptok + c_attn*attn,
            Td = c_bs*nd + c_kv*dkv + c_draft*nd*S + c_ver*nd*T + c_vkv*dkv*T (the lin4mode decode terms)
            (FA2 decode rows run beside FA3 prefill; fit by gradient descent on relative error)
  mlp       3-layer MLP on log1p(features), log target
  gbt       gradient-boosted trees (scikit-learn), log target
  *_acc     lin4mode, mlp and gbt with logged pre-step acceptance and nd*acceptance
Usage: fit_4mode_step.py [--target gpu|gap|both] [--pool] <server.log>[@S:T] [...]
  S/T come from the logged speculative configuration unless explicitly overridden by @S:T.
  Missing acceptance stays distinguishable from a measured acceptance of one; *_acc fits are skipped without it.
  Earlier fits without @S:T used zero draft/verify counts and ignored acc; rerun them before making modeling claims.
  each log is fit and scored on its own; --pool also fits one model on the first halves of all logs and scores it on
  every log's second half (the draft / verify terms are only identifiable across speculative configs).
Claude, 2026-10-19.
"""
import argparse
import re

import numpy as np
import torch

PAT = re.compile(r"STEPTRACE (\w+) et=(\d+) bs=(\d+) np=(\d+) ptok=(\d+) pf=(\d+) nd=(\d+) dkv=(\d+) kv=(\d+) "
                 r"mx=(\d+)(?: sg=(\d+) lay=(\d+))?(?: acc=([0-9.]+))?(?: dacc=[0-9.]+)? t=([0-9.]+) gpu=([0-9.]+)")
GAP_MS = 200.0


def speculative_config(text, override):
    if override:
        values = override.split(":")
        if len(values) != 2:
            raise ValueError("Speculative override must be @draft_steps:verify_tokens")
        config = tuple(int(value) for value in values)
        if min(config) < 0:
            raise ValueError("Speculative counts must be nonnegative")
        return config
    config = []
    for name in ("speculative_num_steps", "speculative_num_draft_tokens"):
        values = {int(value) for value in re.findall(rf"\b{name}\b['\"]?\s*(?:=|:)\s*(\d+)", text)}
        if len(values) > 1:
            raise ValueError(f"Conflicting {name} values; split the trace or supply @S:T")
        config.append(next(iter(values)) if values else None)
    if all(value is None for value in config):
        return 0, 0
    if any(value is None for value in config):
        raise ValueError("Incomplete speculative configuration; supply @S:T")
    return tuple(config)


def load(spec):
    path, _, st = spec.partition("@")
    with open(path, errors="replace") as source:
        text = source.read()
    steps, toks = speculative_config(text, st)
    feature_sources = set(re.findall(r"STEPTRACE_FEATURES source=(\w+)", text))
    if len(feature_sources) > 1:
        raise ValueError("Multiple STEPTRACE feature sources in one log")
    feature_source = next(iter(feature_sources), "legacy")
    rs = []
    for line in text.splitlines():
        m = PAT.search(line)
        if not m:
            continue
        g = m.groups()
        acceptance = float(g[12]) if g[12] is not None else 1.0
        if not np.isfinite(acceptance) or acceptance <= 0:
            raise ValueError("STEPTRACE acceptance must be positive and finite")
        rs.append(dict(np=int(g[3]), ptok=int(g[4]), pf=int(g[5]), nd=int(g[6]), dkv=int(g[7]), kv=int(g[8]),
                       sg=int(g[10] or 0), lay=int(g[11] or 0), t=float(g[13]) * 1e3, gpu=float(g[14]), S=steps,
                       T=toks, acc=acceptance, acc_known=g[12] is not None, feature_source=feature_source))
    if not steps and not toks and any(row["acc"] > 1 for row in rs):
        raise ValueError("Speculative acceptance is logged but S/T are unknown; supply @S:T")
    rs.sort(key=lambda r: r["t"])
    for a, b in zip(rs, rs[1:]):
        iv = b["t"] - a["t"]
        a["gap"] = iv if 0 < iv < GAP_MS else None
    if rs:
        rs[-1]["gap"] = None
    for r in rs:
        r["attn"] = (r["ptok"] * r["pf"] / max(r["np"], 1) + 0.5 * r["ptok"] ** 2 / max(r["np"], 1)) / 1e6
        r["kind"] = "mixed" if r["np"] and r["nd"] else ("prefill" if r["np"] else "decode")
    if rs and re.search(r"MIXEDSPEC_STEPS [1-9]\d*", text) and not any(row["kind"] == "mixed" for row in rs):
        raise ValueError("Mixed-speculative execution is logged but STEPTRACE has no mixed rows; recollect comparable features")
    return rs


def design(rs, form):
    cols = [np.ones(len(rs)), [r["ptok"] for r in rs], [r["attn"] for r in rs], [r["nd"] for r in rs],
            [r["dkv"] / 1e3 for r in rs]]
    if form in ("lin7", "lin4mode", "lin4mode_acc"):
        cols += [[r["np"] for r in rs], [float(r["np"] > 0 and r["nd"] > 0) for r in rs]]
    if form in ("lin4mode", "lin4mode_acc"):
        cols += [[r["nd"] * r["S"] for r in rs], [r["nd"] * r["T"] for r in rs], [r["dkv"] * r["T"] / 1e3 for r in rs]]
    if form == "lin4mode_acc":
        cols += [[row["acc"] for row in rs], [row["nd"] * row["acc"] for row in rs]]
    return np.array(cols, dtype=float).T


def fit_linear(tr, y, form):
    # Least squares on relative error (rows weighted by 1/y), the quantity MAPE scores.
    X = design(tr, form)
    w = 1.0 / np.maximum(y, 1e-3)
    coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    return lambda rs: design(rs, form) @ coef


def fit_overlap(tr, y):
    X = torch.tensor(design(tr, "lin4mode"), dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    lin = np.linalg.lstsq(design(tr, "lin4mode"), y, rcond=None)[0]
    p = torch.tensor(np.maximum(lin, 1e-6), dtype=torch.float64).log().requires_grad_()
    beta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([p, beta], lr=0.02)

    def pred(Xm):
        c = p.exp()
        tp = Xm[:, 1] * c[1] + Xm[:, 2] * c[2]
        td = Xm[:, 3] * c[3] + Xm[:, 4] * c[4] + Xm[:, 7] * c[7] + Xm[:, 8] * c[8] + Xm[:, 9] * c[9]
        return c[0] + torch.maximum(tp, td) + torch.sigmoid(beta) * torch.minimum(tp, td)

    for _ in range(3000):
        opt.zero_grad()
        loss = ((pred(X) - yt).abs() / yt.clamp_min(1e-3)).mean()
        loss.backward()
        opt.step()
    return lambda rs: pred(torch.tensor(design(rs, "lin4mode"), dtype=torch.float64)).detach().numpy()


def feats(rs, include_acceptance=False):
    values = [[row["ptok"], row["pf"], row["np"], row["nd"], row["dkv"], row["kv"], row["attn"] * 1e3,
               row["S"], row["T"], row["sg"], row["lay"]] for row in rs]
    if include_acceptance:
        for features, row in zip(values, rs):
            features.extend((row["acc"], row["nd"] * row["acc"]))
    return np.log1p(np.array(values, dtype=float))


def fit_mlp(tr, y, include_acceptance=False):
    torch.manual_seed(0)
    Xn = feats(tr, include_acceptance)
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-6
    X = torch.tensor((Xn - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(np.log(np.maximum(y, 1e-3)), dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 32),
                              torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    for _ in range(2000):
        opt.zero_grad()
        loss = ((net(X).squeeze(1) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    return lambda rs: np.exp(net(torch.tensor((feats(rs, include_acceptance) - mu) / sd, dtype=torch.float32)).squeeze(1)
                             .detach().numpy())


def fit_gbt(tr, y, include_acceptance=False):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
    m.fit(feats(tr, include_acceptance), np.log(np.maximum(y, 1e-3)))
    return lambda rs: np.exp(m.predict(feats(rs, include_acceptance)))


FITTERS = {
    "lin5": lambda tr, y: fit_linear(tr, y, "lin5"),
    "lin7": lambda tr, y: fit_linear(tr, y, "lin7"),
    "lin4mode": lambda tr, y: fit_linear(tr, y, "lin4mode"),
    "lin4mode_acc": lambda rows, targets: fit_linear(rows, targets, "lin4mode_acc"),
    "overlap": fit_overlap,
    "mlp": fit_mlp,
    "gbt": fit_gbt,
    "mlp_acc": lambda rows, targets: fit_mlp(rows, targets, include_acceptance=True),
    "gbt_acc": lambda rows, targets: fit_gbt(rows, targets, include_acceptance=True),
}


def score(pred, rs, target):
    out = {}
    for kind in ("decode", "mixed", "prefill"):
        ks = [r for r in rs if r["kind"] == kind and r[target] is not None]
        if len(ks) < 20:
            continue
        y = np.array([r[target] for r in ks])
        ape = np.abs(pred(ks) - y) / np.maximum(y, 1e-3)
        out[kind] = (len(ks), ape.mean() * 100, np.percentile(ape, 90) * 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="both", choices=["gpu", "gap", "both"])
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--export-lin4mode", default=None, metavar="JSON",
                    help="fit lin4mode on every step of all logs (target gap, else gpu) and write its coefficients, "
                    "the column order and S / T for the scheduler's SLO-budgeted chunk (apply_slo_budget_chunk)")
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()
    targets = ["gpu", "gap"] if a.target == "both" else [a.target]
    splits = []
    loaded = [(spec, load(spec)) for spec in a.logs]
    if len({row["feature_source"] for _, rows in loaded for row in rows}) > 1:
        raise ValueError("Logs have incompatible feature sources; recollect matched traces")
    for spec, rs in loaded:
        print(f"== {spec}: {len(rs)} steps "
              + " ".join(f"{k}={sum(r['kind'] == k for r in rs)}" for k in ("decode", "mixed", "prefill")))
        if rs:
            print(f"  features={rs[0]['feature_source']} config S={rs[0]['S']} T={rs[0]['T']} acceptance_logged={sum(row['acc_known'] for row in rs)}/{len(rs)} "
                  f"acceptance_range={min(row['acc'] for row in rs):.4f}..{max(row['acc'] for row in rs):.4f}")
        if len(rs) < 200:
            continue
        half = len(rs) // 2
        splits.append((spec, rs[:half], rs[half:]))
        for target in targets:
            run(rs[:half], {spec: rs[half:]}, target)
    if a.export_lin4mode:
        export_lin4mode([r for _, rows in loaded for r in rows], a.export_lin4mode)
    if a.pool and len(splits) > 1:
        print("== pooled fit over", len(splits), "logs")
        for target in targets:
            run([r for _, tr, _ in splits for r in tr], {spec: te for spec, _, te in splits}, target)


def export_lin4mode(rows, path):
    import json
    configs = {(r["S"], r["T"]) for r in rows}
    if len(configs) != 1:
        raise ValueError(f"export needs one speculative configuration, got {sorted(configs)}")
    result = {}
    for target in ("gap", "gpu"):
        tr = [r for r in rows if r[target] is not None]
        if len(tr) < 200:
            continue
        y = np.array([r[target] for r in tr])
        X = design(tr, "lin4mode")
        w = 1.0 / np.maximum(y, 1e-3)
        coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
        ape = np.abs(X @ coef - y) / y
        result[target] = dict(coef=[float(c) for c in coef], steps=len(tr), in_sample_mape=float(ape.mean() * 100))
    if not result:
        raise ValueError("too few steps to export")
    S, T = configs.pop()
    json.dump(dict(form="lin4mode", columns=["1", "ptok", "attn", "nd", "dkv/1e3", "np", "mixed", "nd*S", "nd*T",
                                             "dkv*T/1e3"], S=S, T=T, **result), open(path, "w"), indent=1)
    print(f"EXPORTED lin4mode -> {path}: " + ", ".join(f"{k} MAPE {v['in_sample_mape']:.1f}% over {v['steps']} steps"
                                                       for k, v in result.items()))


def run(train, tests, target):
    tr = [r for r in train if r[target] is not None]
    y = np.array([r[target] for r in tr])
    for name, fit in FITTERS.items():
        if name.endswith("_acc") and not all(row["acc_known"] for row in tr):
            print(f"  {target:4s} {name:12s} SKIPPED incomplete acceptance features")
            continue
        try:
            pred = fit(tr, y)
        except Exception as exc:  # a candidate that cannot fit is reported, not dropped silently
            print(f"  {target:4s} {name:8s} FAILED {type(exc).__name__}: {exc}")
            continue
        for spec, te in tests.items():
            if name.endswith("_acc") and not all(row["acc_known"] for row in te if row[target] is not None):
                print(f"  {target:4s} {name:12s} [{spec}] SKIPPED incomplete acceptance features")
                continue
            s = score(pred, te, target)
            cells = "  ".join(f"{k} n={n} MAPE={m:5.1f}% p90={p:5.1f}%" for k, (n, m, p) in s.items())
            tag = "" if len(tests) == 1 else f" [{spec.split('/')[-2] if '/' in spec else spec}]"
            print(f"  {target:4s} {name:8s}{tag} {cells}")


if __name__ == "__main__":
    main()
