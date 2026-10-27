"""Predict the largest prefill chunk that keeps every running decode row inside its mean-TPOT SLO.

The user's requirement (2026-10-28): no fixed chunk, and no dynamic program as in SLOs-Serve -- "if we can come up
with a simple model that predicts the chunk size for running ITL + slack (model TPOT) to be met without DP, this
would be a contribution, people like to use simple methods".

The rule is three lines of arithmetic on top of the step model we already fit to the fused kernel:

    budget_ms = (1 - h) * min over running rows r of [ s_r * (o_r + a) - E_r ]
    chunk     = max { c : T_step(c, prefix, nd, dkv) <= budget_ms }
    admit c if c >= one page, else run decode only this step

where s_r is row r's mean-TPOT target, o_r the tokens it has emitted, E_r the time it has spent, a the tokens one
step emits per row (EWMA of speculative acceptance), and h a headroom fraction. The max is found by bisection on
the step model, which needs no inversion and no DP: 16 evaluations of a 10-term linear model, about 2 us.

Why this is not a fixed chunk: when few rows are decoding, or when the rows have banked slack by running faster
than their target, the budget is large and the chunk is large -- at the start of serving, with no rows decoding at
all, the budget is unbounded and the chunk is the server's maximum. When many rows are near their target the budget
shrinks and prefill yields. Mean TPOT (not per-step ITL) is what the SLO states, so banked slack counts in full.

This module is the offline validator and the reference implementation of the rule: it replays recorded STEPTRACE
steps, applies the rule to each, and reports what chunk it would have chosen against what the run actually used.
  python chunk_predictor.py <steptrace log> [...] --model step_qwen_h20_a6_gpu_20261027.json [--chat-ms 50]
Claude, 2026-10-28.
"""
import argparse
import json
import re

import numpy as np

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
            rows.append(dict(ptok=int(g[4]), pf=int(g[5]), nd=int(g[6]), dkv=int(g[7]),
                             acc=float(g[10] or 1.0), t=float(g[11]), gpu=float(g[12])))
    rows.sort(key=lambda r: r["t"])
    return rows


class StepModel:
    """The lin4mode step model fit to the fused kernel's steps (tasks/perf_model/fit_4mode_step.py)."""

    def __init__(self, path):
        spec = json.load(open(path))
        target = "gap" if "gap" in spec else "gpu"
        self.coef = np.array(spec[target]["coef"], dtype=float)
        self.cols = spec["columns"]
        self.S, self.T = float(spec["S"]), float(spec["T"])

    def predict_ms(self, chunk, prefix, nd, dkv):
        """Predicted step time for a chunk of `chunk` tokens on `prefix` cached tokens beside `nd` decode rows."""
        attn = (chunk * prefix + 0.5 * chunk * chunk) / 1e6
        feats = {"1": 1.0, "ptok": chunk, "attn": attn, "nd": nd, "dkv/1e3": dkv / 1e3,
                 "np": 1.0 if chunk > 0 else 0.0, "mixed": 1.0 if (chunk > 0 and nd > 0) else 0.0,
                 "nd*S": nd * self.S, "nd*T": nd * self.T, "dkv*T/1e3": dkv * self.T / 1e3}
        return float(sum(self.coef[i] * feats[c] for i, c in enumerate(self.cols)))


def max_chunk(model, budget_ms, prefix, nd, dkv, hi, page=64):
    """Largest chunk whose predicted step fits `budget_ms`, by bisection. No inversion, no DP."""
    if model.predict_ms(0.0, prefix, nd, dkv) > budget_ms:
        return 0  # even a decode-only step exceeds the budget: nothing to admit
    if model.predict_ms(hi, prefix, nd, dkv) <= budget_ms:
        return int(hi)
    lo, high = 0, int(hi)
    while high - lo > page:
        mid = (lo + high) // 2
        if model.predict_ms(mid, prefix, nd, dkv) <= budget_ms:
            lo = mid
        else:
            high = mid
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--chat-ms", type=float, default=50.0, help="mean-TPOT target of the tightest tenant")
    ap.add_argument("--headroom", type=float, default=0.1)
    ap.add_argument("--max-chunk", type=float, default=16384)
    args = ap.parse_args()
    model = StepModel(args.model)
    rows = load(args.logs)
    merged = [r for r in rows if r["ptok"] > 0 and r["nd"] > 0]
    if not merged:
        raise SystemExit("no merged steps in these logs")

    # Per-row slack is not in the trace, so the validator uses the step-level stand-in the rule reduces to when
    # every row sits exactly at its target: budget = s * a, the per-step allowance. This is the PESSIMISTIC end of
    # the rule (no banked slack), so the chunks below are a lower bound on what the rule would admit online.
    chosen, actual, pred_ms, real_ms = [], [], [], []
    for r in merged:
        a = max(1.0, r["acc"])
        budget = (1 - args.headroom) * args.chat_ms * a
        c = max_chunk(model, budget, r["pf"], r["nd"], r["dkv"], args.max_chunk)
        chosen.append(c)
        actual.append(r["ptok"])
        pred_ms.append(model.predict_ms(r["ptok"], r["pf"], r["nd"], r["dkv"]))
        real_ms.append(r["gpu"])
    chosen, actual = np.array(chosen, float), np.array(actual, float)
    pred_ms, real_ms = np.array(pred_ms), np.array(real_ms)
    ape = np.abs(pred_ms - real_ms) / np.maximum(real_ms, 1e-3)
    print(f"{len(merged)} merged steps; step-model error on the chunk the run actually used: "
          f"MAPE {100 * ape.mean():.1f}% p90 {100 * np.percentile(ape, 90):.1f}%")
    print(f"per-step allowance at {args.chat_ms:.0f} ms mean TPOT and acceptance "
          f"{np.median([r['acc'] for r in merged]):.2f}: "
          f"{(1 - args.headroom) * args.chat_ms * np.median([max(1.0, r['acc']) for r in merged]):.0f} ms")
    print(f"chunk the rule would pick (no banked slack, the pessimistic bound): "
          f"p10/p50/p90 {np.percentile(chosen, 10):.0f}/{np.percentile(chosen, 50):.0f}/"
          f"{np.percentile(chosen, 90):.0f} tokens")
    print(f"chunk the run actually used (fixed 2048 policy):               "
          f"p10/p50/p90 {np.percentile(actual, 10):.0f}/{np.percentile(actual, 50):.0f}/"
          f"{np.percentile(actual, 90):.0f} tokens")
    print(f"steps where the rule would admit MORE than the fixed policy: {100 * np.mean(chosen > actual):.1f}%")
    print(f"steps where it would admit LESS (protecting TPOT):            {100 * np.mean(chosen < actual):.1f}%")
    print(f"steps where it would admit nothing:                           {100 * np.mean(chosen == 0):.1f}%")
    # What the fixed chunk costs: predicted step time against the allowance it is supposed to respect.
    over = pred_ms / np.maximum((1 - args.headroom) * args.chat_ms *
                                np.array([max(1.0, r["acc"]) for r in merged]), 1e-9)
    print(f"the fixed 2048 chunk's predicted step costs {np.median(over):.1f}x the per-step allowance "
          f"(p90 {np.percentile(over, 90):.1f}x), which is the TPOT damage the rule removes")
    for lo, hi, name in ((0, 8192, "<8K"), (8192, 32768, "8-32K"), (32768, 131072, "32-128K"),
                         (131072, 10 ** 9, ">128K")):
        sel = [i for i, r in enumerate(merged) if lo <= r["pf"] < hi]
        if len(sel) >= 50:
            print(f"  prefix {name:>7}: rule picks p50 {np.percentile(chosen[sel], 50):7.0f} tokens, "
                  f"nd p50 {np.median([merged[i]['nd'] for i in sel]):4.0f} rows")


if __name__ == "__main__":
    main()
