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

Three forms of the budget are implemented, because the choice of form turned out to decide whether the rule helps
or hurts (`--rule`):

  step   budget = (1 - h) * s * a. The per-step allowance: every step must fit the time one step of output is
         "worth" at the target rate. This is what the measured cluster arm used, and it is a large regression,
         because the budget is a minimum over resident rows and a row admitted this step has banked no slack, so
         its bare allowance pins the whole step no matter how much slack the other rows hold.
  life   budget = (1 - h) * min_r [ s_r * L_r - E_r - (n_r - a) * rate_r ]: the step may take whatever keeps every
         row's mean TPOT at COMPLETION inside its target, assuming the rest of its output runs at the rate it has
         been getting. This is the correct reading of a mean-TPOT SLO -- one slow step costs a row's mean only
         1/n of its excess -- but it is loose by exactly the margin by which the server beats its SLO, and it
         permits spending a whole request's slack on one step.
  amort  budget = (1 - h) * min_r [ (s_r * L_r - E_r) / n_r ] * a: the remaining slack amortized over the
         remaining output, so a banked surplus raises the allowance without ever allowing one step to spend it
         all. Equal to `step` for a row exactly on target and strictly larger for a row running ahead of it.
  tail   budget = (1 - h) * min( the `life` budget, the stated inter-token tail target ). The mean-TPOT budget is
         what the SLO states, but it stops binding as soon as the server beats its target by any margin (see
         below), and some bound must remain or one step swallows a whole request's slack. The inter-token tail
         target is a second SLO the workload states, it binds at every load, and because the chunk is still solved
         from the step model it is not a fixed token count: with no rows resident the bound is vacuous and the
         chunk is the server's maximum; with many long rows it shrinks.

**What this validator established about the three unbounded forms (2026-10-28, 10 185 merged steps of the alpha-3
replay, a 13.9 ms/token observed rate against a 50 ms target).** `step` grants 89-91 ms and picks 64-320 token
chunks, admitting less than the run actually used on 99.0% of steps -- the measured cluster regression. `life`
grants 16.3 s and saturates the 16 384-token maximum on 100% of steps -- completely inert. `amort` is identical to
`step`, because its minimum is always attained at a freshly admitted row, whose amortized allowance is exactly
s*a by construction. Sweeping the target shows the usable band of `life` is about 1.5 ms wide: it picks 1-4 K
chunks only at a 14.5-16 ms target, i.e. within 8-15% of the rate the rows already get, and is vacuous above that.
Any budget that is a minimum over resident rows and must hold for a row admitted this step therefore either
collapses to the bare per-step allowance or must count that row's whole future, at which point it stops bounding
anything -- which is why the bound that binds has to be the tail target.

This module is the offline validator and the reference implementation of the rule: it replays recorded STEPTRACE
steps, applies the rule to each, and reports what chunk it would have chosen against what the run actually used.
  python chunk_predictor.py <steptrace log> [...] --model step_qwen_h20_a6_gpu_20261027.json [--chat-ms 50]
                            [--rule step|life|amort] [--out-len 500]
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


def budget_ms(rule, chat_ms, headroom, a_tok, out_len, rate_ms, itl_ms=150.0,
              fracs=(0.0, 0.25, 0.5, 0.75, 0.95)):
    """The budget each rule grants this step, as a minimum over rows at different points in their output.

    A row that has emitted a fraction `f` of its `out_len` tokens at `rate_ms` per token has spent f*L*rate and
    has n = (1-f)*L tokens left. The trace does not record per-row progress, so the rules are evaluated over a
    spread of `f` and the minimum is taken, which is what the online rule computes over its resident rows.
    """
    k = 1.0 - headroom
    if rule == "step":
        return k * chat_ms * a_tok
    out = []
    for f in fracs:
        spent, n = f * out_len * rate_ms, max(out_len * (1.0 - f), a_tok)
        if rule in ("life", "tail"):
            life = chat_ms * out_len - spent - max(n - a_tok, 0.0) * rate_ms
            out.append(life if rule == "life" else min(life, itl_ms))
        elif rule == "amort":
            out.append((chat_ms * out_len - spent) / n * a_tok)
        else:
            raise SystemExit(f"unknown rule {rule}")
    return k * max(min(out), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--chat-ms", type=float, default=50.0, help="mean-TPOT target of the tightest tenant")
    ap.add_argument("--headroom", type=float, default=0.1)
    ap.add_argument("--max-chunk", type=float, default=16384)
    ap.add_argument("--rule", default="step", choices=("step", "life", "amort", "tail"))
    ap.add_argument("--itl-ms", type=float, default=150.0,
                    help="inter-token tail target, the bound that binds under rule=tail")
    ap.add_argument("--model-error", type=float, default=0.0,
                    help="scale the step model's prediction by (1+e), to see how far the chosen chunk moves when "
                         "the model is wrong by e. This is what decides whether a more accurate model is worth "
                         "fitting: the rule does not need the step time, only the chunk where it crosses the "
                         "budget.")
    ap.add_argument("--out-len", type=float, default=500.0, help="projected output length per row (tokens)")
    args = ap.parse_args()
    model = StepModel(args.model)
    rows = load(args.logs)
    merged = [r for r in rows if r["ptok"] > 0 and r["nd"] > 0]
    if not merged:
        raise SystemExit("no merged steps in these logs")

    # Per-row slack is not in the trace, so the validator uses the step-level stand-in the rule reduces to when
    # every row sits exactly at its target: budget = s * a, the per-step allowance. This is the PESSIMISTIC end of
    # the rule (no banked slack), so the chunks below are a lower bound on what the rule would admit online.
    # The per-token rate the rows actually get, which `life` and `amort` price the rest of a row's output at:
    # the decode-only steps' time divided by the tokens one step emits.
    dec = [r for r in rows if r["ptok"] == 0 and r["nd"] > 0]
    rate_ms = float(np.median([r["gpu"] / max(1.0, r["acc"]) for r in dec])) if dec else args.chat_ms
    print(f"observed decode rate {rate_ms:.1f} ms/token against a {args.chat_ms:.0f} ms target "
          f"({len(dec)} decode-only steps); rule = {args.rule}, projected output {args.out_len:.0f} tokens")

    chosen, actual, pred_ms, real_ms, budgets = [], [], [], [], []
    for r in merged:
        a = max(1.0, r["acc"])
        budget = budget_ms(args.rule, args.chat_ms, args.headroom, a, args.out_len, rate_ms, args.itl_ms)
        budgets.append(budget)
        c = max_chunk(model, budget / (1.0 + args.model_error), r["pf"], r["nd"], r["dkv"], args.max_chunk)
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
    budgets = np.array(budgets, float)
    print(f"budget this rule grants: p10/p50/p90 {np.percentile(budgets, 10):.0f}/"
          f"{np.percentile(budgets, 50):.0f}/{np.percentile(budgets, 90):.0f} ms; "
          f"steps saturating the {args.max_chunk:.0f}-token maximum: {100 * np.mean(chosen >= args.max_chunk):.1f}%")
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
