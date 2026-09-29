#!/usr/bin/env python3
"""Realistic accepted-length distribution for speculative decoding.

Acceptance is not i.i.d. per token: a rejection truncates the rest of the
candidate sequence, so accepted length L follows a truncated geometric law

    P(L = j) = (1-a) a^j        for 0 <= j < k
    P(L = k) = a^k              (all k speculated tokens accepted)

with L+1 tokens committed per verify step (the bonus token always lands). Mean
accepted length is a*(1-a^k)/(1-a), and mean tokens per step is 1 + that.

Per-token acceptance `a` is set from reported EAGLE-style figures on recent
models; the shape of the law matters more than the exact constant, so each is
given with its source and treated as a scenario rather than a measurement.
"""
import argparse

# label -> (per-token acceptance a, k speculated, note)
PROFILES = {
    # EAGLE-family draft heads report ~2.5-3.2x mean accepted length at k=4-7 on
    # chat/reasoning traces; a=0.75-0.8 reproduces that range.
    "eagle_strong": (0.80, 4, "EAGLE-style head, easy/repetitive traffic"),
    "eagle_typical": (0.70, 4, "EAGLE-style head, mixed traffic"),
    "eagle_weak": (0.55, 4, "EAGLE-style head, hard/diverse traffic"),
    # Self-speculative sparse draft (SparseSpec PillarAttn) runs deeper k at
    # slightly lower per-token acceptance.
    "sparse_self_k8": (0.72, 8, "sparse self-speculation, k=8"),
    # Short-depth conservative setting, the "2 spec steps" design point.
    "short_k1": (0.75, 1, "2 candidates (k=1)"),
}


def dist(a, k):
    """Return P(L=j) for j=0..k."""
    p = [(1 - a) * a ** j for j in range(k)]
    p.append(a ** k)
    return p


def mean_accepted(a, k):
    return a * (1 - a ** k) / (1 - a) if a < 1 else k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None)
    a = ap.parse_args()
    keys = [a.profile] if a.profile else list(PROFILES)
    print(f"{'profile':<16} {'a':>5} {'k':>3} {'E[acc]':>7} {'tok/step':>9} {'P(L=0)':>8} {'P(L=k)':>8}  note")
    for kk in keys:
        acc, k, note = PROFILES[kk]
        p = dist(acc, k)
        m = mean_accepted(acc, k)
        print(f"{kk:<16} {acc:>5.2f} {k:>3} {m:>7.2f} {1+m:>9.2f} {p[0]:>8.2f} {p[-1]:>8.2f}  {note}")
    print("\nVerify rows per step under the SparseSpec unified schedule are 1/(k+1) of")
    print("the running set regardless of `a`; `a` sets how fast each sequence's")
    print("context grows (and so the KV lengths the kernel sees), not the mixture.")


if __name__ == "__main__":
    main()
