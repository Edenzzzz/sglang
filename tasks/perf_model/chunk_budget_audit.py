"""Audit the SLO-budgeted chunk rule against recorded steps (apply_slo_budget_chunk_20261026.py).

The rule sizes each step's prefill chunk so that the predicted step time stays within every running row's remaining
mean-TPOT slack. This checks its premise on STEPTRACE logs of a real run: how long a decode-only step actually takes
(the floor the rule cannot remove), how much room is left under a tenant's per-step allowance (SLO x tokens per step),
and what a measured chunk actually costs. A chunk is worth admitting only if the node is not already past the
allowance without it.
  python chunk_budget_audit.py <steptrace log> [...] [--chat-ms 50] [--agent-ms 80]
Claude, 2026-10-27.
"""
import argparse
import collections
import re

import numpy as np

PAT = re.compile(r"STEPTRACE (\w+) et=(\d+) bs=(\d+) np=(\d+) ptok=(\d+) pf=(\d+) nd=(\d+) dkv=(\d+) kv=(\d+) "
                 r"mx=(\d+)(?: sg=\d+ lay=\d+)?(?: acc=([0-9.]+))?(?: dacc=[0-9.]+)? t=([0-9.]+) gpu=([0-9.]+)")


def load(paths):
    rows = []
    for path in paths:
        for line in open(path):
            m = PAT.search(line)
            if m:
                g = m.groups()
                rows.append(dict(np=int(g[3]), ptok=int(g[4]), pf=int(g[5]), nd=int(g[6]), dkv=int(g[7]),
                                 acc=float(g[10] or 1.0), gpu=float(g[12])))
    if not rows:
        raise SystemExit("no STEPTRACE lines found")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--chat-ms", type=float, default=50.0)
    parser.add_argument("--agent-ms", type=float, default=80.0)
    args = parser.parse_args()
    rows = load(args.logs)
    decode = [r for r in rows if r["np"] == 0 and r["nd"] > 0]
    merged = [r for r in rows if r["np"] > 0 and r["nd"] > 0]
    acc = float(np.median([r["acc"] for r in rows if r["acc"] > 0]) or 1.0)
    print(f"{len(rows)} steps: {len(decode)} decode-only, {len(merged)} with a prefill chunk beside rows; "
          f"median acceptance {acc:.2f}")
    print(f"per-step allowance: chat {args.chat_ms * acc:.0f} ms, agent {args.agent_ms * acc:.0f} ms")
    if decode:
        d = np.array([r["gpu"] for r in decode])
        print(f"decode-only step ms p50/p90/p99 {np.percentile(d, 50):.1f}/{np.percentile(d, 90):.1f}/"
              f"{np.percentile(d, 99):.1f}; above the chat allowance: {100 * np.mean(d > args.chat_ms * acc):.1f}%")
        by = collections.defaultdict(list)
        for r in decode:
            by[r["nd"]].append(r["gpu"])
        print("  rows -> median decode-only ms, room left for a chunk under the chat allowance:")
        for k in sorted(by):
            if len(by[k]) >= 50:
                med = float(np.median(by[k]))
                print(f"    {k:>4} rows: {med:7.1f} ms, room {args.chat_ms * acc - med:8.1f} ms")
    if merged:
        m = np.array([r["gpu"] for r in merged])
        p = np.array([r["ptok"] for r in merged])
        print(f"merged step ms p50/p90 {np.percentile(m, 50):.1f}/{np.percentile(m, 90):.1f}; chunk tokens p50/p90 "
              f"{np.percentile(p, 50):.0f}/{np.percentile(p, 90):.0f}; above the chat allowance: "
              f"{100 * np.mean(m > args.chat_ms * acc):.1f}%")
        print(f"  a merged step costs {np.median(m) / (args.chat_ms * acc):.1f} chat allowances, so each resident row "
              "waits that many of its own SLO budgets for one of these steps")


if __name__ == "__main__":
    main()
