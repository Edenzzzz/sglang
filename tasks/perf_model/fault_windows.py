"""Fault-tolerance scoring (mn_run.py FAULT_AT / FAULT_POD) under the paper's SLOs, in time windows around the kill.

Per arm directory: fanout.json (agents; record start in seconds from the driver's release), chat.json (bench_serving,
request_start_epoch when the client records it), fault_t.txt (epoch of the kill, written on the killed pod), and for
arms behind our router router_decisions.jsonl. The release epoch is the earliest of: the first router decision, the
first chat request start; one of them must exist.

Reports, per tenant: attainment (agent: TTFT 10 s, 20 s above 200K uncached tokens, mean TPOT 80 ms; chat: TTFT 2 s,
mean TPOT 50 ms) before the kill, in the first 60 s after it and for the rest of the run; requests lost (errors, which
count as misses); and the recovery time, the first 30 s window after the kill whose attainment is back within 5 points
of the pre-kill level. With --series, the 30 s windows themselves (fault_windows.json in the arm directory).
  python fault_windows.py [--series] DIR [DIR ...]
Claude, 2026-10-26.
"""
import argparse
import json
from pathlib import Path

import numpy as np

WIDTH = 30.0


def agent_rows(d):
    f = json.loads((d / "fanout.json").read_text())
    rows = []
    for r in f["records"]:
        ok = r.get("ttft") is not None and r.get("out_tokens", 0) > 1 and "error" not in r
        meets = False
        if ok:
            uncached = int(r.get("input_len") or 0) - int(r.get("cached") or 0)
            tpot = (r["e2e"] - r["ttft"]) / (r["out_tokens"] - 1) * 1e3
            meets = r["ttft"] <= (20.0 if uncached > 200_000 else 10.0) and tpot <= 80.0
        rows.append(dict(t=r["start"], meets=meets, lost="error" in r, out=r.get("out_tokens", 0)))
    return rows, f["wall_s"]


def chat_rows(d, release):
    c = json.loads((d / "chat.json").read_text().splitlines()[-1])
    starts = c.get("request_start_epoch")
    if starts is None:
        return None
    rows = []
    for i, e in enumerate(c["errors"]):
        n = c["output_lens"][i]
        ok = not e and n > 1
        meets = ok and c["ttfts"][i] <= 2.0 and sum(c["itls"][i]) / (n - 1) * 1e3 <= 50.0
        rows.append(dict(t=starts[i] - release, meets=meets, lost=bool(e), out=n))
    return rows


def release_epoch(d):
    candidates = []
    log = d / "router_decisions.jsonl"
    if log.exists():
        for line in log.open():
            row = json.loads(line)
            if "t" in row:
                candidates.append(row["t"])
                break
    chat = d / "chat.json"
    if chat.exists():
        starts = json.loads(chat.read_text().splitlines()[-1]).get("request_start_epoch")
        if starts:
            candidates.append(min(starts))
    if not candidates:
        raise SystemExit(f"{d}: no release time (no router log, chat client without request_start_epoch)")
    return min(candidates)


def summarize(rows, fault, end):
    def att(lo, hi):
        w = [r for r in rows if lo <= r["t"] < hi]
        return (sum(r["meets"] for r in w) / len(w) * 100 if w else None), len(w), sum(r["lost"] for r in w)
    pre, after60, rest = att(0, fault), att(fault, fault + 60), att(fault + 60, end)
    recovery = None
    if pre[0] is not None:
        for lo in np.arange(fault, end, WIDTH):
            a, n, _ = att(lo, lo + WIDTH)
            if n and a >= pre[0] - 5.0:
                recovery = lo + WIDTH - fault
                break
    fmt = lambda x: "-" if x[0] is None else f"{x[0]:5.1f}% (n={x[1]}, lost={x[2]})"
    return (f"before {fmt(pre)} | first 60 s {fmt(after60)} | after {fmt(rest)} | recovered "
            f"{'never' if recovery is None else f'{recovery:.0f} s'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", action="store_true")
    parser.add_argument("dirs", nargs="+")
    args = parser.parse_args()
    for d in map(Path, args.dirs):
        release = release_epoch(d)
        fault = float((d / "fault_t.txt").read_text().split()[0]) - release
        agents, wall = agent_rows(d)
        chat = chat_rows(d, release)
        print(f"== {d.name} kill at {fault:.0f} s after release")
        print(f"  agent {summarize(agents, fault, wall)}")
        print(f"  chat  {'no request start times' if chat is None else summarize(chat, fault, wall)}")
        if args.series:
            series = []
            for lo in np.arange(0, wall, WIDTH):
                entry = dict(t=float(lo))
                for name, rows in (("agent", agents), ("chat", chat or [])):
                    w = [r for r in rows if lo <= r["t"] < lo + WIDTH]
                    entry[f"{name}_n"] = len(w)
                    entry[f"{name}_attain"] = sum(r["meets"] for r in w) / len(w) * 100 if w else None
                    entry[f"{name}_lost"] = sum(r["lost"] for r in w)
                    entry[f"{name}_tok_s"] = sum(r["out"] for r in w if not r["lost"]) / WIDTH
                series.append(entry)
            (d / "fault_windows.json").write_text(json.dumps(series, indent=1) + "\n")


if __name__ == "__main__":
    main()
