"""5e load-spike scoring: SLO attainment through a chat rate step (mn_bench.sh CHAT_SPIKE, mn_spike_8gpu_20261020.sh).

Agents (fanout.json, per-request start times from the release barrier): attainment (TTFT <= 10 s and per-request TPOT
<= 80 ms) and goodput in three windows -- before the spike, during it, after it -- plus the 60 s right after it starts
(how fast the schedule absorbs the step). Chat: the base stream over the whole run and the spike stream (which only
exists during the spike), attainment at TTFT <= 0.5 s and TPOT <= 20 ms. With a second fan-out driver (FO_SPIKE, mixture
shift), its own agents' attainment over its run.
  python spike_windows.py --spike 300:240 DIR [DIR ...]
Claude, 2026-10-20.
"""
import argparse
import json

import numpy as np

AGENT_SLO = (10.0, 80.0)
CHAT_SLO = (0.5, 20.0)


def agent_windows(path, start, dur):
    f = json.load(open(path))
    rs = [r for r in f["records"] if r.get("ttft") is not None and r.get("out_tokens", 0) > 1]
    out = []
    for name, lo, hi in (("pre", 0.0, start), ("spike", start, start + dur), ("spike+60s", start, start + 60),
                         ("post", start + dur, f["wall_s"])):
        w = [r for r in rs if lo <= r["start"] < hi]
        if not w:
            out.append(f"{name} n=0")
            continue
        good = [r for r in w if r["ttft"] <= AGENT_SLO[0]
                and (r["e2e"] - r["ttft"]) / (r["out_tokens"] - 1) * 1e3 <= AGENT_SLO[1]]
        out.append(f"{name} n={len(w)} att={len(good) / len(w) * 100:5.1f}% "
                   f"goodput={sum(r['out_tokens'] for r in good) / (hi - lo):6.1f}tok/s "
                   f"ttft={np.mean([r['ttft'] for r in w]):5.2f}s")
    return out


def chat(path):
    c = json.load(open(path))
    ok = [i for i, e in enumerate(c["errors"]) if not e and c["output_lens"][i] > 1]
    ttft = np.array([c["ttfts"][i] for i in ok])
    tpot = np.array([sum(c["itls"][i]) / (c["output_lens"][i] - 1) * 1e3 for i in ok])
    att = ((ttft <= CHAT_SLO[0]) & (tpot <= CHAT_SLO[1])).mean() * 100
    return f"n={len(ok)} att={att:5.1f}% ttft={ttft.mean():6.2f}s (p99 {np.percentile(ttft, 99):6.1f}s) tpot={tpot.mean():5.1f}ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spike", default="300:240", help="start_s:duration_s")
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    start, dur = (float(x) for x in a.spike.split(":"))
    for d in a.dirs:
        print(f"== {d.rstrip('/').split('/')[-1]}")
        print("  agent  " + " | ".join(agent_windows(f"{d}/fanout.json", start, dur)))
        print("  chat   base  " + chat(f"{d}/chat.json"))
        try:
            print("  chat   spike " + chat(f"{d}/chat_spike.json"))
        except FileNotFoundError:
            print("  chat   spike none")
        try:
            # Second fan-out driver (FO_SPIKE): its records start at its own t0, which follows its start delay.
            f = json.load(open(f"{d}/fanout_spike.json"))
            print("  agent  spike-driver " + agent_windows(f"{d}/fanout_spike.json", 0.0, f["wall_s"])[1])
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
