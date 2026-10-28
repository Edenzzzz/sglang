"""Judge the superset sweep (mn_superset_16g_20261028.sh) against the rules agreed for it.

The claim under test (user, 2026-10-28): with SLO control and a latency model, our arm is a SUPERSET of every
baseline -- at or above each one at every load on the curve. Anywhere a baseline wins is a defect to investigate,
not a result to report. Baselines are sglang's own policies with sglang's own chunk policy: prefill-prioritized
(pp), mixed chunk (mc) and PD multiplexing (pdmux). TPOT matters more than TTFT.

Reads the driver log's `SCORE PAPER <arm> ...` and `ACCEPT <arm> accept=...` lines and prints, per load:
  - a table of every arm: chat / agent attainment (tight), agent SLO goodput, chat and agent TPOT p50 / p90
  - the VALIDITY gate: an arm whose mean draft acceptance is below --min-accept is marked INVALID. This catches a
    baseline running a mis-configured model (the 2026-10-28 rope-base defect gave 1.42 against a clean 1.99), which
    otherwise silently inflates our margin.
  - the SUPERSET verdict per metric against each valid baseline, and every metric where a baseline beats us.
  - the publishability bar: >= 30% on TPOT (or its attainment) and >= 10% on throughput, against the best valid
    baseline on each metric.
  python superset_report.py /tmp/h20/mn_superset_16g_r4.log [--min-accept 1.8]
Claude, 2026-10-28.
"""
import argparse
import re

# The middle group is the sweep tag (superset / qcap / decomp / overload / life / ...); it is kept so
# that arms from different drivers can be read into one table, with the tag appended to the arm name
# when it is not the plain "superset" sweep.
ARM = re.compile(r"_tl2_a(\d+)_([a-z_]+)_r\d+_(\w+)")


def parse_tenant(block):
    d = {}
    m = re.search(r"tight=\s*([\d.]+)%\s*\((\d+) tok/s\)", block)
    if m:
        d["attain"], d["goodput"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"TPOT p50/p90=([\d.]+)/([\d.]+)ms", block)
    if m:
        d["tpot50"], d["tpot90"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"TTFT p50/p90=([\d.]+)/([\d.]+)s", block)
    if m:
        d["ttft50"], d["ttft90"] = float(m.group(1)), float(m.group(2))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="sweep log(s); a qcap log adds an ours_qcap row")
    ap.add_argument("--min-accept", type=float, default=1.8)
    # Which arm is "ours". Defaults to the arm literally named ours; the decomposition drivers name their rungs
    # L0 / L1 instead, so pass e.g. --ours L1_overload to judge that rung against the baselines.
    ap.add_argument("--ours", default="ours")
    a = ap.parse_args()
    arms, accept = {}, {}
    for line in (l for f in a.logs for l in open(f)):
        if line.startswith("SCORE PAPER "):
            name = line.split()[2]
            m = ARM.search(name)
            if not m:
                continue
            alpha, kind = int(m.group(1)), m.group(3) + ("" if m.group(2) == "superset" else "_" + m.group(2))
            chat = line.split("chat:", 1)[1].split("|", 1)[0] if "chat:" in line else ""
            agent = line.split("agent:", 1)[1] if "agent:" in line else ""
            arms[(alpha, kind)] = dict(chat=parse_tenant(chat), agent=parse_tenant(agent))
        elif line.startswith("ACCEPT "):
            parts = line.split()
            m = ARM.search(parts[1])
            v = re.search(r"accept=([\d.]+)", line)
            if m and v:
                accept[(int(m.group(1)), m.group(3) + ("" if m.group(2) == "superset" else "_" + m.group(2)))] = float(v.group(1))
    if not arms:
        raise SystemExit("no SCORE PAPER lines for superset arms in this log")

    # Higher is better for attainment / goodput; lower is better for TPOT.
    metrics = [("chat", "attain", +1), ("agent", "attain", +1), ("agent", "goodput", +1),
               ("chat", "tpot90", -1), ("agent", "tpot90", -1)]
    for alpha in sorted({k[0] for k in arms}):
        print(f"\n=== load factor alpha = {alpha}")
        print(f"{'arm':16}{'accept':>8}{'valid':>7}{'chat%':>8}{'agent%':>8}{'agent tok/s':>12}"
              f"{'chat TPOT p50/p90':>19}{'agent TPOT p50/p90':>20}")
        valid = {}
        for (al, kind), d in sorted(arms.items()):
            if al != alpha:
                continue
            acc = accept.get((al, kind))
            ok = acc is not None and acc >= a.min_accept
            valid[kind] = ok
            c, g = d["chat"], d["agent"]
            print(f"{kind:16}{(f'{acc:.3f}' if acc else '?'):>8}{('yes' if ok else 'NO'):>7}"
                  f"{c.get('attain', float('nan')):>7.1f}%{g.get('attain', float('nan')):>7.1f}%"
                  f"{g.get('goodput', float('nan')):>12.0f}"
                  f"{c.get('tpot50', float('nan')):>10.1f}/{c.get('tpot90', float('nan')):<8.1f}"
                  f"{g.get('tpot50', float('nan')):>11.1f}/{g.get('tpot90', float('nan')):<8.1f}")
        ours = arms.get((alpha, a.ours))
        if not ours:
            print(f"  no {a.ours!r} arm at this load")
            continue
        if not valid.get(a.ours):
            print("  OUR ARM FAILS THE ACCEPTANCE GATE -- nothing at this load can be used")
            continue
        losses = []
        for kind in sorted(k for (al, k) in arms if al == alpha and k != a.ours):
            if not valid.get(kind):
                base = arms[(alpha, kind)]
                wins = [f"{t} {k}" for t, k, sg in metrics
                        if ours[t].get(k) is not None and base[t].get(k) is not None
                        and sg * (ours[t][k] - base[t][k]) < 0]
                print(f"  {kind}: INVALID (acceptance below {a.min_accept}); excluded."
                      + (f" It would beat ours on {', '.join(wins)} -- RE-RUN it with a fixed model rather than"
                         f" drop it, or the comparison is not credible." if wins else ""))
                continue
            base = arms[(alpha, kind)]
            for tenant, key, sign in metrics:
                o, b = ours[tenant].get(key), base[tenant].get(key)
                if o is None or b is None:
                    continue
                if sign * (o - b) < 0:
                    losses.append(f"{kind} beats ours on {tenant} {key}: {b} vs {o}")
        print("  SUPERSET HOLDS at this load" if not losses else "  SUPERSET VIOLATED -- investigate each:")
        for l in losses:
            print("    " + l)
        valid_bases = [k for (al, k) in arms if al == alpha and k != a.ours and valid.get(k)]
        if valid_bases:
            best_tpot = min(arms[(alpha, k)]["agent"].get("tpot90", 1e9) for k in valid_bases)
            best_gp = max(arms[(alpha, k)]["agent"].get("goodput", 0) for k in valid_bases)
            o_tpot, o_gp = ours["agent"].get("tpot90"), ours["agent"].get("goodput")
            if o_tpot and o_gp:
                dt = 100 * (best_tpot - o_tpot) / best_tpot
                dg = 100 * (o_gp - best_gp) / best_gp if best_gp else float("nan")
                print(f"  BAR vs the BEST valid baseline: agent TPOT p90 {dt:+.1f}% (need >= 30), "
                      f"agent goodput {dg:+.1f}% (need >= 10) -> {'MET' if dt >= 30 and dg >= 10 else 'NOT MET'}")


if __name__ == "__main__":
    main()
