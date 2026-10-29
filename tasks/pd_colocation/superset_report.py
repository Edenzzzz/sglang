"""Judge the superset sweep (mn_superset_16g_20261028.sh) against the rules agreed for it.

The claim under test (user, 2026-10-28): with SLO control and a latency model, our arm is a SUPERSET of every
baseline -- at or above each one at every load on the curve. Anywhere a baseline wins is a defect to investigate,
not a result to report. Baselines are sglang's own policies with sglang's own chunk policy: prefill-prioritized
(pp), mixed chunk (mc) and PD multiplexing (pdmux). TPOT matters more than TTFT.

Reads the driver log's `SCORE PAPER <arm> ...` and `ACCEPT <arm> accept=...` lines and prints, per load:
  - a table of every arm: chat and agent SLO attainment, then mean TPOT and mean TTFT per tenant, each with a
    p10-p99 spread as its error bar. No token rate: attainment plus the two latency means are what the SLOs are
    written against, and a rate is a third axis that moves with both of them and with the request mix, so it makes
    a table harder to read without answering a question the first two leave open (user, 2026-10-29). Where capacity
    is itself the question, the capacity figure answers it.
  - logs written before 2026-10-29 carry p50/p90 and a token rate instead. Those are still read, and their TPOT
    column is marked `*` to show it is a MEDIAN and not a mean; the two must not be compared across that line.
  - the VALIDITY gate: an arm whose mean draft acceptance is below --min-accept is marked INVALID. This catches a
    baseline running a mis-configured model (the 2026-10-28 rope-base defect gave 1.42 against a clean 1.99), which
    otherwise silently inflates our margin.
  - the SUPERSET verdict per metric against each valid baseline, and every metric where a baseline beats us.
  - the publishability bar on latency: >= 30% on TPOT or on its attainment, against the best valid baseline. The
    separate >= 10% throughput bar is NOT checked here, because this table no longer carries a rate; it is read off
    the capacity curve, which is the measurement that actually locates capacity.
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
    """Fields of one tenant's cell. Handles both the current line and pre-2026-10-29 logs."""
    d = {}
    m = re.search(r"tight=\s*([\d.]+)%", block)
    if m:
        d["attain"] = float(m.group(1))
    m = re.search(r"loose=\s*([\d.]+)%", block)
    if m:
        d["attain_loose"] = float(m.group(1))
    # Current form: a mean carried by a p10-p99 spread.
    m = re.search(r"TPOT mean=([\d.]+)ms \[p10 ([\d.]+) p99 ([\d.]+)\]", block)
    if m:
        d["tpot"], d["tpot10"], d["tpot99"] = (float(g) for g in m.groups())
    m = re.search(r"TTFT mean=([\d.]+)s \[p10 ([\d.]+) p99 ([\d.]+)\]", block)
    if m:
        d["ttft"], d["ttft10"], d["ttft99"] = (float(g) for g in m.groups())
    # Legacy form. The median goes in the same slot so old arms still sort and compare, but `median` marks it so
    # the table can say so: a mean and a median of a skewed latency distribution are different quantities, and
    # silently mixing them would manufacture a difference between arms measured on either side of the change.
    if "tpot" not in d:
        m = re.search(r"TPOT p50/p90(?:/p99)?=([\d.]+)/([\d.]+)", block)
        if m:
            d["tpot"], d["tpot99"], d["median"] = float(m.group(1)), float(m.group(2)), True
    if "ttft" not in d:
        m = re.search(r"TTFT p50/p90=([\d.]+)/([\d.]+)s", block)
        if m:
            d["ttft"], d["ttft99"], d["median"] = float(m.group(1)), float(m.group(2)), True
    return d


def comparable(ours_t, base_t, key):
    """Whether one tenant cell's metric may be compared with another's.

    Attainment is a share of requests and is defined identically either side of the 2026-10-29 metric change, so it
    always compares. A latency metric does not: a pre-change log reports the MEDIAN and a post-change log the MEAN
    of a right-skewed distribution, and the mean of such a distribution sits well above its median. Comparing them
    would read the change in reporting as a difference between the arms -- in the first mixed table it produced two
    spurious "a baseline beats ours on TTFT" findings. So refuse the comparison and say why; the fix is to re-score
    the old arm's records, which are kept, not to let the table guess.
    """
    if key == "attain":
        return True
    return bool(ours_t.get("median")) == bool(base_t.get("median"))


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

    # Higher is better for attainment; lower is better for latency. TPOT carries more weight than TTFT by the
    # user's ordering, and both tenants' attainment is checked at the tight level.
    metrics = [("chat", "attain", +1), ("agent", "attain", +1),
               ("chat", "tpot", -1), ("agent", "tpot", -1),
               ("chat", "ttft", -1), ("agent", "ttft", -1)]
    for alpha in sorted({k[0] for k in arms}):
        print(f"\n=== load factor alpha = {alpha}")
        print(f"{'arm':16}{'accept':>8}{'valid':>7}{'chat%':>8}{'agent%':>8}"
              f"{'chat TPOT ms [p10-p99]':>26}{'agent TPOT ms [p10-p99]':>27}"
              f"{'chat TTFT s':>22}{'agent TTFT s':>22}")
        valid = {}
        legacy = False
        for (al, kind), d in sorted(arms.items()):
            if al != alpha:
                continue
            acc = accept.get((al, kind))
            ok = acc is not None and acc >= a.min_accept
            valid[kind] = ok
            c, g = d["chat"], d["agent"]
            legacy |= bool(c.get("median") or g.get("median"))

            def cell(t, key, w, prec):
                v, lo, hi = t.get(key), t.get(key + "10"), t.get(key + "99")
                if v is None:
                    return f"{'-':>{w}}"
                mark = "*" if t.get("median") else " "
                span = f"[{lo:.{prec}f}-{hi:.{prec}f}]" if lo is not None else f"[?-{hi:.{prec}f}]"
                return f"{f'{v:.{prec}f}{mark} {span}':>{w}}"

            print(f"{kind:16}{(f'{acc:.3f}' if acc else '?'):>8}{('yes' if ok else 'NO'):>7}"
                  f"{c.get('attain', float('nan')):>7.1f}%{g.get('attain', float('nan')):>7.1f}%"
                  f"{cell(c, 'tpot', 26, 1)}{cell(g, 'tpot', 27, 1)}"
                  f"{cell(c, 'ttft', 22, 2)}{cell(g, 'ttft', 22, 2)}")
        if legacy:
            print("  * a MEDIAN, from a log written before the metric change, with p90 in place of p99. Do not read"
                  " it as a mean.")
        ours = arms.get((alpha, a.ours))
        if not ours:
            print(f"  no {a.ours!r} arm at this load")
            continue
        if not valid.get(a.ours):
            print("  OUR ARM FAILS THE ACCEPTANCE GATE -- nothing at this load can be used")
            continue
        losses, skipped = [], set()
        for kind in sorted(k for (al, k) in arms if al == alpha and k != a.ours):
            if not valid.get(kind):
                base = arms[(alpha, kind)]
                wins = [f"{t} {k}" for t, k, sg in metrics
                        if ours[t].get(k) is not None and base[t].get(k) is not None
                        and comparable(ours[t], base[t], k) and sg * (ours[t][k] - base[t][k]) < 0]
                print(f"  {kind}: INVALID (acceptance below {a.min_accept}); excluded."
                      + (f" It would beat ours on {', '.join(wins)} -- RE-RUN it with a fixed model rather than"
                         f" drop it, or the comparison is not credible." if wins else ""))
                continue
            base = arms[(alpha, kind)]
            for tenant, key, sign in metrics:
                o, b = ours[tenant].get(key), base[tenant].get(key)
                if o is None or b is None:
                    continue
                if not comparable(ours[tenant], base[tenant], key):
                    skipped.add(f"{kind} {tenant} {key}")
                    continue
                if sign * (o - b) < 0:
                    losses.append(f"{kind} beats ours on {tenant} {key}: {b} vs {o}")
        print("  SUPERSET HOLDS at this load" if not losses else "  SUPERSET VIOLATED -- investigate each:")
        for l in losses:
            print("    " + l)
        if skipped:
            print(f"  NOT JUDGED ({len(skipped)}): a mean against a median, which is not a comparison. Re-score "
                  f"those arms' saved records with the current scorer. " + ", ".join(sorted(skipped)))
        if skipped and not losses:
            print("  -- so the superset verdict above is PARTIAL: it holds only over the metrics that were judged.")
        valid_bases = [k for (al, k) in arms if al == alpha and k != a.ours and valid.get(k)]
        # The bar takes the BEST baseline per metric, so it has to apply the same mean-vs-median rule the superset
        # verdict does -- otherwise the single most favourable incomparable number sets the bar, which is the worst
        # case of the error rather than a harmless one. Attainment is comparable across the change; TPOT is not.
        tpot_bases = [k for k in valid_bases if comparable(ours["agent"], arms[(alpha, k)]["agent"], "tpot")]
        if valid_bases and not tpot_bases:
            print(f"  BAR not judged on TPOT: the {len(valid_bases)} valid baseline(s) here report a median, ours a"
                  f" mean. Re-score their records before claiming a TPOT margin at this load.")
        if tpot_bases:
            best_tpot = min(arms[(alpha, k)]["agent"].get("tpot", 1e9) for k in tpot_bases)
            best_att = max(arms[(alpha, k)]["agent"].get("attain", 0) for k in tpot_bases)
            o_tpot, o_att = ours["agent"].get("tpot"), ours["agent"].get("attain")
            if o_tpot and o_att is not None:
                dt = 100 * (best_tpot - o_tpot) / best_tpot
                # Attainment is a PERCENTAGE, so its improvement is stated in points as well as relatively: going
                # from 94% to 96% is +2 points but only +2.1% relative, and quoting the relative figure alone on a
                # metric that saturates near 100 would make a small absolute gain read as a large one.
                da_pts = o_att - best_att
                da_rel = 100 * da_pts / best_att if best_att else float("nan")
                print(f"  BAR vs the BEST valid baseline: agent mean TPOT {dt:+.1f}% "
                      f"(need >= 30 on this OR on attainment), agent attainment {da_pts:+.1f} points "
                      f"({da_rel:+.1f}%) -> {'MET' if max(dt, da_rel) >= 30 else 'NOT MET'}")
                print("  The >= 10% throughput bar is not judged here: read it off the capacity curve.")


if __name__ == "__main__":
    main()
