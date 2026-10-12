#!/usr/bin/env python3
"""S1/S2 evaluation curves from bench_serving --output-details JSONs.

Layout: <root>/<workload>_r<rate>/<ARM>_<stream>.json (one run per rate point). For every
(workload, rate, arm, stream) it reports output throughput, per-user interactivity (tok/s/user,
the median over requests of output tokens / decode time, decode time being the sum of the
request's ITLs), TTFT/ITL statistics, acceptance length and SLO attainment against PER-TENANT
SLOs. The two required figures are drawn from these rows:
  * throughput vs tok/s/user Pareto curve (one point per rate, per arm, per workload);
  * SLO attainment vs request rate.

A request attains its SLO when TTFT <= ttft_ms and its mean per-token ITL (TPOT) <= tpot_ms.
SLOs are given per stream, e.g. --slo '{"chat": [2000, 50], "codex": [10000, 80]}'
(ms; streams missing from the map use --default-slo). --slo-relative K_TTFT,K_TPOT instead scales
each (workload, stream)'s SLO from its unloaded latency (SLOs-Serve style): K times the median TTFT
and TPOT of the --ref-arm running that stream ALONE at a low rate, read from
<ref-root>/<workload>_<anything>/<ref-arm>_<stream>.json; the same SLO for every arm. (The swept
runs' lowest rate is not a reference: there the chat tenant already queues behind agent prefills,
e.g. mix128 chat median TTFT 38 s at 4/s.)

Usage: eval_curves.py --root DIR (--slo JSON [--default-slo 5000,100] |
       --slo-relative 5,2 --ref-root UNLOADED_DIR [--ref-arm MTP_LPT]) [--csv out.csv] [--plot out.png]
"""
import argparse, csv, glob, json, os, re, sys

import numpy as np


def load(path):
    return json.loads(open(path).read().strip().splitlines()[-1])


def unloaded(path):
    """Median TTFT and TPOT (ms) of one run, the reference for relative SLOs."""
    j = load(path)
    out = np.array(j["output_lens"], dtype=float)
    decode = np.array([float(np.sum(x)) for x in j["itls"]])
    tpot = decode[out > 1] * 1e3 / (out[out > 1] - 1)
    return float(np.median(np.array(j["ttfts"], dtype=float) * 1e3)), float(np.median(tpot))


def rows_for(path, workload, rate, arm, stream, slo):
    j = load(path)
    ttft = np.array(j["ttfts"], dtype=float) * 1e3
    out = np.array(j["output_lens"], dtype=float)
    decode = np.array([float(np.sum(x)) for x in j["itls"]])          # seconds
    tpot = np.where(out > 1, decode * 1e3 / np.maximum(out - 1, 1), 0.0)  # ms per token
    user_tps = np.where(decode > 0, (out - 1) / np.maximum(decode, 1e-9), np.nan)
    tok_itl = np.concatenate([np.array(x) * 1e3 for x in j["itls"] if len(x)]) if any(
        len(x) for x in j["itls"]) else np.array([0.0])
    ttft_slo, tpot_slo = slo
    ok = (ttft <= ttft_slo) & (tpot <= tpot_slo)
    proposed = sum(j.get("spec_num_proposed_drafts") or []) or 0
    correct = sum(j.get("spec_num_correct_drafts") or []) or 0
    verify = sum(j.get("spec_verify_ct") or []) or 0
    return {
        "workload": workload, "rate": rate, "arm": arm, "stream": stream, "n": len(ttft),
        "duration_s": round(j["duration"], 2),
        "out_tok_s": round(float(out.sum()) / j["duration"], 1),
        "user_tok_s_median": round(float(np.nanmedian(user_tps)), 2),
        "ttft_mean_ms": round(float(ttft.mean()), 1),
        "ttft_p99_ms": round(float(np.percentile(ttft, 99)), 1),
        "tpot_mean_ms": round(float(tpot.mean()), 2),
        "itl_p99_ms": round(float(np.percentile(tok_itl, 99)), 1),
        "tokens_per_verify": round((correct + verify) / verify, 3) if verify else "",
        "slo_ttft_ms": ttft_slo, "slo_tpot_ms": tpot_slo,
        "slo_attain": round(float(ok.mean()), 4),
        "goodput_req_s": round(float(ok.sum()) / j["duration"], 3),
        "accept_rate": round(correct / proposed, 4) if proposed else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--slo", default="{}", help='per-stream {"stream": [ttft_ms, tpot_ms]}')
    ap.add_argument("--default-slo", default="5000,100")
    ap.add_argument("--slo-relative", help="K_TTFT,K_TPOT x the reference arm's unloaded medians")
    ap.add_argument("--ref-root", help="unloaded single-stream runs for --slo-relative")
    ap.add_argument("--ref-arm", default="MTP_LPT")
    ap.add_argument("--csv")
    ap.add_argument("--plot")
    a = ap.parse_args()
    per_stream = {k: tuple(map(float, v)) for k, v in json.loads(a.slo).items()}
    default = tuple(map(float, a.default_slo.split(",")))
    runs = []
    for d in sorted(glob.glob(os.path.join(a.root, "*_r*"))):
        m = re.match(r"(.+)_r([0-9.]+)$", os.path.basename(d))
        if not m:
            continue
        workload, rate = m.group(1), float(m.group(2))
        for f in sorted(glob.glob(os.path.join(d, "*_*.json"))):
            base = os.path.basename(f)[:-5]
            if base.endswith("_validation") or ".previous." in base:
                continue
            arm, stream = base.rsplit("_", 1)
            runs.append((f, workload, rate, arm, stream))
    slo_of = lambda workload, stream: per_stream.get(stream, default)
    if a.slo_relative:
        k_ttft, k_tpot = map(float, a.slo_relative.split(","))
        if not a.ref_root:
            sys.exit("--slo-relative needs --ref-root (unloaded single-stream runs)")
        ref = {}
        for f in glob.glob(os.path.join(a.ref_root, "*_*", f"{a.ref_arm}_*.json")):
            base = os.path.basename(f)[:-5]
            if base.endswith("_validation") or ".previous." in base:
                continue
            workload = os.path.basename(os.path.dirname(f)).rsplit("_", 1)[0]
            ref[(workload, base[len(a.ref_arm) + 1:])] = ("unloaded", f)
        rel = {}
        for key, (rate, f) in ref.items():
            ttft_ms, tpot_ms = unloaded(f)
            rel[key] = (round(k_ttft * ttft_ms, 1), round(k_tpot * tpot_ms, 2))
            print(f"relative SLO {key[0]}/{key[1]}: {a.ref_arm} {rate} median TTFT "
                  f"{ttft_ms:.0f} ms, TPOT {tpot_ms:.1f} ms -> {rel[key]}", file=sys.stderr)
        missing = {(w, st) for _, w, _, _, st in runs} - set(rel)
        if missing:
            sys.exit(f"no unloaded reference for {sorted(missing)} under {a.ref_root}")
        slo_of = lambda workload, stream: rel[(workload, stream)]
    rows = [rows_for(f, workload, rate, arm, stream, slo_of(workload, stream))
            for f, workload, rate, arm, stream in runs]
    if not rows:
        sys.exit(f"no <workload>_r<rate>/<ARM>_<stream>.json under {a.root}")
    cols = list(rows[0])
    w = csv.DictWriter(open(a.csv, "w", newline="") if a.csv else sys.stdout, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = sorted({(r["workload"], r["stream"]) for r in rows})
        fig, axes = plt.subplots(len(keys), 2, figsize=(11, 3.6 * len(keys)), squeeze=False)
        for i, (wl, st) in enumerate(keys):
            for arm in sorted({r["arm"] for r in rows}):
                pts = sorted((r for r in rows if r["workload"] == wl and r["stream"] == st
                              and r["arm"] == arm), key=lambda r: r["rate"])
                if not pts:
                    continue
                axes[i][0].plot([p["user_tok_s_median"] for p in pts], [p["out_tok_s"] for p in pts],
                                "o-", label=arm)
                axes[i][1].plot([p["rate"] for p in pts], [p["slo_attain"] for p in pts], "o-",
                                label=arm)
            axes[i][0].set(title=f"{wl} / {st}: throughput vs interactivity",
                           xlabel="tok/s/user (median)", ylabel="output tok/s")
            axes[i][1].set(title=f"{wl} / {st}: SLO attainment", xlabel="request rate (req/s)",
                           ylabel="fraction meeting SLO", ylim=(0, 1.02))
            axes[i][0].legend(fontsize=7)
            axes[i][1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=120)


if __name__ == "__main__":
    main()
