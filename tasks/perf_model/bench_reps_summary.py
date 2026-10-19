"""Six-repeat summary of the single-node standard-mix arms (H200 dadf94f0e1da7238, 2026-10-19/20): per arm, the mean and
std across repeats of throughput / TTFT / per-request TPOT and of per-tenant SLO attainment (TTFT and TPOT targets both
met), plus every repeat's value, so stock's two modes stay visible.
  python bench_reps_summary.py <mn dir>
Claude, 2026-10-20.
"""
import glob
import json
import sys

import numpy as np

ARMS = {
    "stock (mixed chunk)": "coloc_sgl_h200_c8_pt_rep{}",
    "stock prefill-prioritized": "coloc_sgl_h200_pp_c8_pt_rep{}",
    "ours, no MTP": "coloc_cache_edf_h200_split_ovl_ch2048_dl1024-2048-6000_c8_pt_rep{}",
    "stock + MTP": "coloc_sgl_mtps_h200_c8_pt_rep{}",
    "stock prefill-prioritized + MTP": "coloc_sgl_mtps_h200_pp_c8_pt_rep{}",
    "ours + MTP": "coloc_cache_edf_mtp_h200_4mode_unc_ch2048_dl1024-2048-6000_c8_pt_rep{}",
}
SLOS = {"agent": (10.0, 80.0), "chat": (0.5, 20.0)}


def tenant(path, slo):
    c = json.load(open(path))
    ok = [i for i, e in enumerate(c["errors"]) if not e and c["output_lens"][i] > 1]
    ttft = np.array([c["ttfts"][i] for i in ok])
    tpot = np.array([sum(c["itls"][i]) / (c["output_lens"][i] - 1) * 1e3 for i in ok])
    return dict(tput=c["output_throughput"], ttft=ttft.mean(), tpot=tpot.mean(),
                att=((ttft <= slo[0]) & (tpot <= slo[1])).mean() * 100)


root = sys.argv[1]
for arm, pat in ARMS.items():
    reps = [(n, f"{root}/{pat.format(n)}") for n in range(1, 7) if glob.glob(f"{root}/{pat.format(n)}/chat.json")]
    if not reps:
        continue
    rows = {t: [tenant(f"{d}/{t}.json", SLOS[t]) for _, d in reps] for t in SLOS}
    print(f"== {arm}: reps {[n for n, _ in reps]}")
    for t in SLOS:
        cells = []
        for k, fmt in (("tput", "{:.1f}"), ("ttft", "{:.2f}s"), ("tpot", "{:.1f}ms"), ("att", "{:.1f}%")):
            v = np.array([r[k] for r in rows[t]])
            cells.append(f"{k} {fmt.format(v.mean())} +- {fmt.format(v.std())} [" + " ".join(fmt.format(x) for x in v) + "]")
        print(f"  {t:5s} " + " | ".join(cells))
