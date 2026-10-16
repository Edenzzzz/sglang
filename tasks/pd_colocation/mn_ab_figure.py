"""Per-arm, per-rep spread for a multi-node routing A/B: per-request tok/s/user (median, p10-p90 bar, p1-p99
whisker) and mean TTFT / mean ITL for each rep, both streams, plus a CSV of the eval_curves rows.

  python mn_ab_figure.py <out_prefix> <label>=<arm dir prefix>[,...]   (rep suffix _rep1/_rep2 appended)

Runs locally: links each arm/rep's agent.json / chat.json on pod 0 into eval_curves.py's
<root>/<workload>_r<rate>/<ARM>_<stream>.json layout with workload "ab" and rate = rep number, runs
eval_curves.py there (CPU only), pulls the CSV and draws it.
"""
import base64, csv, io, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from px_multi import Cluster

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C = Cluster(os.environ.get("SID", "bb854a10b8954223"), "cn-seed", nodes=["worker-0"])
MN, ROOT = "/dev/shm/w/mn", "/dev/shm/w/mn_eval/ab"
SLO = '{"chat":[2000,50],"agent":[10000,80]}'


def main():
    out, spec = sys.argv[1], sys.argv[2]
    arms = [x.split("=", 1) for x in spec.split(",")]
    cmds = [f"rm -rf {ROOT}", f"mkdir -p {ROOT}/ab_r1 {ROOT}/ab_r2"]
    for label, prefix in arms:
        for rep in (1, 2):
            for st in ("agent", "chat"):
                cmds.append(f"[ -s {MN}/{prefix}_rep{rep}/{st}.json ] && ln -sf {MN}/{prefix}_rep{rep}/{st}.json "
                            f"{ROOT}/ab_r{rep}/{label}_{st}.json || true")
    src = base64.b64encode(open(os.path.join(REPO, "drivers/serving/eval_curves.py"), "rb").read()).decode()
    rc, o = C.run(0, "; ".join(cmds) + f"; echo {src} | base64 -d > /tmp/eval_curves.py; "
                     f"/dev/shm/w/venv/bin/python /tmp/eval_curves.py --root {ROOT} --slo '{SLO}' --csv /tmp/ab.csv "
                     f"> /dev/null 2>&1; echo BEGINCSV; cat /tmp/ab.csv", timeout=300)
    text = "\n".join(l for l in o.split("BEGINCSV", 1)[1].strip().splitlines() if l.strip()) + "\n"
    open(out + ".csv", "w").write(text)
    rows = list(csv.DictReader(io.StringIO(text)))
    print(f"{len(rows)} rows -> {out}.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [l for l, _ in arms]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for i, st in enumerate(("agent", "chat")):
        for j, lab in enumerate(labels):
            for rep, dx, mk in ((1, -0.12, "o"), (2, 0.12, "s")):
                r = [x for x in rows if x["arm"] == lab and x["stream"] == st and float(x["rate"]) == rep]
                if not r:
                    continue
                r = r[0]
                m = float(r["user_tok_s_median"])
                lo1, lo10 = float(r["user_tok_s_p1"]), float(r["user_tok_s_p10"])
                hi90, hi99 = float(r["user_tok_s_p90"]), float(r["user_tok_s_p99"])
                x = j + dx
                axes[i][0].errorbar([x], [m], yerr=[[m - lo1], [hi99 - m]], fmt="none", ecolor=f"C{j}", elinewidth=0.8, capsize=3)
                axes[i][0].errorbar([x], [m], yerr=[[m - lo10], [hi90 - m]], fmt=mk, color=f"C{j}", elinewidth=3, ms=5,
                                    label=f"rep {rep}" if j == 0 else None)
                axes[i][1].plot([x], [float(r["ttft_mean_ms"]) / 1000], mk, color=f"C{j}")
                axes[i][2].plot([x], [float(r["tpot_mean_ms"])], mk, color=f"C{j}")
        for k, (title, yl) in enumerate((("tok/s per user (median; bar p10-p90, whisker p1-p99)", "tok/s"),
                                          ("mean TTFT per rep", "s"), ("mean TPOT per rep", "ms"))):
            ax = axes[i][k]
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(f"{st}: {title}", fontsize=9)
            ax.set_ylabel(yl)
            ax.grid(alpha=0.3)
        axes[i][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=110)
    print(f"-> {out}.png")


if __name__ == "__main__":
    main()
