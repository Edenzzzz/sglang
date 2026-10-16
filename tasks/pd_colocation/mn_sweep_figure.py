"""Throughput vs tok/s/user (p1/p10/p90/p99 error bars), SLO attainment, TTFT and TPOT vs chat rate for
the 8-GPU chat request-rate sweep (mn_chat_ratesweep_8gpu_20261016.sh), plus the matrix's chat 4 req/s
point at the same agent concurrency (C=16).

  python mn_sweep_figure.py <out_prefix>      -> <out_prefix>.csv, <out_prefix>.png

Runs locally: links each arm's agent.json / chat.json on pod 0 into eval_curves.py's
<root>/<workload>_r<rate>/<ARM>_<stream>.json layout, runs eval_curves.py there (CPU only), pulls the CSV
and draws it with drivers/serving/plot_eval.py.
"""
import base64, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from px_multi import Cluster

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
C = Cluster(os.environ.get("SID", "bb854a10b8954223"), "cn-seed", nodes=["worker-0"])
MN = "/dev/shm/w/mn"
ROOT = "/dev/shm/w/mn_eval/chatsweep_c16"
ARMS = {"coloc_sgl_mtps": "stock_coloc", "coloc_cache_edf_mtp": "ours", "disagg_2p2d_mtps": "stock_disagg"}
# (chat req/s, run tag); the disagg r4 point is the clean-upstream rerun.
POINTS = [(4, "_8g_c16_r4"), (8, "_8g_chatsweep_c16_r8"), (16, "_8g_chatsweep_c16_r16"),
          (32, "_8g_chatsweep_c16_r32"), (64, "_8g_chatsweep_c16_r64")]
SLO = '{"chat":[2000,50],"agent":[10000,80]}'


def main():
    out = sys.argv[1]
    links = []
    for rate, tag in POINTS:
        for arm, name in ARMS.items():
            t = tag + "_clean" if (arm.startswith("disagg") and rate == 4) else tag
            for st in ("agent", "chat"):
                links.append(f"[ -s {MN}/{arm}{t}/{st}.json ] && ln -sf {MN}/{arm}{t}/{st}.json "
                             f"{ROOT}/h20x8_r{rate}/{name}_{st}.json")
    mk = " ; ".join(f"mkdir -p {ROOT}/h20x8_r{r}" for r, _ in POINTS)
    src = base64.b64encode(open(os.path.join(REPO, "drivers/serving/eval_curves.py"), "rb").read()).decode()
    rc, o = C.run(0, f"rm -rf {ROOT}; {mk}; {' ; '.join(links)}; echo {src} | base64 -d > /tmp/eval_curves.py; "
                     f"/dev/shm/w/venv/bin/python /tmp/eval_curves.py --root {ROOT} --slo '{SLO}' --csv /tmp/chatsweep.csv "
                     f"> /dev/null 2>&1; echo BEGINCSV; cat /tmp/chatsweep.csv", timeout=300)
    csv = o.split("BEGINCSV", 1)[1].strip()
    open(out + ".csv", "w").write("\n".join(l for l in csv.splitlines() if l.strip()) + "\n")
    print(f"{len(csv.splitlines()) - 1} rows -> {out}.csv")
    subprocess.run([sys.executable, os.path.join(REPO, "drivers/serving/plot_eval.py"), out + ".csv", out + ".png"],
                   check=True)


if __name__ == "__main__":
    main()
