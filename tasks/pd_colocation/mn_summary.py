"""One line per stream of a multi-node arm, plus worker-log health counts.  mn_summary.py <logdir>"""
import glob, json, os, sys
d = sys.argv[1]
K = ("completed", "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "mean_itl_ms", "p99_itl_ms", "output_throughput")
for s in ("agent", "chat"):
    p = os.path.join(d, f"{s}.json")
    if not os.path.exists(p):
        print(s, "MISSING"); continue
    # bench_serving APPENDS one JSON line per run; the last line is this run.
    j = json.loads([l for l in open(p) if l.strip()][-1])
    print(s, " ".join(f"{k}={float(j.get(k) or 0):.1f}" for k in K))
tb = sum(open(f, errors="replace").read().count("Traceback") for f in glob.glob(os.path.join(d, "worker_*.log")))
fb = sum(open(f, errors="replace").read().lower().count("fallback") for f in glob.glob(os.path.join(d, "worker_*.log")))
print(f"workers={len(glob.glob(os.path.join(d, 'worker_*.log')))} tracebacks={tb} fallbacks={fb}")
