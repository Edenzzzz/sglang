import argparse, glob, json, os, statistics as st, sys
ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/dev/shm/w/rep"); ap.add_argument("--base", default="LPT")
A = ap.parse_args()
K = ("mean_ttft_ms", "p99_ttft_ms", "mean_itl_ms", "p99_itl_ms", "output_throughput", "completed")
rows = {}
for f in sorted(glob.glob(A.root + "/r*_*/*.json")):
    if os.path.basename(f).startswith(("composition",)):
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    rd, arm = os.path.basename(os.path.dirname(f)).split("_", 1)
    stream = os.path.basename(f)[:-5].split("_")[-1]
    rows.setdefault((stream, arm), []).append((rd, {k: d.get(k) for k in K}))
for (stream, arm), v in sorted(rows.items()):
    for rd, m in v:
        print(stream, arm, rd, " ".join(f"{k}={m[k]:.1f}" if isinstance(m[k], (int, float)) else f"{k}=NA" for k in K))
print(f"--- mean over rounds, each arm vs {A.base}")
for stream in sorted({s for s, _ in rows}):
    a = rows.get((stream, A.base), [])
    for arm in sorted({x for _, x in rows if x != A.base}):
        b = rows.get((stream, arm), [])
        if not a or not b:
            continue
        for k in K[:5]:
            la = [m[k] for _, m in a if m[k] is not None]; lb = [m[k] for _, m in b if m[k] is not None]
            if la and lb:
                ma, mb = st.mean(la), st.mean(lb)
                print(f"{stream:6} {arm:13} {k:18} {A.base} {ma:9.1f} [{min(la):.1f}-{max(la):.1f}]  arm {mb:9.1f} [{min(lb):.1f}-{max(lb):.1f}]  {(mb/ma-1)*100:+6.1f}%  n={len(la)}/{len(lb)}")
