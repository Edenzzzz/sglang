"""Chat tail and attainment for run_native_serving_pair arms: TPOT percentiles, worst requests, requests with a
>1 s single gap (part 1); attainment at 2 s / 50 ms, 2 s / 100 ms, 1 s / 100 ms and end-to-end latency (part 2).
  python chat_tail_attainment_20261026.py <arm dir> [...]   Claude, 2026-10-26."""
import json, sys, statistics
for a in sys.argv[1:]:
    d = json.load(open(f"{a}/requests/chat.json"))
    rows = []
    for i, e in enumerate(d["errors"]):
        n = d["output_lens"][i]
        if e or n < 2: continue
        it = d["itls"][i]
        rows.append((sum(it) / (n - 1) * 1e3, n, max(it) * 1e3, d["ttfts"][i]))
    t = sorted(r[0] for r in rows)
    q = lambda p: t[min(len(t) - 1, int(p * len(t)))]
    print(a.split("/")[-1], f"TPOT p50={q(.5):.1f} p75={q(.75):.1f} p90={q(.9):.1f} p95={q(.95):.1f} p99={q(.99):.1f} max={t[-1]:.0f}")
    worst = sorted(rows, reverse=True)[:5]
    print("   worst (tpot ms, out tokens, max single gap ms, ttft s):", [(round(w[0]), w[1], round(w[2]), round(w[3], 2)) for w in worst])
    big = [r for r in rows if r[2] > 1000]
    print(f"   requests with a single gap > 1 s: {len(big)}; their share of summed TPOT: {sum(r[0] for r in big) / sum(r[0] for r in rows) * 100:.0f}%")
import json, sys, statistics
for a in sys.argv[1:]:
    d = json.load(open(f"{a}/requests/chat.json"))
    ok = [i for i, e in enumerate(d["errors"]) if not e and d["output_lens"][i] > 1]
    tt = [d["ttfts"][i] for i in ok]; tp = [sum(d["itls"][i]) / (d["output_lens"][i] - 1) * 1e3 for i in ok]
    e2e = [d["ttfts"][i] + sum(d["itls"][i]) for i in ok]
    att = lambda T, M: sum(1 for x, y in zip(tt, tp) if x <= T and y <= M) / len(ok) * 100
    s = sorted(e2e)
    print(a.split("/")[-1], f"chat att 2s/50ms={att(2,50):.1f}% 2s/100ms={att(2,100):.1f}% 1s/100ms={att(1,100):.1f}% | e2e mean={statistics.mean(e2e):.2f}s p50={s[len(s)//2]:.2f} p90={s[int(.9*len(s))]:.2f}")
