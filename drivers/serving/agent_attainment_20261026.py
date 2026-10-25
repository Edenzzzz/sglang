"""Agent SLO attainment / SLO goodput of run_native_serving_pair arms under the paper SLOs: TTFT 10 s (20 s for
prompts > 200K tokens; total prompt length, since the bench records no cached length) and mean TPOT 80 / 100 ms.
  python agent_attainment_20261026.py <arm dir> [...]   Claude, 2026-10-26."""
import json, sys
for a in sys.argv[1:]:
    d = json.load(open(f"{a}/requests/agent.json"))
    ok = [i for i, e in enumerate(d["errors"]) if not e and d["output_lens"][i] > 1]
    def att(m, scale=1.0):
        g = [i for i in ok if d["ttfts"][i] <= (20.0 if d["input_lens"][i] > 200_000 else 10.0) * scale
             and sum(d["itls"][i]) / (d["output_lens"][i] - 1) * 1e3 <= m * scale]
        return len(g) / len(ok) * 100, sum(d["output_lens"][i] for i in g) / d["duration"]
    t, l = att(80), att(100)
    print(a.split("/")[-1], f"agent n={len(ok)} att 10s/80ms={t[0]:.1f}% ({t[1]:.0f} tok/s good) 10s/100ms={l[0]:.1f}% ({l[1]:.0f} tok/s)")
