"""Per-tenant scoring of TraceLab timestamp-replay arms (mn_bench.sh AGENT_MODE=tracelab_ts) under the paper's SLOs,
overall and in time windows.

Inputs per arm directory (bench_serving --output-details): agent.json, chat.json, optional chat_spike.json (a chat
rate step, CHAT_SPIKE) and math/math.json (a third tenant). Each request's start is request_start_epoch, so the
streams share one time axis; its end is start + TTFT + the sum of its inter-token gaps.

SLOs (paper matrix): chat and math TTFT 2 s with mean TPOT 50 ms (tight) / 100 ms (loose); agent rounds TTFT 10 s,
20 s when the round has more than 200K uncached tokens (prompt minus cached tokens), mean TPOT 80 / 100 ms. TPOT is
sum(itls) / (output_len - 1), so MTP arms compare fairly.

Output: one PAPER line per arm. Per tenant it reports SLO attainment at both levels, then mean TTFT and mean TPOT
each carried by a p10-p99 spread as an error bar. It does NOT report an SLO-attaining token rate: that is a third
axis moving with attainment, latency and the request mix at once, and attainment plus the two latency means already
answer what the SLO asks. Where capacity itself is the question, the capacity figure answers it. With --windows S
it also writes a JSON time series per arm
(<dir>/tracelab_windows.json): per window, offered prefill demand (uncached prompt tokens of requests starting in it,
per second), offered decode demand (their output tokens per second), requests started and each tenant's attainment.
  python tracelab_windows.py [--windows 30] DIR [DIR ...]
Claude, 2026-10-26.
"""
import argparse
import json
from pathlib import Path

import numpy as np

CHAT = {"tight": (2.0, 50.0), "loose": (2.0, 100.0)}
AGENT_TPOT = {"tight": 80.0, "loose": 100.0}
LONG_UNCACHED = 200_000
NOT_ISSUED = {}  # tenant -> rounds past the replay horizon (BENCH_REPLAY_HORIZON_S): the arm's backlog
LOST = {}  # tenant -> rounds never issued because an earlier round of their session failed


def load(path, tenant):
    if not path.exists():
        return []
    d = json.loads(path.read_text().splitlines()[-1])
    starts = d.get("request_start_epoch")
    if starts is None:
        raise SystemExit(f"{path}: no request_start_epoch (client predates the windowed-scoring overlay)")
    cached = d.get("cached_tokens") or [0] * len(starts)
    out = []
    for i, error in enumerate(d["errors"]):
        if error and error.startswith("not issued: failed predecessor"):
            LOST[tenant] = LOST.get(tenant, 0) + 1
            continue
        if error and error.startswith("not issued"):
            NOT_ISSUED[tenant] = NOT_ISSUED.get(tenant, 0) + 1
            continue
        n = d["output_lens"][i]
        # A one-token reply (the trace has them) has no inter-token gap: it is judged on TTFT alone.
        ok = not error and n >= 1 and d["ttfts"][i] is not None
        itl = sum(d["itls"][i]) if ok else 0.0
        gaps = d["itls"][i] if ok else []
        out.append(dict(tenant=tenant, ok=ok, start=starts[i], ttft=d["ttfts"][i] if ok else None,
                        tpot_ms=(itl / (n - 1) * 1e3 if n > 1 else 0.0) if ok else None,
                        end=starts[i] + (d["ttfts"][i] or 0) + itl, gaps=gaps,
                        out=n, uncached=max(0, d["input_lens"][i] - (cached[i] or 0))))
    return out


def meets(r, level):
    if not r["ok"]:
        return False
    if r["tenant"] == "agent":
        ttft = 20.0 if r["uncached"] > LONG_UNCACHED else 10.0
        return r["ttft"] <= ttft and r["tpot_ms"] <= AGENT_TPOT[level]
    ttft, tpot = CHAT[level]
    return r["ttft"] <= ttft and r["tpot_ms"] <= tpot


def summary(rows, duration):
    parts = []
    for tenant in ("chat", "chat_spike", "math", "agent"):
        rs = [r for r in rows if r["tenant"] == tenant]
        if not rs:
            continue
        cells = []
        # ATTAINMENT ONLY, no token rate (user, 2026-10-29). The SLO-attaining token rate ("goodput") is a THIRD
        # axis on top of attainment and latency, and it moves with both of them plus the request mix, so a reader
        # has to hold three numbers to decide whether an arm is better. Attainment already answers the question the
        # SLO poses -- what share of requests were served inside their target -- and the latency means below say by
        # how much they made or missed it. A rate is still needed to say where capacity lies, which is what the
        # capacity figure is for; it does not belong in the per-arm line.
        for level in ("tight", "loose"):
            good = [r for r in rs if meets(r, level)]
            cells.append(f"{level}={len(good) / len(rs) * 100:5.1f}%")
        done = [r for r in rs if r["ok"]]
        p = lambda xs, q: float(np.percentile(xs, q)) if len(xs) else float("nan")
        if NOT_ISSUED.get(tenant):
            cells.append(f"not_issued={NOT_ISSUED[tenant]}")
        if LOST.get(tenant):
            cells.append(f"lost_after_failure={LOST[tenant]}")
        # Crossflow scores agentic serving on ITL (mean, and a 120 ms p99 constraint), where SLOs-Serve / JITServe
        # and our attainment use a request's mean TPOT. Report TPOT only.
        # A per-request mean is what the SLO is written against and what a user feels -- their own request's
        # typical token rate. An inter-token percentile is taken over every gap in the run, so it is dominated by
        # single long steps and is not comparable to a TPOT percentile; reporting both side by side invited
        # reading one as the other.
        # MEAN with a p10-p99 spread, for TTFT and TPOT (user, 2026-10-29). The mean is the headline because it is
        # the quantity the SLO is written against; p10 and p99 carry it as an error bar so a mean cannot hide a
        # tail. The spread is deliberately asymmetric: a latency distribution is bounded below and has a long upper
        # tail, so a symmetric bar around the mean would misdescribe it, and p99 is where queueing shows up.
        tt = [r["ttft"] for r in done]
        tp = [r["tpot_ms"] for r in done]
        cells.append(f"TTFT mean={sum(tt) / len(tt) if tt else float('nan'):.2f}s "
                     f"[p10 {p(tt, 10):.2f} p99 {p(tt, 99):.2f}] "
                     f"TPOT mean={sum(tp) / len(tp) if tp else float('nan'):.1f}ms "
                     f"[p10 {p(tp, 10):.1f} p99 {p(tp, 99):.1f}] "
                     f"n={len(rs)} failed={len(rs) - len(done)}")
        parts.append(f"{tenant}: " + " ".join(cells))
    return " | ".join(parts)


def windows(rows, t0, width):
    end = max(r["end"] for r in rows)
    series = []
    for lo in np.arange(t0, end, width):
        hi = lo + width
        inside = [r for r in rows if lo <= r["start"] < hi]
        entry = dict(t=float(lo - t0), prefill_demand_tok_s=sum(r["uncached"] for r in inside) / width,
                     decode_demand_tok_s=sum(r["out"] for r in inside) / width, started=len(inside))
        for tenant in sorted({r["tenant"] for r in rows}):
            mine = [r for r in inside if r["tenant"] == tenant]
            entry[f"{tenant}_n"] = len(mine)
            entry[f"{tenant}_attain_tight"] = (sum(meets(r, "tight") for r in mine) / len(mine) * 100) if mine else None
        series.append(entry)
    return series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=float, default=0.0, help="window width in seconds (0: no time series)")
    parser.add_argument("dirs", nargs="+")
    args = parser.parse_args()
    for d in map(Path, args.dirs):
        NOT_ISSUED.clear()
        LOST.clear()
        rows = (load(d / "agent.json", "agent") + load(d / "chat.json", "chat")
                + load(d / "chat_spike.json", "chat_spike") + load(d / "math" / "math.json", "math"))
        # An arm whose workers never came up leaves a directory with no records at all. Report that as its own
        # outcome and carry on: aborting here discards the scores of every arm after it in the batch, which is how
        # one dead pdmux arm stopped a 40-arm re-scoring pass midway. "No records" is a result, not an error.
        if not rows:
            print(f"PAPER {d.name} NO_RECORDS -- the arm produced no request records; it measured nothing",
                  flush=True)
            continue
        t0 = min(r["start"] for r in rows)
        duration = max(r["end"] for r in rows) - t0
        print(f"PAPER {d.name} duration={duration:.0f}s {summary(rows, duration)}", flush=True)
        if args.windows > 0:
            (d / "tracelab_windows.json").write_text(json.dumps(windows(rows, t0, args.windows), indent=1) + "\n")


if __name__ == "__main__":
    main()
