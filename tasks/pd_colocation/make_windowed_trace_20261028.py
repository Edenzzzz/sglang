"""Cut the TraceLab release to the sessions the windowed replay can select, preserving that replay exactly.

The full release (workloads/tracelab/syfi_coding_trace.jsonl.gz, 357,161 rounds, 54 MB) takes ~2.5 h to push through
the cluster's base64-over-exec channel. The superposed timestamp replay (AGENT_MODE=tracelab_ts with WINDOW_STARTS)
only ever selects sessions that have rounds inside its windows, so the rest is dead weight. Two things must hold for
the cut to be exact, and both are checked:
  1. The window ORIGIN is unchanged. tracelab.py measures every window start in seconds after the trace's first
     round, so dropping the session that holds the earliest event would shift every window onto different hours
     with no error. The anchor session is always kept.
  2. Every round with an event inside a window is present.
Whole sessions are kept (never single rounds), so per-session prefix reconstruction is untouched.
--slim additionally drops every field tracelab.py never reads, keeping session_id, round_index, provider, the four
token counts, and only the EARLIEST and LATEST timing event (tracelab.py's _record_span_ms uses exactly min and max
of the event timestamps, and nothing else from an event). Both kept fields and per-record span are asserted equal.
  python make_windowed_trace_20261028.py <full trace> <window json> <out> [--window-s 3600] [--slim]
Claude.
"""
import argparse
import gzip
import json
from datetime import datetime


def ts(event):
    return datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("windows")
    ap.add_argument("out")
    ap.add_argument("--window-s", type=float, default=3600)
    ap.add_argument("--slim", action="store_true")
    a = ap.parse_args()
    starts = json.load(open(a.windows))
    recs, lo, lo_sid = [], None, None
    with gzip.open(a.trace, "rt") as f:
        for line in f:
            r = json.loads(line)
            recs.append((r["session_id"], line, r))
            for e in r.get("timing_events") or []:
                t = ts(e)
                if lo is None or t < lo:
                    lo, lo_sid = t, r["session_id"]
    wins = [(lo + s, lo + s + a.window_s) for s in starts]

    def in_window(r):
        return any(any(x <= ts(e) < y for x, y in wins) for e in (r.get("timing_events") or []))

    keep = {lo_sid} | {sid for sid, _, r in recs if in_window(r)}
    out = [line for sid, line, _ in recs if sid in keep]
    if a.slim:
        fields = ["session_id", "round_index", "provider", "input_tokens_total", "prefix_tokens",
                  "newly_append_tokens", "output_tokens"]

        def span(r):
            t = [ts(e) for e in (r.get("timing_events") or []) if e.get("timestamp")]
            return (min(t), max(t)) if t else (None, None)

        slim = []
        for line in out:
            r = json.loads(line)
            ev = [e for e in (r.get("timing_events") or []) if e.get("timestamp")]
            s_ = {k: r[k] for k in fields if k in r}
            s_["timing_events"] = ([{"timestamp": min(ev, key=ts)["timestamp"]},
                                    {"timestamp": max(ev, key=ts)["timestamp"]}] if ev else [])
            assert span(s_) == span(r) and all(s_.get(k) == r.get(k) for k in fields), "slim changed a record"
            slim.append(json.dumps(s_, separators=(",", ":")) + "\n")
        out = slim
    with gzip.open(a.out, "wt") as g:
        g.writelines(out)
    lo2 = min(ts(e) for line in out for e in (json.loads(line).get("timing_events") or []))
    full = sum(1 for _, _, r in recs if in_window(r))
    kept = sum(1 for line in out if in_window(json.loads(line)))
    assert lo2 == lo, "window origin moved"
    assert full == kept, "in-window rounds lost"
    print(f"origin unchanged; in-window rounds {kept}/{full}; sessions {len(keep)}; rounds {len(out)}/{len(recs)}")


if __name__ == "__main__":
    main()
