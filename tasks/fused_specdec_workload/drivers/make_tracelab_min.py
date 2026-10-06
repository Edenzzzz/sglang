"""Shrink the TraceLab trace to what sample_tracelab_claude_requests actually reads.

The full trace is 52 MB, too large for the pod's ~60 KB-per-call webshell channel. The
sampler only reads provider=="claude" rows with prefix>0 and append>0, seven scalar fields,
and the MIN and MAX timestamp of timing_events. Keeping exactly that (and context >= 16K,
below every scenario's range) gives a 4 MB file.

Verified to produce byte-identical DatasetRows to the full trace under the same seed, at
n=60/12 groups/32-49K and n=40/8 groups/64-100K.
"""
import gzip, json, sys
from datetime import datetime

KEEP = ["input_tokens_total", "newly_append_tokens", "output_tokens", "prefix_tokens",
        "provider", "round_index", "session_id"]


def _ms(t):
    return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp() * 1000


def main(src, dst, min_ctx=16384):
    n = 0
    with gzip.open(src, "rt") as f, gzip.open(dst, "wt", compresslevel=9) as g:
        for line in f:
            r = json.loads(line)
            if r.get("provider") != "claude":
                continue
            if int(r.get("prefix_tokens") or 0) <= 0 or int(r.get("newly_append_tokens") or 0) <= 0:
                continue
            if int(r.get("input_tokens_total") or 0) < min_ctx:
                continue
            o = {k: r.get(k) for k in KEEP}
            ev = sorted((e for e in (r.get("timing_events") or []) if e.get("timestamp")),
                        key=lambda e: _ms(e["timestamp"]))
            o["timing_events"] = ([{"timestamp": ev[0]["timestamp"]},
                                   {"timestamp": ev[-1]["timestamp"]}] if ev else [])
            g.write(json.dumps(o, separators=(",", ":")) + "\n")
            n += 1
    print(f"wrote {n} rows to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
