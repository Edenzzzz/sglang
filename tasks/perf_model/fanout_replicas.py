#!/usr/bin/env python3
"""Replica count per hot prefix over time on the fan-out workload (PD point 2b): from router_decisions.jsonl lines
that carry client_rid "fo-t<task>-..." (fanout_agents.py tags each request with its task id), count the distinct
workers each task prefix was routed to, per popularity epoch, and how many routings went to a worker that had not
served that task before (each such routing leaves a new replica of the prefix).
  python fanout_replicas.py <arm dir> [...]"""
import collections, json, os, re, sys

for d in sys.argv[1:]:
    fo = json.load(open(os.path.join(d, "fanout.json")))
    rot = fo["args"]["rotate_s"]
    t0 = None
    seen = collections.defaultdict(set)
    per_epoch = collections.defaultdict(lambda: collections.defaultdict(set))
    new_copies = total = 0
    for line in open(os.path.join(d, "router_decisions.jsonl")):
        try:
            x = json.loads(line)
        except Exception:
            continue
        m = re.match(r"fo-t(\d+)-", str(x.get("client_rid") or ""))
        if not m or "t" not in x:
            continue
        task, w = int(m.group(1)), x["worker"]
        t0 = x["t"] if t0 is None else t0
        e = int((x["t"] - t0) // rot)
        total += 1
        if w not in seen[task]:
            new_copies += 1
            seen[task].add(w)
        per_epoch[e][task].add(w)
    print(f"{os.path.basename(d)}: routings={total} new-replica routings={new_copies} "
          f"workers per task overall: " + " ".join(f"t{t}:{len(v)}" for t, v in sorted(seen.items())))
    for e in sorted(per_epoch):
        print(f"  epoch {e}: " + " ".join(f"t{t}:{len(v)}" for t, v in sorted(per_epoch[e].items())))
