#!/usr/bin/env python3
"""Parse the A3 v2 winner map: per (kv, rows, qo), the in-kernel time of FA2, FA3 q64 and FA3 q128
(co-launch build), each only if its fp32 verdict and replay check are clean, the winner, and the
native-FA2 references (BatchAttention, unfused two-launch). Usage: a3_parse_v2.py <log dir>"""
import glob, os, re, sys

rows = {}
for f in glob.glob(os.path.join(sys.argv[1], "*.log")):
    t = open(f).read()
    m = re.match(r"kv(\d+)_rows(\d+)_qo(\d+)_(.*)", os.path.basename(f)[:-4])
    kv, r, q, side = int(m[1]), int(m[2]), int(m[3]), m[4]
    bad = re.findall(r"bad rows (\d+)/(\d+)", t)
    rep = re.findall(r"FUSED REPLAY CORRECTNESS: bad_heads=(\d+)", t)
    ok = bool(bad) and bad[-1][0] == "0" and (not rep or rep[-1] == "0")
    num = lambda pat: (float(x[-1]) if (x := re.findall(pat, t)) else None)
    rows[(kv, r, q, side)] = (ok, num(r"\[time\] fused += ([0-9.]+)"),
                              num(r"\[time\] unfused = ([0-9.]+)"),
                              num(r"BatchAttention \(flash[^=]*= ([0-9.]+)"))
sides = ("fa2", "fa3_q64", "fa3_q128")
bad = [k for k, v in rows.items() if not v[0]]
print(f"arms={len(rows)} failing={len(bad)} {sorted(bad)[:8]}")
print("kv rows qo | fa2 | fa3_q64 | fa3_q128 | winner | BA | unfused-fa2 | best/BA")
for kv in sorted({k[0] for k in rows}):
    for r in sorted({k[1] for k in rows}):
        for q in sorted({k[2] for k in rows}):
            v = {s: rows.get((kv, r, q, s)) for s in sides}
            if not all(v.values()):
                continue
            cand = {s: x[1] for s, x in v.items() if x[0] and x[1]}
            w = min(cand, key=cand.get) if cand else "-"
            ba = v["fa2"][3]
            un = v["fa2"][2]
            fmt = lambda x: ("%.4f" % x[1] if x[1] else "NA") + ("" if x[0] else "!")
            print(kv, r, q, "|", " | ".join(fmt(v[s]) for s in sides), "|", w, "|",
                  "%.4f" % ba if ba else "NA", "|", "%.4f" % un if un else "NA", "|",
                  "%.2f" % (cand[w] / ba) if cand and ba else "")
