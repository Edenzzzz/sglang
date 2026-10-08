#!/usr/bin/env python3
"""Does GQA packing move the wgmma/FA2 boundary for the SHORT-q classes?

This decides how much of the dispatch belongs on each mainloop, and therefore
whether the FA2 overlap machinery is still wanted at all:

  unpacked FA3 (what the server plans today) tiles qo_len and emits one work item
  per qo head, so a 4-token verify row occupies 4 of 128 tile rows, 28 times over.
  packed FA3 presents each KV head's group as one head with qo_len*gqa rows, so
  the same row becomes 28 packed rows in ONE tile per kv head -- 7x less tile
  waste at gqa 7. Packing is legal here because the cached-prefix part is
  non-causal, which is exactly how the persistent kernel already tiles.

Packing is emulated at the batch level (num_qo_heads = num_kv_heads, qo_len
multiplied by the group), so the boundary can be measured before the mainloop
change lands. Timing a packed FA3 against the persistent kernel per mode class is
the number that says which classes to hand over.
"""
import argparse, json, sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_4mode import Bench, uncapped, med

# (label, qo_len, kv_len, rows) covering the four scheduling classes at the
# measured online shapes: decode/draft, verify at two k, and prefill chunks.
CLASSES = [
    ("decode/draft    ",    1,  4357, 11),
    ("verify k=3      ",    4,  4357,  8),
    ("verify k=7      ",    8,  4357,  8),
    ("small chunk     ",  128,  3095,  1),
    ("modal chunk     ",  500,  3095,  1),
    ("long-ctx chunk  ",  500, 32768,  1),
    ("big chunk       ", 2048,  3095,  1),
]


def fa3_packed(b, qo, kv, rows, gqa):
    """FA3 with each KV head's group packed into one head: qo_len * gqa rows."""
    saved = b.H_QO
    b.H_QO = b.H_KV                      # one q head per kv head
    try:
        qf = b.plan_fa3([(qo * gqa, kv)] * rows)
        run = lambda: b.fa3.run(qf, b.kv[: rows * kv])
        run()
    finally:
        b.H_QO = saved
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="28,4,128")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    heads = tuple(int(x) for x in a.heads.split(","))
    gqa = heads[0] // heads[1]
    b = Bench(heads)
    uncapped()
    print(f"heads {heads}  gqa {gqa}\n")
    print(f"{'class':<18}{'qo':>5}{'kv':>7}{'rows':>5}"
          f"{'pers us':>10}{'FA3 raw':>10}{'FA3 pack':>10}"
          f"{'pack/raw':>10}{'pers/pack':>11}{'winner':>9}")
    out = []
    for label, qo, kv, rows in CLASSES:
        pages = rows * kv
        q, o, lse = b.plan_pa([(qo, kv)] * rows, False)
        t_pers = med(lambda: b.pa.run(q, b.kv[:pages], out=o, lse=lse))
        qf = b.plan_fa3([(qo, kv)] * rows)
        t_raw = med(lambda: b.fa3.run(qf, b.kv[:pages]))
        try:
            t_pack = med(fa3_packed(b, qo, kv, rows, gqa))
        except Exception as e:
            print(f"{label:<18}{qo:>5}{kv:>7}{rows:>5}  packed FAILED: {str(e)[:50]}")
            continue
        win = "FA3pack" if t_pack < t_pers else "pers"
        print(f"{label:<18}{qo:>5}{kv:>7}{rows:>5}{t_pers:>10.1f}{t_raw:>10.1f}"
              f"{t_pack:>10.1f}{t_raw / t_pack:>10.2f}{t_pers / t_pack:>11.2f}{win:>9}")
        out.append(dict(cls=label.strip(), qo=qo, kv=kv, rows=rows, pers_us=t_pers,
                        fa3_raw_us=t_raw, fa3_pack_us=t_pack,
                        pack_over_raw=t_raw / t_pack, pers_over_pack=t_pers / t_pack,
                        winner=win))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
