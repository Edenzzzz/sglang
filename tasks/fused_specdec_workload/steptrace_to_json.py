#!/usr/bin/env python3
"""Turn per-step server traces into a replayable JSON workload corpus.

The point is that kernel work should iterate against what a live server actually
produced, not against hand-written shapes. Both arms run the identical request
stream, so the steps pair up by index and each step carries the measured GPU span
for the fused and unfused arm side by side.

Usage: steptrace_to_json.py out.json fused=<trace> baseline=<trace> [--meta k=v ...]
"""
import argparse, json, re, sys

FEATS = ["et", "bs", "np", "ptok", "pf", "nd", "dkv", "kv", "mx"]
LINE = re.compile(r"STEPTRACE (\w+) " + " ".join(f"{f}=(-?\\d+)" for f in FEATS)
                  + r" gpu=([0-9.]+)")


def parse(path):
    out = []
    for ln in open(path, errors="replace"):
        m = LINE.search(ln)
        if m:
            d = {"mode": m.group(1)}
            d.update({f: int(m.group(i + 2)) for i, f in enumerate(FEATS)})
            d["gpu_ms"] = float(m.group(len(FEATS) + 2))
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("arms", nargs="+", help="name=path")
    ap.add_argument("--meta", action="append", default=[])
    a = ap.parse_args()

    arms = {}
    for spec in a.arms:
        name, path = spec.split("=", 1)
        arms[name] = parse(path)
    n = min(len(v) for v in arms.values())
    names = list(arms)

    steps = []
    for i in range(n):
        base = {k: arms[names[0]][i][k] for k in ["mode"] + FEATS}
        # Arms only pair up if they really ran the same step; say so rather than
        # assuming it, because a mismatched pairing would silently fabricate a
        # per-arm comparison.
        agree = all(all(arms[nm][i][k] == base[k] for k in ["mode"] + FEATS) for nm in names)
        base["shapes_agree"] = agree
        for nm in names:
            base[f"gpu_ms_{nm}"] = arms[nm][i]["gpu_ms"]
        base["i"] = i
        steps.append(base)

    agreed = sum(s["shapes_agree"] for s in steps)
    doc = {
        "meta": dict(kv.split("=", 1) for kv in a.meta),
        "schema": {
            "mode": "forward mode: MIXED | EXTEND | DECODE",
            "et": "extend_num_tokens for the step",
            "bs": "batch size (rows)",
            "np": "prefill rows, i.e. rows whose extend length is > 1",
            "ptok": "new tokens across those prefill rows",
            "pf": "cached prefix behind those prefill rows, summed",
            "nd": "decode rows, i.e. rows whose extend length is 1 "
                  "(a mixed batch carries decode rows this way)",
            "dkv": "KV behind the decode rows, summed",
            "kv": "total KV across all rows",
            "mx": "longest single chunk",
            "gpu_ms_<arm>": "that step's own GPU span, from CUDA events read one "
                            "step later so nothing synchronises on the hot path",
            "shapes_agree": "true when every arm presented identical shapes for "
                            "this step index, so the per-arm times are comparable",
        },
        "row_reconstruction": (
            "Per-row shapes are not logged individually. For kernel replay, a step "
            "is np prefill rows and nd decode rows: give one prefill row mx tokens "
            "with prefix pf/np and the rest (ptok - mx)/(np - 1) each, and nd "
            "decode rows of 1 token with dkv/nd KV. The medians of this "
            "reconstruction are the shapes used by row_cmp.py and attn_share.py."
        ),
        "steps_total": n,
        "steps_with_agreeing_shapes": agreed,
        "steps": steps,
    }
    json.dump(doc, open(a.out, "w"), indent=1)
    print(f"{a.out}: {n} steps, {agreed} with identical shapes across "
          f"{len(names)} arms ({', '.join(names)})")
    for mode in ("MIXED", "EXTEND", "DECODE"):
        k = [s for s in steps if s["mode"] == mode]
        if k:
            print(f"  {mode:<7} {len(k):>5} steps")


if __name__ == "__main__":
    main()
