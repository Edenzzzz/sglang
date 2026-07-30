"""Build the mixed-attention kernel gate reference table from a 3-way sweep.

Input: gate_sweep3 JSON (see tasks/debug_kernel_server_perf/gate_sweep3.py)
with per-case bp_ms / pa_flip_ms / pa_noflip_ms.

Output: compact k-NN reference table consumed by
sglang.srt.layers.attention.flashinfer_backend.MixedKernelGate.

Label policy: use the persistent kernel only where its best-schedule time
beats BatchPrefill by >= 5% (leave-one-out worst-case regression 2.8% on the
H800 sweep); flip label is the faster schedule per point.
"""

import argparse
import json

import numpy as np

LABEL_SPEEDUP = 1.05


def features(D, P, Q, d_bs):
    return [
        float(np.log1p(D)),
        float(np.log1p(P)),
        float(np.log1p(Q)),
        float(np.log1p(d_bs)),
        3.0 * D / (D + P + 1),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_json")
    ap.add_argument("out_json")
    ap.add_argument("--gqa", type=int, required=True)
    ap.add_argument("--gpu", type=str, default="H800")
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()

    rs = json.load(open(args.sweep_json))
    feats, use_pa, flip = [], [], []
    for r in rs:
        D = r["d_bs"] * (r["d_kv"] - 1)
        P = sum(q * c for q, c in zip(r["p_lens"], r["p_prefix"]))
        Q = sum(r["p_lens"])
        best = min(r["pa_flip_ms"], r["pa_noflip_ms"])
        feats.append(features(D, P, Q, r["d_bs"]))
        use_pa.append(int(r["bp_ms"] / best >= LABEL_SPEEDUP))
        flip.append(int(r["pa_flip_ms"] <= r["pa_noflip_ms"]))

    X = np.array(feats)
    mu, sd = X.mean(0), X.std(0)
    Xs = (X - mu) / sd

    out = {
        "meta": {
            "gpu": args.gpu,
            "gqa": args.gqa,
            "sweep": args.sweep_json,
            "label_speedup": LABEL_SPEEDUP,
            "note": args.note,
        },
        "feature_names": ["log1p_D", "log1p_P", "log1p_Q", "log1p_d_bs", "3*D/(D+P+1)"],
        "mu": [round(float(v), 6) for v in mu],
        "sd": [round(float(v), 6) for v in sd],
        "points": [[round(float(v), 4) for v in row] for row in Xs],
        "use_pa": use_pa,
        "flip": flip,
    }
    json.dump(out, open(args.out_json, "w"))
    print(f"{len(rs)} points -> {args.out_json}; use_pa on {sum(use_pa)}, flip on {sum(flip)}")


if __name__ == "__main__":
    main()
