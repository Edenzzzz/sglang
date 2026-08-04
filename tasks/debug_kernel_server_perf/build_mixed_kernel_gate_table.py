"""Build the mixed-attention kernel gate reference table from a 3-way sweep.

Input: gate_sweep3 JSON (see tasks/debug_kernel_server_perf/gate_sweep3.py)
with per-case bp_ms / pa_flip_ms / pa_noflip_ms.

Output: compact k-NN reference table consumed by
sglang.srt.layers.attention.flashinfer_backend.MixedKernelGate.

Label policy: use the persistent kernel only where its best-schedule time
beats BatchPrefill by >= 5%, and only when all K nearest samples agree.
K=5 on the merged (base + fresh-prefix) H800 sweep gives leave-one-out
worst-case regression 1.000 with 75% of the available win captured; K=3 was
enough for the base sweep alone but admits 1.4-1.6x regressions on the
steep gradients of the fresh-prefix region. The flip label is the faster
schedule per point.
"""

import argparse
import json

import numpy as np

LABEL_SPEEDUP = 1.05
KNN_K = 5


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

    # Out-of-support guard. A k-NN silently extrapolates for queries far from
    # every sampled point, which produced exactly the wrong answer for
    # fresh-prefix chunks before that region was swept. Record how far a query
    # may sit from its nearest sample before the lookup stops being trusted.
    d2 = ((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn_d2 = d2.min(1)
    max_d2 = float(np.percentile(nn_d2, 99) * 4.0)

    out = {
        "meta": {
            "gpu": args.gpu,
            "gqa": args.gqa,
            "sweep": args.sweep_json,
            "label_speedup": LABEL_SPEEDUP,
            "note": args.note,
        },
        "knn_k": KNN_K,
        "feature_names": ["log1p_D", "log1p_P", "log1p_Q", "log1p_d_bs", "3*D/(D+P+1)"],
        "mu": [round(float(v), 6) for v in mu],
        "sd": [round(float(v), 6) for v in sd],
        "max_nn_dist_sq": round(max_d2, 6),
        "points": [[round(float(v), 4) for v in row] for row in Xs],
        "use_pa": use_pa,
        "flip": flip,
    }
    json.dump(out, open(args.out_json, "w"))
    print(f"{len(rs)} points -> {args.out_json}; use_pa on {sum(use_pa)}, flip on {sum(flip)}")


if __name__ == "__main__":
    main()
