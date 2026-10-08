"""Build the mixed-attention kernel gate reference table from a 3-way sweep.

Input: gate_sweep3 JSON (see tasks/debug_kernel_server_perf/gate_sweep3.py)
with per-case bp_ms / pa_flip_ms / pa_noflip_ms.

Output: compact k-NN reference table consumed by
sglang.srt.layers.attention.flashinfer_backend.MixedKernelGate.

Label policy: use the persistent kernel only where its best-schedule time
beats BatchPrefill by >= 10%, and only when all K=5 nearest samples agree.
On the merged H800 sweep (uniform + fresh-prefix + blended-KV, 1316 points)
this gives leave-one-out worst-case regression 1.000 with 81% of the
available win captured and a 2.71x mean speedup where persistent is chosen.
A 5% label admits one 1.039x regression; k=3 admits 1.4-1.6x ones on the
steep gradients of the fresh-prefix region. The flip label is the faster
schedule per point.
"""

import argparse
import json

import numpy as np

LABEL_SPEEDUP = 1.10
KNN_K = 5


def features(dec_kvs, p_lens, p_prefix):
    """Feature vector for one mixed-batch composition.

    The KV-concentration term is essential: BatchPrefill's cost tracks the
    *per-row* context length, because a handful of long rows produce only a
    handful of query tiles and leave most SMs idle while each streams its whole
    KV. Total KV and batch size cannot tell "8 rows at 64K" from "128 rows at
    4K" -- measured 8.2x versus 1.02x -- so both are needed.
    """
    dec = np.asarray(dec_kvs, dtype=np.float64)
    D = float(dec.sum())
    P = float(sum(q * c for q, c in zip(p_lens, p_prefix)))
    Q = float(sum(p_lens))
    n_dec = len(dec_kvs)
    conc = float((dec * dec).sum() / D) if D > 0 else 0.0
    return [
        float(np.log1p(D)),
        float(np.log1p(P)),
        float(np.log1p(Q)),
        float(np.log1p(n_dec)),
        3.0 * D / (D + P + 1),
        float(np.log1p(conc)),
    ]


def row_decode_kvs(r):
    """Uniform-KV sweep rows store (d_bs, d_kv); blended rows store dec_kvs."""
    if "dec_kvs" in r:
        return r["dec_kvs"]
    return [r["d_kv"] - 1] * r["d_bs"]


def main():
    global KNN_K, LABEL_SPEEDUP
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_json")
    ap.add_argument("out_json")
    ap.add_argument("--gqa", type=int, required=True)
    ap.add_argument("--gpu", type=str, default="H800")
    ap.add_argument("--note", type=str, default="")
    ap.add_argument("--k", type=int, default=KNN_K,
                    help="k-NN neighbours that must unanimously agree. Raising k "
                         "tightens the accept region; k=7 is what keeps worst-case "
                         "leave-one-out regression at 1.000 once real serving "
                         "compositions are mixed into the grid.")
    ap.add_argument("--label", type=float, default=LABEL_SPEEDUP)
    args = ap.parse_args()
    KNN_K, LABEL_SPEEDUP = args.k, args.label

    rs = []
    for path in args.sweep_json.split(","):
        rs.extend(json.load(open(path)))
    feats, use_pa, flip = [], [], []
    for r in rs:
        best = min(r["pa_flip_ms"], r["pa_noflip_ms"])
        feats.append(features(row_decode_kvs(r), r["p_lens"], r["p_prefix"]))
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
        "feature_names": [
            "log1p_D", "log1p_P", "log1p_Q", "log1p_n_dec",
            "3*D/(D+P+1)", "log1p_kv_concentration",
        ],
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
