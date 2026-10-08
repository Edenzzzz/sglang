#!/usr/bin/env python3
"""Fit the analytic overlap cost model's per-GPU constants.

Inputs: overlap_attribution.json (bench_overlap_attribution.py) for the
attention law; two serving anchors for the non-attention line.

    T_d(layer)  = a_d * decode_kv + c_d           (bandwidth roofline)
    T_p(layer)  = a_p * (n*prefix) + c_p          (compute roofline)
    T_attn      = max + beta * min
    T_step      = c0 + c1 * tokens + L * T_attn

c0 ~= weights_bytes / HBM_BW + fixed overhead; c1 ~= 2 * params / eff_FLOPs.
Both are also validated against measured serving anchors.
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribution", default="tasks/debug_kernel_server_perf/overlap_attribution.json")
    ap.add_argument("--out", default="tasks/debug_kernel_server_perf/overlap_cost_h800_qwen25.json")
    ap.add_argument("--n-layers", type=int, default=28)
    ap.add_argument("--params-gb", type=float, default=15.2, help="model weights, GB (bf16)")
    ap.add_argument("--hbm-tbs", type=float, default=3.35, help="HBM bandwidth TB/s")
    ap.add_argument("--eff-tflops", type=float, default=600.0, help="effective bf16 TFLOP/s")
    ap.add_argument("--fixed-overhead-ms", type=float, default=0.6)
    args = ap.parse_args()

    rs = json.load(open(args.attribution))
    Td = np.array([r["T_d"] for r in rs])
    Tp = np.array([r["T_p"] for r in rs])
    Tm = np.array([r["T_mix"] for r in rs])
    Dkv = np.array([r["n_d"] * (r["d_kv"] - 1) for r in rs], dtype=float)
    Pw = np.array([r["q"] * r["pfx"] for r in rs], dtype=float)

    wd = 1.0 / Td
    A = np.stack([Dkv * wd, wd], 1)
    (a_d, c_d), *_ = np.linalg.lstsq(A, Td * wd, rcond=None)
    s = Pw > 0
    # Weight by 1/T so relative error is minimized: an absolute fit lets the
    # long-prefix points set the intercept and overestimates small chunks 14x.
    w = 1.0 / Tp[s]
    A = np.stack([Pw[s] * w, w], 1)
    (a_p, c_p), *_ = np.linalg.lstsq(A, Tp[s] * w, rcond=None)

    mx, mn = np.maximum(Tp, Td), np.minimum(Tp, Td)
    beta = float(np.linalg.lstsq(mn.reshape(-1, 1), Tm - mx, rcond=None)[0][0])
    mape = float(np.mean(np.abs(mx + beta * mn - Tm) / Tm))

    # Non-attention line from physical constants.
    c0 = args.params_gb / args.hbm_tbs + args.fixed_overhead_ms       # ms
    c1 = 2.0 * args.params_gb / 2.0 * 1e9 / (args.eff_tflops * 1e12) * 1e3
    # 2*params FLOP per token; params = GB/2 (bf16). Result in ms/token.

    out = dict(
        a_d=float(a_d), c_d=float(c_d), a_p=float(a_p), c_p=float(c_p),
        beta=beta, c0=float(c0), c1=float(c1), n_layers=args.n_layers,
    )
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"attention-law MAPE on attribution set: {mape*100:.1f}%")

    # Serving anchors (measured in the co-tenancy benchmarks):
    def step(n_p, r_p, avg_kv, bs):
        dk = avg_kv * bs
        t_d = a_d * dk + c_d if dk else 0.0
        t_p = a_p * (n_p * r_p) + c_p if n_p else 0.0
        attn = (max(t_p, t_d) + beta * min(t_p, t_d)) if t_p and t_d else t_p + t_d
        return c0 + c1 * (n_p + bs) + args.n_layers * attn

    print(f"anchor decode bs=47 @ ~11K avg KV: predicted {step(0,0,11000,47):.1f} ms "
          f"(measured serving mean ITL ~15.8 ms)")
    print(f"anchor mixed 450q@pfx0 over that pool: predicted {step(450,0,11000,47):.1f} ms")


if __name__ == "__main__":
    main()
