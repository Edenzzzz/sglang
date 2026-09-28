#!/usr/bin/env python3
"""Max resident batch size per hardware config, from model size and KV bytes.

Used to pick realistic decode-row counts for the large-model workload sets.
Everything is derived from published shapes; no measurement on those GPUs.
"""
import argparse

# name: (layers, qo_heads, kv_heads, head_dim, params_B, moe_active_B)
MODELS = {
    "Qwen2.5-7B":       (28, 28,  4, 128,   7.6, 7.6),
    "Qwen3-8B":         (36, 32,  8, 128,   8.2, 8.2),
    "Qwen3-235B-A22B":  (94, 64,  4, 128, 235.0, 22.0),
}
# name: (GPU mem GB, GPUs per TP domain, TP domains, note)
HW = {
    "H800 x1 (local)":        (80,  1,  1, "this machine"),
    "GH200 4x4":              (96,  4,  4, "local TP4, cross-node DP4"),
    "H200 4x8":              (141,  8,  4, "local TP8, cross-node DP4"),
    "GB200 NVL72 (TP8)":     (186,  8,  9, "NVL72, 72 GPUs, TP8 x DP9"),
    "GB200 NVL72 (TP16)":    (186, 16,  4, "NVL72, larger TP domain"),
}


def kv_bytes_per_token(layers, kv_heads, head_dim, dtype=2, tp=1):
    """KV is sharded across TP, so per-GPU cost divides by tp."""
    return 2 * layers * kv_heads * head_dim * dtype / tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem-frac", type=float, default=0.85)
    ap.add_argument("--weight-dtype", type=float, default=2.0, help="bytes/param (2=bf16, 1=fp8)")
    ap.add_argument("--ctx", type=int, nargs="+", default=[8192, 32768, 131072])
    a = ap.parse_args()

    print(f"mem_fraction_static={a.mem_frac}, weights at {a.weight_dtype} B/param\n")
    for mname, (L, qo, kv, hd, params, _) in MODELS.items():
        print(f"=== {mname}  ({L} layers, qo/kv {qo}/{kv}, gqa {qo//kv}, head_dim {hd})")
        for hname, (mem, tp, dp, note) in HW.items():
            # weights shard across the TP domain; KV likewise
            w_gb = params * a.weight_dtype / tp
            static = mem * a.mem_frac
            kv_gb = static - w_gb
            if kv_gb <= 1:
                print(f"   {hname:<22} weights {w_gb:6.1f} GB/GPU > budget {static:5.1f} -> DOES NOT FIT")
                continue
            bpt = kv_bytes_per_token(L, kv, hd, tp=tp)
            toks = kv_gb * 1e9 / bpt
            cols = "  ".join(f"{c//1024}K:{toks/c:>5.0f}" for c in a.ctx)
            print(f"   {hname:<22} weights {w_gb:6.1f} GB/GPU, KV {kv_gb:5.1f} GB -> "
                  f"{toks/1e3:7.0f}K tok/GPU | max resident reqs @ctx  {cols}")
        print()


if __name__ == "__main__":
    main()
