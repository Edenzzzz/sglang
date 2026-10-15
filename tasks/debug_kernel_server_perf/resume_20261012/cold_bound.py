import json, sys, numpy as np
# Qwen2.5-7B: 28 layers, 28 query heads, head_dim 128; causal attention FLOPs for a prefill of u new tokens
# after a cached prefix p: 4 * d * H * L * sum over new tokens of (p + i) = 4*d*H*L*(u*p + u^2/2).
# Linear (GEMM) FLOPs: 2 * params(7.6e9) per token.
L, H, D, P = 28, 28, 128, 7.6e9
for f in sys.argv[1:]:
    d = json.load(open(f))
    t = np.array(d["ttfts"]); inp = np.array(d["input_lens"]); c = np.array(d.get("cached_tokens") or [0]*len(t))
    u = inp - c
    m = u >= 32768
    attn = 4 * D * H * L * (u * c + u * u / 2.0)
    lin = 2 * P * u
    fl = (attn + lin)[m]
    print(f"{f.split('/')[-2]:12s} cold n={m.sum()} uncached mean={u[m].mean()/1e3:.0f}K TTFT mean={t[m].mean():.1f}s "
          f"PFLOP total={fl.sum()/1e15:.1f} achieved={fl.sum()/t[m].sum()/1e12:.0f} TFLOP/s (TTFT incl. queueing) "
          f"| share of all TTFT={t[m].sum()/t.sum():.2f} | at 700 TFLOP/s floor={fl.sum()/700e12/m.sum():.1f}s/req")
