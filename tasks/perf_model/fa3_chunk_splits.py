"""FA3 time for the 4-mode mixed step's prefill chunk (one request, q = chunk tokens, causal, over a deep cached prefix)
as a function of num_splits / pack_gqa: does the auto heuristic leave SMs idle when chunk tiles < 1-2 waves?
  python fa3_chunk_splits.py   (Qwen2.5-7B shapes: 28 q heads, 4 kv heads, d128, page 1)
Claude, 2026-10-19.
"""
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

HQ, HK, D = 28, 4, 128
dev = "cuda"


def bench(fn, it=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1e3


for prefix in (4096, 16384, 50000, 120000):
    for chunk in (512, 1024, 2048):
        tot = prefix + chunk
        k = torch.randn(tot, 1, HK, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        q = torch.randn(chunk, HQ, D, device=dev, dtype=torch.bfloat16)
        pt = torch.arange(tot, device=dev, dtype=torch.int32)[None]
        sl = torch.tensor([tot], device=dev, dtype=torch.int32)
        cu = torch.tensor([0, chunk], device=dev, dtype=torch.int32)
        row = []
        ref = None
        for ns, pg in ((0, None), (1, None), (1, True), (1, False), (2, None), (3, None), (4, None), (6, None)):
            f = lambda: flash_attn_with_kvcache(q=q, k_cache=k, v_cache=v, page_table=pt, cache_seqlens=sl,
                                                cu_seqlens_q=cu, max_seqlen_q=chunk, causal=True,
                                                softmax_scale=D ** -0.5, num_splits=ns, pack_gqa=pg)
            o = f()
            o = o[0] if isinstance(o, tuple) else o
            if ref is None:
                ref = o.float()
            err = (o.float() - ref).abs().max().item()
            row.append(f"ns={ns}{'' if pg is None else ('/pk' if pg else '/nopk')}:{bench(f):7.1f}us(e{err:.0e})")
        print(f"prefix={prefix:6d} chunk={chunk:4d}  " + "  ".join(row), flush=True)
        del k, v
