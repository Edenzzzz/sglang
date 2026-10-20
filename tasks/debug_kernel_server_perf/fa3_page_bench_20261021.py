"""FA3 (sgl-kernel flash_attn_with_kvcache) on the serving shape that sets chat TPOT at chat 8 req/s: one agent chunk of
C new tokens over a P-token prefix, Qwen2.5-7B heads (28 q / 4 kv, d128), causal, bf16, KV paged at page size 1 (what
serving runs), 128 and 256. Reports ms per layer and TFLOP/s (causal FLOPs of the chunk over prefix + itself) and checks
every arm against an fp32 causal reference on a sampled head slice.
  python fa3_page_bench_20261021.py
Claude, 2026-10-21.
"""
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

torch.manual_seed(0)
HQ, HK, D = 28, 4, 128
dev = "cuda"


def flops(c, p):
    # QK^T and PV over the prefix (full) and the chunk's own causal triangle.
    return 4 * HQ * D * c * (p + (c + 1) / 2)


def bench(c, p, page):
    n = p + c
    npages = (n + page - 1) // page
    pool_pages = npages + 7
    k_pool = torch.randn(pool_pages, page, HK, D, device=dev, dtype=torch.bfloat16) * 0.5
    v_pool = torch.randn(pool_pages, page, HK, D, device=dev, dtype=torch.bfloat16) * 0.5
    perm = torch.randperm(pool_pages, device=dev)[:npages].to(torch.int32)  # scattered pages, as in a live pool
    page_table = perm[None, :]
    q = torch.randn(c, HQ, D, device=dev, dtype=torch.bfloat16) * 0.5
    cu_q = torch.tensor([0, c], device=dev, dtype=torch.int32)
    seqlens = torch.tensor([n], device=dev, dtype=torch.int32)

    def run():
        return flash_attn_with_kvcache(q=q, k_cache=k_pool, v_cache=v_pool, page_table=page_table,
                                       cache_seqlens=seqlens, cu_seqlens_q=cu_q, max_seqlen_q=c, causal=True)
    out = run()
    if isinstance(out, tuple):
        out = out[0]
    # fp32 causal reference on heads 0 and 27 (kv heads 0 and 3), last 256 query rows
    k = k_pool[perm.long()].reshape(-1, HK, D)[:n].float()
    v = v_pool[perm.long()].reshape(-1, HK, D)[:n].float()
    err = 0.0
    rows = torch.arange(c - 256, c, device=dev)
    for h in (0, HQ - 1):
        kh = h // (HQ // HK)
        s = (q[rows, h].float() @ k[:, kh].T) / D ** 0.5
        pos = p + rows
        s.masked_fill_(torch.arange(n, device=dev)[None, :] > pos[:, None], float("-inf"))
        ref = torch.softmax(s, -1) @ v[:, kh]
        err = max(err, ((out[rows, h].float() - ref).abs().max() / ref.abs().max()).item())
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    it = 20
    for _ in range(it):
        run()
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / it
    return ms, flops(c, p) / ms / 1e9, err


for c, p in ((2048, 65536), (1024, 65536), (2048, 32768), (512, 49152)):
    for page in (1, 128, 256):
        ms, tf, err = bench(c, p, page)
        print(f"chunk {c:5d} @ {p:6d}  page {page:3d}: {ms:7.3f} ms/layer  {tf:6.1f} TFLOP/s  "
              f"rel err {err:.2e} {'OK' if err < 2e-2 else 'FAIL'}", flush=True)
