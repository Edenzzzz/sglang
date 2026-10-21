"""Does the split's FA3 page table hold for page_size > 1? A req_to_token-shaped row (token -> KV slot, pages of P
scattered in a pool) turned into FA3's page table two ways:
  per-token   `row[:max_k] // P`        (the split's construction before apply_split_rows_fa3.py)
  per-page    `row[:max_k:P] // P`      (upstream flashattention_backend's, and the patch's)
each run through sgl-kernel flash_attn_with_kvcache on a 1024-token chunk over a 15K prefix plus 8 q4 verify rows
(Qwen2.5-7B heads), against an fp32 causal reference over the true K/V. Pages 1 and 128.
  python split_page_table_check_20261021.py
Claude, 2026-10-21.
"""
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

dev, D, HQ, HK = "cuda", 128, 28, 4
torch.manual_seed(0)


def run(P):
    kv = [16384] + [9000 + 517 * i for i in range(8)]
    ql = [1024] + [4] * 8
    npg = [(n + P - 1) // P for n in kv]
    pool_pages = sum(npg) + 11
    kc = torch.randn(pool_pages, P, HK, D, device=dev, dtype=torch.bfloat16) * 0.5
    vc = torch.randn(pool_pages, P, HK, D, device=dev, dtype=torch.bfloat16) * 0.5
    perm = torch.randperm(pool_pages, device=dev)
    cols = max(npg) * P
    r2t = torch.zeros(len(kv), cols, dtype=torch.int32, device=dev)
    off = 0
    for i, n in enumerate(kv):
        pg = perm[off:off + npg[i]]
        off += npg[i]
        slots = (pg[:, None] * P + torch.arange(P, device=dev)[None, :]).reshape(-1)[:n]
        r2t[i, :n] = slots.to(torch.int32)
    q = torch.randn(sum(ql), HQ, D, device=dev, dtype=torch.bfloat16) * 0.5
    cu = torch.tensor([0] + list(torch.tensor(ql).cumsum(0)), dtype=torch.int32, device=dev)
    kvl = torch.tensor(kv, dtype=torch.int32, device=dev)
    kflat, vflat = kc.view(-1, HK, D), vc.view(-1, HK, D)
    res = {}
    for name, pt in (("per-token", r2t[:, :max(kv)] // P), ("per-page", r2t[:, :max(kv):P] // P)):
        o = flash_attn_with_kvcache(q=q, k_cache=kc, v_cache=vc, page_table=pt.contiguous(), cache_seqlens=kvl,
                                    cu_seqlens_q=cu, max_seqlen_q=max(ql), causal=True)
        o = o[0] if isinstance(o, tuple) else o
        err = 0.0
        for i in (0, 1, 8):
            n, l, q0 = kv[i], ql[i], int(cu[i])
            rows = torch.arange(max(0, l - 64), l, device=dev)
            kk, vv = kflat[r2t[i, :n].long()].float(), vflat[r2t[i, :n].long()].float()
            for h in (0, HQ - 1):
                s = q[q0 + rows, h].float() @ kk[:, h // 7].T / D ** 0.5
                s.masked_fill_(torch.arange(n, device=dev)[None, :] > (n - l + rows)[:, None], float("-inf"))
                r = torch.softmax(s, -1) @ vv[:, h // 7]
                err = max(err, ((o[q0 + rows, h].float() - r).abs().max() / r.abs().max()).item())
        res[name] = err
    print(f"page {P:3d}: " + "  ".join(f"{k} rel err {v:.2e} {'OK' if v < 2e-2 else 'FAIL'}" for k, v in res.items()),
          flush=True)


for P in (1, 128):
    run(P)
