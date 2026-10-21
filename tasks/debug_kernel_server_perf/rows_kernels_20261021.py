"""Decode / verify rows of a mixed step on decode-shaped kernels (Qwen2.5-7B heads 28 / 4, d128, bf16, page 1): the
decomposition (fusion_decompose_20261021.py) found these rows, not the prefill chunk, set the step -- 48 rows at 4-64K
KV take ~8 ms per layer through FA2 BatchPrefill, ~10x their KV-read time. Arms per row set:
  prefill-FA2   flashinfer BatchPrefill (what the unfused step and the benchmark's split ran)
  decode-FA2    flashinfer BatchDecode with tensor cores (q = 1 rows only)
  persistent-FA2  flashinfer BatchAttention (the persistent FA2-lineage kernel the serving split routes rows to)
  FA3           sgl-kernel flash_attn_with_kvcache over the rows (varlen q, packed GQA), q = 1 and q = 4
Each against an fp32 causal reference; also the KV-read floor at 4.8 TB/s.
  python rows_kernels_20261021.py
Claude, 2026-10-21.
"""
import torch
import flashinfer
from sgl_kernel.flash_attn import flash_attn_with_kvcache

dev, D, HQ, HK = "cuda", 128, 28, 4
ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)


def t(fn, it=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(it):
        fn()
    e.record(); torch.cuda.synchronize()
    return a.elapsed_time(e) / it


def case(rows, qrow, kv):
    g = torch.Generator(device=dev).manual_seed(0)
    n = sum(kv)
    k = torch.randn(n + 64, 1, HK, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    v = torch.randn(n + 64, 1, HK, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    perm = torch.randperm(n + 64, device=dev, generator=g)[:n].to(torch.int32)
    q = torch.randn(rows * qrow, HQ, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    kv_ind = torch.tensor([0] + list(torch.tensor(kv).cumsum(0)), dtype=torch.int32, device=dev)
    qo_ind = torch.arange(0, rows * qrow + 1, qrow, dtype=torch.int32, device=dev)
    last = torch.ones(rows, dtype=torch.int32, device=dev)
    kvl = torch.tensor(kv, dtype=torch.int32, device=dev)
    out = {}
    wp = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    wp.plan(qo_ind, kv_ind, perm, last, HQ, HK, D, 1, causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    out["prefill-FA2"] = (lambda: wp.run(q, (k, v)))
    if qrow == 1:
        wd = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)
        wd.plan(kv_ind, perm, last, HQ, HK, D, 1, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        out["decode-FA2"] = (lambda: wd.run(q, (k, v)))
    wa = flashinfer.BatchAttention("NHD")
    wa.plan(qo_ind, kv_ind, perm, kvl, HQ, HK, D, D, 1, causal=True, q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16)
    out["persistent-FA2"] = (lambda: wa.run(q, (k, v)))
    maxkv = max(kv)
    pt = torch.zeros(rows, maxkv, dtype=torch.int32, device=dev)
    for i in range(rows):
        pt[i, :kv[i]] = perm[kv_ind[i]:kv_ind[i + 1]]
    out["FA3"] = (lambda: flash_attn_with_kvcache(q=q, k_cache=k, v_cache=v, page_table=pt, cache_seqlens=kvl,
                                                  cu_seqlens_q=qo_ind, max_seqlen_q=qrow, causal=True))
    # fp32 reference for row 0, heads 0 and 27
    kk = k[perm[:kv[0]].long(), 0].float(); vv = v[perm[:kv[0]].long(), 0].float()
    res = []
    for name, fn in out.items():
        o = fn(); o = o[0] if isinstance(o, tuple) else o
        err = 0.0
        for h in (0, HQ - 1):
            s = q[:qrow, h].float() @ kk[:, h // 7].T / D ** 0.5
            pos = kv[0] - qrow + torch.arange(qrow, device=dev)
            s.masked_fill_(torch.arange(kv[0], device=dev)[None, :] > pos[:, None], float("-inf"))
            r = torch.softmax(s, -1) @ vv[:, h // 7]
            err = max(err, ((o[:qrow, h].float() - r).abs().max() / r.abs().max()).item())
        res.append(f"{name} {t(fn):6.3f} ({'OK' if err < 2e-2 else 'FAIL'} {err:.0e})")
    floor = n * HK * D * 2 * 2 / 4.8e12 * 1e3
    print(f"{rows:3d} rows x q{qrow}, KV mean {n // rows:6d}: " + "  ".join(res) + f"  | KV-read floor {floor:.3f} ms",
          flush=True)


g = torch.Generator().manual_seed(1)
mix = [int(x) for x in torch.randint(4096, 65536, (64,), generator=g)]
print("ms per layer, Qwen2.5-7B heads")
for rows in (16, 48):
    for qrow in (1, 4):
        case(rows, qrow, mix[:rows])
