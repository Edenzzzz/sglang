"""Decompose one mixed attention step (Qwen2.5-7B heads 28 / 4, d128, bf16, page 1) into its parts, to explain why the
serving ablation lost 34% throughput without the split while fusion_microbench_20261021.py measured 0-6% per step:
  chunk-FA3     the prefill chunk alone on FA3 (sgl-kernel), as in the split
  chunk-FA2     the same chunk alone through flashinfer FA2 BatchPrefill, as in the unfused step
  rows-FA2      the decode / verify rows alone through FA2 BatchPrefill (default plan: split-KV allowed)
  rows-FA2-nosplit  the same rows with split-KV disabled, as when they share one plan with a long prefill chunk
  merged        everything in one FA2 BatchPrefill plan (the unfused step)
All arms against an fp32 causal reference on a sampled slice.
  python fusion_decompose_20261021.py
Claude, 2026-10-21.
"""
import torch
import flashinfer
from sgl_kernel.flash_attn import flash_attn_with_kvcache

dev, D, HQ, HK = "cuda", 128, 28, 4
ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)


def make(chunk, prefix, rows, qrow, row_kv, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    kv_lens = [prefix + chunk] + row_kv
    q_lens = [chunk] + [qrow] * rows
    n = sum(kv_lens)
    k = torch.randn(n + 64, 1, HK, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    v = torch.randn(n + 64, 1, HK, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    perm = torch.randperm(n + 64, device=dev, generator=g)[:n].to(torch.int32)
    pages, off = [], 0
    for L in kv_lens:
        pages.append(perm[off:off + L]); off += L
    q = torch.randn(sum(q_lens), HQ, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    return dict(k=k, v=v, pages=pages, q=q, q_lens=q_lens, kv_lens=kv_lens)


def fa2(b, idx, disable_split_kv=False):
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    qo, kvp, ind = [0], [0], []
    for i in idx:
        qo.append(qo[-1] + b["q_lens"][i]); kvp.append(kvp[-1] + b["kv_lens"][i]); ind.append(b["pages"][i])
    kw = dict(causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    try:
        w.plan(torch.tensor(qo, dtype=torch.int32, device=dev), torch.tensor(kvp, dtype=torch.int32, device=dev),
               torch.cat(ind), torch.ones(len(idx), dtype=torch.int32, device=dev), HQ, HK, D, 1,
               disable_split_kv=disable_split_kv, **kw)
    except TypeError:
        if disable_split_kv:
            return None
        w.plan(torch.tensor(qo, dtype=torch.int32, device=dev), torch.tensor(kvp, dtype=torch.int32, device=dev),
               torch.cat(ind), torch.ones(len(idx), dtype=torch.int32, device=dev), HQ, HK, D, 1, **kw)
    qs = torch.cat([b["q"][sum(b["q_lens"][:i]):sum(b["q_lens"][:i + 1])] for i in idx])
    return lambda: w.run(qs, (b["k"], b["v"]))


def fa3_chunk(b):
    c, n = b["q_lens"][0], b["kv_lens"][0]
    return lambda: flash_attn_with_kvcache(q=b["q"][:c], k_cache=b["k"], v_cache=b["v"],
                                           page_table=b["pages"][0][None, :],
                                           cache_seqlens=torch.tensor([n], dtype=torch.int32, device=dev),
                                           cu_seqlens_q=torch.tensor([0, c], dtype=torch.int32, device=dev),
                                           max_seqlen_q=c, causal=True)


def ref_err(b, out, i, q_off):
    ql, n = b["q_lens"][i], b["kv_lens"][i]
    rows = torch.arange(max(0, ql - 64), ql, device=dev)
    kk = b["k"][b["pages"][i].long(), 0].float(); vv = b["v"][b["pages"][i].long(), 0].float()
    err = 0.0
    for h in (0, HQ - 1):
        s = b["q"][sum(b["q_lens"][:i]) + rows, h].float() @ kk[:, h // (HQ // HK)].T / D ** 0.5
        s.masked_fill_(torch.arange(n, device=dev)[None, :] > (n - ql + rows)[:, None], float("-inf"))
        r = torch.softmax(s, -1) @ vv[:, h // (HQ // HK)]
        o = out[q_off + rows, h].float()
        err = max(err, ((o - r).abs().max() / r.abs().max()).item())
    return err


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


g = torch.Generator().manual_seed(1)
mix = [int(x) for x in torch.randint(4096, 65536, (64,), generator=g)]
print("ms per layer, Qwen2.5-7B heads")
for chunk, prefix in ((1024, 16384), (1024, 49152), (2048, 32768), (2048, 61440)):
    for rows, qrow in ((16, 4), (48, 4), (48, 1)):
        b = make(chunk, prefix, rows, qrow, mix[:rows])
        n = 1 + rows
        f3, f2c, f2r, mg = fa3_chunk(b), fa2(b, [0]), fa2(b, range(1, n)), fa2(b, range(n))
        f2r_ns = fa2(b, range(1, n), disable_split_kv=True)
        o3 = f3(); o3 = o3[0] if isinstance(o3, tuple) else o3
        errs = [ref_err(b, o3, 0, 0), ref_err(b, f2c(), 0, 0), ref_err(b, f2r(), 1, 0), ref_err(b, mg(), 0, 0)]
        line = (f"{chunk:5d}@{prefix:6d} + {rows:2d}x q{qrow}: chunk FA3 {t(f3):6.3f}  chunk FA2 {t(f2c):6.3f}  "
                f"rows FA2 {t(f2r):6.3f}")
        if f2r_ns is not None:
            line += f"  rows FA2 no-split {t(f2r_ns):6.3f}"
        line += f"  merged {t(mg):6.3f}  max err {max(errs):.1e} {'OK' if max(errs) < 2e-2 else 'FAIL'}"
        print(line, flush=True)
