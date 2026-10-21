"""Does FA3 care about the page table's ROW STRIDE? (H200, 2026-10-22.)

Serving finding that prompted this: with the split disabled so every extend step is one varlen FA3 call -- the same
kernel stock's FA3 backend runs -- our mixed prefill steps still cost +43% in the shallow bin (ptok < 600, prefix <
16K: 10.67 vs 7.48 ms per step) and ours' agent throughput stays 9% below stock's
(h200_nosplit_probe_20261021.sh, SPLIT_POLICY split=0 one_call=2001).

The remaining difference is how the page table is built. Upstream's FA3 backend gathers
`req_to_token[req_pool_indices, :max_len]`, which is a CONTIGUOUS [rows, max_len] tensor. Ours index_selects whole
req_to_token rows into a persistent [pool_rows, max_context] buffer and passes `buf[:, :max_k]` -- same values, but
row stride = max_context (131072 ints = 512 KB) instead of max_k.

Arms, identical values and identical cache_seqlens, Qwen2.5-7B heads, causal, page 1:
  narrow     page_table contiguous, row stride = max_k          (upstream's shape)
  wide       a [rows, 131072] buffer sliced to [:, :max_k]      (ours; stride 131072)
  wide-cont  the same slice made contiguous again               (the candidate fix: one small copy per step)
Each against an fp32 causal reference on sampled heads / rows.
  python page_table_stride_20261022.py
Claude, 2026-10-22.
"""
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

dev, D, HQ, HK = "cuda", 128, 28, 4
CTX = 131072  # req_to_token's column count in the serving runs (--context-length)


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


def first(o):
    return o[0] if isinstance(o, tuple) else o


def case(chunk, prefix, rows, qrow, row_kv, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    kv = [prefix + chunk] + row_kv
    ql = [chunk] + [qrow] * rows
    n = sum(kv)
    pool = n + 4096
    k = torch.randn(pool, 1, HK, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    v = torch.randn(pool, 1, HK, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    slots = torch.randperm(pool, device=dev, generator=g)[:n].to(torch.int32)
    q = torch.randn(sum(ql), HQ, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    q0, off = [0], [0]
    for l in ql:
        q0.append(q0[-1] + l)
    for L in kv:
        off.append(off[-1] + L)
    max_k = max(kv)
    nrow = len(kv)
    # narrow: contiguous [rows, max_k]
    narrow = torch.zeros(nrow, max_k, dtype=torch.int32, device=dev)
    for i in range(nrow):
        narrow[i, :kv[i]] = slots[off[i]:off[i + 1]]
    # wide: the same values inside a [rows, CTX] buffer, passed as a stride-CTX view
    buf = torch.zeros(nrow, CTX, dtype=torch.int32, device=dev)
    buf[:, :max_k] = narrow
    wide = buf[:, :max_k]
    cu = torch.tensor(q0, dtype=torch.int32, device=dev)
    kvl = torch.tensor(kv, dtype=torch.int32, device=dev)
    mq = max(ql)

    def run(pt):
        return flash_attn_with_kvcache(q=q, k_cache=k, v_cache=v, page_table=pt, cache_seqlens=kvl,
                                       cu_seqlens_q=cu, max_seqlen_q=mq, causal=True)

    arms = {"narrow": narrow, "wide": wide, "wide-cont": wide.contiguous()}
    # fp32 reference on request 0 (last 64 rows) and request 1, heads 0 and 27
    res, err = {}, 0.0
    for name, pt in arms.items():
        o = first(run(pt))
        for i in (0, 1):
            L, N = ql[i], kv[i]
            kk = k[slots[off[i]:off[i + 1]].long(), 0].float()
            vv = v[slots[off[i]:off[i + 1]].long(), 0].float()
            rr = torch.arange(max(0, L - 64), L, device=dev)
            for h in (0, HQ - 1):
                sc = q[q0[i] + rr, h].float() @ kk[:, h // 7].T / D ** 0.5
                sc.masked_fill_(torch.arange(N, device=dev)[None, :] > (N - L + rr)[:, None], float("-inf"))
                r = torch.softmax(sc, -1) @ vv[:, h // 7]
                err = max(err, ((o[q0[i] + rr, h].float() - r).abs().max() / r.abs().max()).item())
        res[name] = t(lambda pt=pt: run(pt))
    base = res["narrow"]
    extra = "  ".join(f"{k2} {v2:6.3f} ({(v2 / base - 1) * 100:+5.1f}%)" for k2, v2 in res.items() if k2 != "narrow")
    print(f"{chunk:5d}@{prefix:6d} + {rows:2d}x q{qrow} (max_k {max_k:6d}, stride ratio {CTX / max_k:5.1f}x): "
          f"narrow {base:6.3f}  {extra}  err {err:.1e} {'OK' if err < 2e-2 else 'FAIL'}", flush=True)


g0 = torch.Generator().manual_seed(3)
agent_kv = [int(x) for x in torch.randint(32768, 65536, (16,), generator=g0)]
chat_kv = [int(x) for x in torch.randint(512, 4096, (64,), generator=g0)]
print(f"GPU {torch.cuda.get_device_name(0)}; req_to_token width {CTX}; ms per layer, Qwen2.5-7B heads")
print("# the serving shallow bin: a small chunk over a short prefix beside chat decode rows")
for chunk, prefix in ((512, 4096), (512, 16384), (1024, 16384)):
    for rows in (8, 32):
        case(chunk, prefix, rows, 4, chat_kv[:rows])
print("# deeper steps, where the stride ratio is smaller")
for chunk, prefix in ((2048, 32768), (2048, 61440)):
    for rows in (8, 32):
        case(chunk, prefix, rows, 4, agent_kv[:8] + chat_kv[:max(0, rows - 8)])
