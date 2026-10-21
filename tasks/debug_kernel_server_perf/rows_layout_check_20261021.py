"""Two checks on the kernel microbench before its numbers carry the paper's first stage (H200, 2026-10-21):
 1. KV layout. fusion_microbench / rows_kernels scatter EVERY KV slot at random across the pool; a serving pool hands a
    request mostly contiguous slots. Rows arms (FA2 BatchPrefill, FA2 BatchDecode, FA3) at 48 rows x q1 / q4, KV 4-64K,
    with slots scattered per token, contiguous per request, and in 16-token runs.
 2. Stock sglang's default on Hopper is the FA3 backend (get_default_attn_backend: MHA, no spec or topk 1), not
    flashinfer. Its mixed step is ONE FA3 call over chunk + rows (varlen). Arm "stock-FA3" vs our split (FA3 chunk then
    FA3 rows) and the co-run (chunk with sm_margin m, rows on the m SMs left; m = 8..112), Qwen2.5-7B heads.
Every arm against an fp32 causal reference (row 0 heads 0 / 27; for check 2 also the chunk's last 64 rows).
  python rows_layout_check_20261021.py
Claude, 2026-10-21.
"""
import torch
import flashinfer
from sgl_kernel.flash_attn import flash_attn_with_kvcache

dev, D, HQ, HK = "cuda", 128, 28, 4
NSM = torch.cuda.get_device_properties(0).multi_processor_count
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


def first(o):
    return o[0] if isinstance(o, tuple) else o


def slots(kv, layout, g):
    n = sum(kv)
    pool = n + 4096
    if layout == "scatter":
        return torch.randperm(pool, device=dev, generator=g)[:n].to(torch.int32)
    if layout == "contig":
        base, out = 0, []
        for L in kv:
            out.append(torch.arange(base, base + L, device=dev)); base += L + 7
        return torch.cat(out).to(torch.int32)
    # 16-token runs at random 16-aligned places (a fragmented pool)
    nb = (pool + 15) // 16
    blocks = torch.randperm(nb, device=dev, generator=g)
    out, b = [], 0
    for L in kv:
        k = (L + 15) // 16
        s = (blocks[b:b + k, None] * 16 + torch.arange(16, device=dev)[None, :]).reshape(-1)[:L]
        out.append(s); b += k
    return torch.cat(out).to(torch.int32)


def errs(o, q, k, v, sl, kv, ql, q0s, rows_of):
    err = 0.0
    off = 0
    for i, n in enumerate(kv):
        if i in rows_of:
            kk, vv = k[sl[off:off + n].long(), 0].float(), v[sl[off:off + n].long(), 0].float()
            l = ql[i]
            rr = torch.arange(max(0, l - 64), l, device=dev)
            for h in (0, HQ - 1):
                s = q[q0s[i] + rr, h].float() @ kk[:, h // 7].T / D ** 0.5
                s.masked_fill_(torch.arange(n, device=dev)[None, :] > (n - l + rr)[:, None], float("-inf"))
                r = torch.softmax(s, -1) @ vv[:, h // 7]
                err = max(err, ((o[q0s[i] + rr, h].float() - r).abs().max() / r.abs().max()).item())
        off += n
    return err


def setup(kv, ql, layout, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    sl = slots(kv, layout, g)
    pool = int(sl.max()) + 1
    k = torch.randn(pool, 1, HK, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    v = torch.randn(pool, 1, HK, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    q = torch.randn(sum(ql), HQ, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    q0s = [0]
    for l in ql:
        q0s.append(q0s[-1] + l)
    return sl, k, v, q, q0s


def fa3_call(q, k, v, sl, kv, ql, idx, q0s, sm_margin=0):
    offs = [0]
    for n in kv:
        offs.append(offs[-1] + n)
    mk = max(kv[i] for i in idx)
    pt = torch.zeros(len(idx), mk, dtype=torch.int32, device=dev)
    for j, i in enumerate(idx):
        pt[j, :kv[i]] = sl[offs[i]:offs[i + 1]]
    qq = torch.cat([q[q0s[i]:q0s[i + 1]] for i in idx])
    cu = torch.tensor([0] + list(torch.tensor([ql[i] for i in idx]).cumsum(0)), dtype=torch.int32, device=dev)
    kvl = torch.tensor([kv[i] for i in idx], dtype=torch.int32, device=dev)
    mq = max(ql[i] for i in idx)
    return lambda: flash_attn_with_kvcache(q=qq, k_cache=k, v_cache=v, page_table=pt, cache_seqlens=kvl,
                                           cu_seqlens_q=cu, max_seqlen_q=mq, causal=True, sm_margin=sm_margin)


g0 = torch.Generator().manual_seed(1)
mix = [int(x) for x in torch.randint(4096, 65536, (64,), generator=g0)]
print(f"GPU {torch.cuda.get_device_name(0)}, {NSM} SMs; ms per layer, Qwen2.5-7B heads")
print("== 1. rows alone, by KV layout")
for qrow in (1, 4):
    for layout in ("scatter", "run16", "contig"):
        kv, ql = mix[:48], [qrow] * 48
        sl, k, v, q, q0s = setup(kv, ql, layout)
        kvi = torch.tensor([0] + list(torch.tensor(kv).cumsum(0)), dtype=torch.int32, device=dev)
        qoi = torch.tensor(q0s, dtype=torch.int32, device=dev)
        last = torch.ones(48, dtype=torch.int32, device=dev)
        arms = {}
        wp = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
        wp.plan(qoi, kvi, sl, last, HQ, HK, D, 1, causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        arms["FA2-prefill"] = lambda: wp.run(q, (k, v))
        if qrow == 1:
            wd = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)
            wd.plan(kvi, sl, last, HQ, HK, D, 1, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
            arms["FA2-decode"] = lambda: wd.run(q, (k, v))
        arms["FA3"] = fa3_call(q, k, v, sl, kv, ql, range(48), q0s)
        res = []
        for name, fn in arms.items():
            e = errs(first(fn()), q, k, v, sl, kv, ql, q0s, {0, 47})
            res.append(f"{name} {t(fn):6.3f} ({'OK' if e < 2e-2 else 'FAIL'} {e:.0e})")
        print(f"48 x q{qrow} {layout:7s}: " + "  ".join(res), flush=True)

print("== 2. mixed step vs stock sglang's Hopper default (FA3 backend: one varlen FA3 call over chunk + rows)")
side = torch.cuda.Stream()
for layout in ("contig", "scatter"):
    for chunk, prefix in ((1024, 16384), (1024, 49152), (2048, 32768)):
        for rows, qrow in ((16, 4), (48, 4), (48, 1)):
            kv, ql = [prefix + chunk] + mix[:rows], [chunk] + [qrow] * rows
            sl, k, v, q, q0s = setup(kv, ql, layout)
            n = 1 + rows
            stock = fa3_call(q, k, v, sl, kv, ql, range(n), q0s)
            pre, rws = fa3_call(q, k, v, sl, kv, ql, [0], q0s), fa3_call(q, k, v, sl, kv, ql, range(1, n), q0s)
            e = errs(first(stock()), q, k, v, sl, kv, ql, q0s, {0, 1})
            o_pre, o_rows = first(pre()), first(rws())
            e = max(e, errs(torch.cat([o_pre, o_rows]), q, k, v, sl, kv, ql, q0s, {0, 1}))
            split = lambda: (pre(), rws())
            best = None
            for m in (8, 16, 24, 32, 48, 64, 80, 96, 112):
                pm, rm = fa3_call(q, k, v, sl, kv, ql, [0], q0s, m), fa3_call(q, k, v, sl, kv, ql, range(1, n), q0s, NSM - m)

                def co(pm=pm, rm=rm):
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        rm()
                    pm()
                    torch.cuda.current_stream().wait_stream(side)
                tm = t(co)
                if best is None or tm < best[0]:
                    best = (tm, m)
            ts, tsp = t(stock), t(split)
            print(f"{layout:7s} {chunk:5d}@{prefix:6d} + {rows:2d}x q{qrow}: stock-FA3 {ts:6.3f}  split {tsp:6.3f} "
                  f"({(tsp / ts - 1) * 100:+5.1f}%)  co-run {best[0]:6.3f} ({(best[0] / ts - 1) * 100:+5.1f}%, m{best[1]})  "
                  f"err {e:.1e} {'OK' if e < 2e-2 else 'FAIL'}", flush=True)
