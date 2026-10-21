"""The co-run on serving-shaped mixed steps (H200, 2026-10-21). The per-step SM partition (apply_split_auto_margin.py)
lost 9% throughput in serving (125 vs 137 tok/s, `_ct_r3d3am_rep1`) although rows_layout_check_20261021.py found the
co-run 4-25% faster than one stock FA3 call at the best partition. Its shapes were 16-48 rows all at 4-64K KV; serving's
logged steps are a deep deadline chunk (predicted 7-8 ms per layer) beside rows worth ~0.2 ms. So here, Qwen2.5-7B heads,
causal, page 1, every arm against an fp32 causal reference:
  steps   chunk C @ prefix P + A agent rows (q4 or q1, KV 32-64K) + S chat rows (same q, KV 0.5-4K)
  arms    stock-FA3 (one varlen call), split (chunk then rows), co-run m=0 (two streams, all SMs each), co-run m (chunk
          sm_margin m, rows sm_margin num_SMs - m) for m in 8..112
  python serving_mix_corun_20261021.py
Claude, 2026-10-21.
"""
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

dev, D, HQ, HK = "cuda", 128, 28, 4
NSM = torch.cuda.get_device_properties(0).multi_processor_count


def t(fn, it=20):
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


def build(kv, ql, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    n = sum(kv)
    base, sl = 0, []
    for L in kv:
        sl.append(torch.arange(base, base + L, device=dev)); base += L + 7
    sl = torch.cat(sl).to(torch.int32)
    pool = int(sl.max()) + 1
    k = torch.randn(pool, 1, HK, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    v = torch.randn(pool, 1, HK, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    q = torch.randn(sum(ql), HQ, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    q0 = [0]
    for l in ql:
        q0.append(q0[-1] + l)
    off = [0]
    for L in kv:
        off.append(off[-1] + L)
    return dict(kv=kv, ql=ql, sl=sl, k=k, v=v, q=q, q0=q0, off=off)


def call(b, idx, sm_margin=0):
    kv, ql = b["kv"], b["ql"]
    pt = torch.zeros(len(idx), max(kv[i] for i in idx), dtype=torch.int32, device=dev)
    for j, i in enumerate(idx):
        pt[j, :kv[i]] = b["sl"][b["off"][i]:b["off"][i + 1]]
    qq = torch.cat([b["q"][b["q0"][i]:b["q0"][i + 1]] for i in idx])
    cu = torch.tensor([0] + list(torch.tensor([ql[i] for i in idx]).cumsum(0)), dtype=torch.int32, device=dev)
    kvl = torch.tensor([kv[i] for i in idx], dtype=torch.int32, device=dev)
    mq = max(ql[i] for i in idx)
    return lambda: flash_attn_with_kvcache(q=qq, k_cache=b["k"], v_cache=b["v"], page_table=pt, cache_seqlens=kvl,
                                           cu_seqlens_q=cu, max_seqlen_q=mq, causal=True, sm_margin=sm_margin)


def err(b, o, idx):
    """Max rel err vs fp32 for request idx[0] (last 64 rows) and idx[1]; o is the concatenated output over idx."""
    e, off = 0.0, 0
    for j, i in enumerate(idx):
        l, n = b["ql"][i], b["kv"][i]
        if j < 2:
            kk = b["k"][b["sl"][b["off"][i]:b["off"][i + 1]].long(), 0].float()
            vv = b["v"][b["sl"][b["off"][i]:b["off"][i + 1]].long(), 0].float()
            rr = torch.arange(max(0, l - 64), l, device=dev)
            for h in (0, HQ - 1):
                s = b["q"][b["q0"][i] + rr, h].float() @ kk[:, h // 7].T / D ** 0.5
                s.masked_fill_(torch.arange(n, device=dev)[None, :] > (n - l + rr)[:, None], float("-inf"))
                r = torch.softmax(s, -1) @ vv[:, h // 7]
                e = max(e, ((o[off + rr, h].float() - r).abs().max() / r.abs().max()).item())
        off += l
    return e


g0 = torch.Generator().manual_seed(3)
agent_kv = [int(x) for x in torch.randint(32768, 65536, (16,), generator=g0)]
chat_kv = [int(x) for x in torch.randint(512, 4096, (64,), generator=g0)]
side = torch.cuda.Stream()
print(f"GPU {torch.cuda.get_device_name(0)}, {NSM} SMs; ms per layer, Qwen2.5-7B heads")
for C, P in ((6000, 49152), (4096, 32768), (2048, 32768), (1024, 16384), (512, 4096)):
    for A, S, qr in ((4, 16, 4), (8, 32, 4), (8, 32, 1), (2, 8, 4)):
        kv = [P + C] + agent_kv[:A] + chat_kv[:S]
        ql = [C] + [qr] * (A + S)
        b = build(kv, ql)
        n = len(kv)
        allr, rows = list(range(n)), list(range(1, n))
        stock, pre, rw = call(b, allr), call(b, [0]), call(b, rows)
        e = err(b, first(stock()), allr)
        e = max(e, err(b, torch.cat([first(pre()), first(rw())]), allr))
        ts = t(stock)
        tsp = t(lambda: (pre(), rw()))
        res = {}
        for m in (0, 8, 16, 32, 48, 64, 80, 96, 112):
            pm, rm = call(b, [0], m), call(b, rows, NSM - m if m else 0)

            def co(pm=pm, rm=rm):
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    rm()
                pm()
                torch.cuda.current_stream().wait_stream(side)
            res[m] = t(co)
        best = min((v, m) for m, v in res.items() if m)
        print(f"{C:5d}@{P:6d} + {A} agent + {S:2d} chat x q{qr}: stock-FA3 {ts:6.3f}  split {tsp:6.3f} "
              f"({(tsp / ts - 1) * 100:+5.1f}%)  co-run m0 {res[0]:6.3f} ({(res[0] / ts - 1) * 100:+5.1f}%)  "
              f"m8 {res[8]:6.3f} ({(res[8] / ts - 1) * 100:+5.1f}%)  best m{best[1]} {best[0]:6.3f} "
              f"({(best[0] / ts - 1) * 100:+5.1f}%)  err {e:.1e} {'OK' if e < 2e-2 else 'FAIL'}", flush=True)
