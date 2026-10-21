"""Kernel microbenchmark for the paper's first stage: one mixed attention step, unfused vs our split, on serving shapes.

  unfused   stock mixed chunk: every row (prefill chunk + decode / verify rows) through one flashinfer FA2 BatchPrefill
  split     ours, sequential: prefill rows on FA3 (sgl-kernel flash_attn_with_kvcache, all SMs), then the decode /
            verify rows on FA2 BatchPrefill
  co-run    ours, overlapped: FA3 with sm_margin SMs left free, the FA2 rows on a second stream at the same time
  rows-FA3  ours with SGLANG_SPLIT_ROWS_FA3=1 (apply_split_rows_fa3.py): prefill rows on FA3, then the decode / verify
            rows on a second FA3 call (varlen q, GQA packed by FA3's heuristic); and its overlapped form, the rows
            taking the margin SMs (sm_margin = num_SMs - margin)

Two batch kinds, per model (per-GPU heads; head dim 128, bf16, KV paged at page 1 as in serving, scattered pages):
  2-mode    a prefill chunk C over prefix P + R decode rows (1 query token each) at KV lengths drawn from the mix
  4-mode    the same chunk + R MTP verify rows (4 query tokens each, the 4-token verify of a 3-step EAGLE3 draft)
Every arm is checked against an fp32 causal reference (sampled heads / rows of the prefill and of one short row).
  python fusion_microbench_20261021.py
Claude, 2026-10-21.
"""
import torch
import flashinfer
from sgl_kernel.flash_attn import flash_attn_with_kvcache

dev = "cuda"
D = 128
MODELS = {"qwen2.5-7b": (28, 4), "qwen3-8b": (32, 8), "glm-4.5-air/tp4": (24, 2)}
NSM = torch.cuda.get_device_properties(0).multi_processor_count
ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)


def build(hq, hk, chunk, prefix, rows, qrow, row_kv, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    kv_lens = [prefix + chunk] + row_kv
    q_lens = [chunk] + [qrow] * rows
    total_kv = sum(kv_lens)
    k = torch.randn(total_kv + 64, 1, hk, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    v = torch.randn(total_kv + 64, 1, hk, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    perm = torch.randperm(total_kv + 64, device=dev, generator=g)[:total_kv].to(torch.int32)
    pages, off = [], 0
    for n in kv_lens:
        pages.append(perm[off:off + n]); off += n
    q = torch.randn(sum(q_lens), hq, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    return dict(k=k, v=v, pages=pages, q=q, q_lens=q_lens, kv_lens=kv_lens, hq=hq, hk=hk)


def fa2_wrapper(b, idx):
    """flashinfer FA2 BatchPrefill over the requests in idx (page size 1, causal: each request's queries are its last
    q_len KV positions)."""
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    qo = [0]; kvp = [0]; ind = []; last = []
    for i in idx:
        qo.append(qo[-1] + b["q_lens"][i]); kvp.append(kvp[-1] + b["kv_lens"][i]); ind.append(b["pages"][i]); last.append(1)
    w.plan(torch.tensor(qo, dtype=torch.int32, device=dev), torch.tensor(kvp, dtype=torch.int32, device=dev),
           torch.cat(ind), torch.tensor(last, dtype=torch.int32, device=dev), b["hq"], b["hk"], D, 1, causal=True,
           q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    qs = torch.cat([b["q"][sum(b["q_lens"][:i]):sum(b["q_lens"][:i + 1])] for i in idx])
    return w, qs


def fa3_prefill(b, sm_margin=0):
    c = b["q_lens"][0]
    n = b["kv_lens"][0]
    return lambda: flash_attn_with_kvcache(
        q=b["q"][:c], k_cache=b["k"], v_cache=b["v"], page_table=b["pages"][0][None, :],
        cache_seqlens=torch.tensor([n], dtype=torch.int32, device=dev),
        cu_seqlens_q=torch.tensor([0, c], dtype=torch.int32, device=dev), max_seqlen_q=c, causal=True,
        sm_margin=sm_margin)


def fa3_rows(b, sm_margin=0):
    n = len(b["q_lens"]) - 1
    kv = b["kv_lens"][1:]
    pt = torch.zeros(n, max(kv), dtype=torch.int32, device=dev)
    for i in range(n):
        pt[i, :kv[i]] = b["pages"][i + 1]
    ql = b["q_lens"][1:]
    cu = torch.tensor([0] + list(torch.tensor(ql).cumsum(0)), dtype=torch.int32, device=dev)
    kvl = torch.tensor(kv, dtype=torch.int32, device=dev)
    q = b["q"][b["q_lens"][0]:]
    return lambda: flash_attn_with_kvcache(q=q, k_cache=b["k"], v_cache=b["v"], page_table=pt, cache_seqlens=kvl,
                                           cu_seqlens_q=cu, max_seqlen_q=max(ql), causal=True, sm_margin=sm_margin)


def first(o):
    return o[0] if isinstance(o, tuple) else o


def ref(b, i, heads, rows):
    """fp32 causal attention of request i, sampled query heads and rows."""
    q0 = sum(b["q_lens"][:i]); ql = b["q_lens"][i]; n = b["kv_lens"][i]
    kk = b["k"][b["pages"][i].long(), 0].float(); vv = b["v"][b["pages"][i].long(), 0].float()
    out = {}
    for h in heads:
        kh = h // (b["hq"] // b["hk"])
        qr = b["q"][q0 + rows, h].float()
        s = qr @ kk[:, kh].T / D ** 0.5
        pos = n - ql + rows
        s.masked_fill_(torch.arange(n, device=dev)[None, :] > pos[:, None], float("-inf"))
        out[h] = torch.softmax(s, -1) @ vv[:, kh]
    return out


def check(b, o_pref, o_rows):
    """Max relative error vs the fp32 reference: prefill request (last 128 rows) and the first short row."""
    err = 0.0
    heads = (0, b["hq"] - 1)
    c = b["q_lens"][0]
    rows = torch.arange(c - 128, c, device=dev)
    r = ref(b, 0, heads, rows)
    for h in heads:
        err = max(err, ((o_pref[rows, h].float() - r[h]).abs().max() / r[h].abs().max()).item())
    ql = b["q_lens"][1]
    rows1 = torch.arange(ql, device=dev)
    r1 = ref(b, 1, heads, rows1)
    for h in heads:
        err = max(err, ((o_rows[:ql, h].float() - r1[h]).abs().max() / r1[h].abs().max()).item())
    return err


def timeit(fn, it=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(it):
        fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / it


def run_case(name, hq, hk, chunk, prefix, rows, qrow, row_kv):
    b = build(hq, hk, chunk, prefix, rows, qrow, row_kv)
    nreq = 1 + rows
    w_all, q_all = fa2_wrapper(b, range(nreq))
    w_rows, q_rows = fa2_wrapper(b, range(1, nreq))
    pre = fa3_prefill(b)
    unfused = lambda: w_all.run(q_all, (b["k"], b["v"]))
    o_all = unfused()
    e_unf = check(b, o_all[:chunk], o_all[chunk:])
    split = lambda: (pre(), w_rows.run(q_rows, (b["k"], b["v"])))
    o_pre = pre(); o_pre = o_pre[0] if isinstance(o_pre, tuple) else o_pre
    e_split = check(b, o_pre, w_rows.run(q_rows, (b["k"], b["v"])))
    side = torch.cuda.Stream()
    best = None
    for margin in (8, 16, 24):
        pre_m = fa3_prefill(b, margin)

        def corun(pre_m=pre_m):
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                w_rows.run(q_rows, (b["k"], b["v"]))
            pre_m()
            torch.cuda.current_stream().wait_stream(side)
        t = timeit(corun)
        if best is None or t < best[0]:
            best = (t, margin)
    rows3 = fa3_rows(b)
    split3 = lambda: (pre(), rows3())
    e_r3 = check(b, o_pre, first(rows3()))
    best3 = None
    for margin in (8, 16, 24, 32, 48):
        pre_m, rows_m = fa3_prefill(b, margin), fa3_rows(b, NSM - margin)

        def corun3(pre_m=pre_m, rows_m=rows_m):
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                rows_m()
            pre_m()
            torch.cuda.current_stream().wait_stream(side)
        t = timeit(corun3)
        if best3 is None or t < best3[0]:
            best3 = (t, margin)
    t_unf, t_split, t_s3 = timeit(unfused), timeit(split), timeit(split3)
    pct = lambda x: (x / t_unf - 1) * 100
    err = max(e_unf, e_split, e_r3)
    print(f"{name:16s} {chunk:5d}@{prefix:6d} + {rows:3d}x q{qrow}  unfused {t_unf:7.3f}  split {t_split:7.3f} "
          f"({pct(t_split):+5.1f}%)  co-run {best[0]:7.3f} ({pct(best[0]):+5.1f}%, m{best[1]})  "
          f"rows-FA3 {t_s3:7.3f} ({pct(t_s3):+5.1f}%)  rows-FA3 co-run {best3[0]:7.3f} ({pct(best3[0]):+5.1f}%, "
          f"m{best3[1]})  err {err:.1e} {'OK' if err < 2e-2 else 'FAIL'}", flush=True)


g = torch.Generator().manual_seed(1)
mix = [int(x) for x in torch.randint(4096, 65536, (64,), generator=g)]  # decode / verify rows' KV: 4K-64K
print(f"GPU {torch.cuda.get_device_name(0)}, {NSM} SMs; ms per layer")
for name, (hq, hk) in MODELS.items():
    for chunk, prefix in ((1024, 16384), (1024, 49152), (2048, 32768)):
        for mode, qrow in (("2-mode", 1), ("4-mode", 4)):
            for rows in (16, 48):
                run_case(f"{name} {mode}", hq, hk, chunk, prefix, rows, qrow, mix[:rows])
