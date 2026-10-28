"""Does an FA3 chunk tile leave SM slack for concurrent 16-row FA2 work, and does the chunk's tile size decide it?

The question (user, 2026-10-28): POD-style intra-SM overlap is unavailable on Hopper because a throughput-tuned
wgmma CTA is sized to fill the SM. But FA3 at a 64-row query tile uses ONE consumer warpgroup instead of two, so it
may leave room a 128-row tile does not -- in which case 16-row rows could ride alongside.

flash-attention's tile is not settable from outside, but it follows the query count: a chunk of <= 64 tokens gives
one 64-row tile, 128 gives one 128-row tile. So for each chunk size we time the SAME two calls the naive baseline
uses (stock FA3 via sgl_kernel.flash_attn for the chunk, stock flashinfer FA2 for the rows):

    FA3 alone, FA2 alone, serial (no overlap), two streams (concurrent), ideal = max(FA3, FA2)

and report streams/serial plus how much of the serial-to-ideal gap the streams recover. If the chunk tile saturates
the SM, streams ~ serial (what we already measured at the default configuration: 1.763x vs 1.759x). If a 64-row
tile leaves slack, streams should approach ideal at chunk 64 and not at 2048.

Run with the pip flashinfer (no port07 on PYTHONPATH), one GPU.
  python bench_q64_overlap_20261028.py [--kv 131072] [--rows 32] [--iters 30]
Claude.
"""
import argparse

import torch

import flashinfer
from sgl_kernel.flash_attn import flash_attn_with_kvcache


def time_it(fn, iters, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", type=int, default=131072)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--row-kv", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--heads", type=int, default=28)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    a = ap.parse_args()
    dev, dt = torch.device("cuda"), torch.bfloat16
    H, KVH, D = a.heads, a.kv_heads, a.dim
    scale = D ** -0.5
    torch.manual_seed(0)

    pages = a.kv + a.rows * a.row_kv + 16
    k_cache = torch.randn(pages, 1, KVH, D, device=dev, dtype=dt)
    v_cache = torch.randn_like(k_cache)
    side = torch.cuda.Stream()

    # row side: `rows` single-token queries, each over row_kv cached tokens -> the 16-row tile class
    row_table = torch.arange(a.kv, a.kv + a.rows * a.row_kv, device=dev, dtype=torch.int32).view(a.rows, a.row_kv)
    row_q = torch.randn(a.rows, H, D, device=dev, dtype=dt)
    row_out = torch.empty_like(row_q)
    wrk = torch.empty(384 * 1024 * 1024, dtype=torch.uint8, device=dev)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(wrk, kv_layout="NHD", backend="fa2")
    qi = torch.arange(a.rows + 1, device=dev, dtype=torch.int32)
    pi = torch.arange(a.rows + 1, device=dev, dtype=torch.int32) * a.row_kv
    pages_flat = row_table.reshape(-1).contiguous()
    wrapper.plan(qi, pi, pages_flat, torch.full((a.rows,), a.row_kv, dtype=torch.int32, device=dev),
                 H, KVH, D, 1, causal=False, sm_scale=scale, q_data_type=dt)

    def fa2_rows():
        wrapper.run(row_q, (k_cache, v_cache), out=row_out)

    print(f"KV={a.kv} rows={a.rows} row_kv={a.row_kv} heads={H}/{KVH} dim={D}")
    print(f"{'chunk':>6}{'FA3':>9}{'FA2':>8}{'serial':>9}{'streams':>9}{'ideal':>8}{'str/ser':>9}{'gap recovered':>15}")
    for chunk in (64, 128, 256, 2048):
        q = torch.randn(chunk, H, D, device=dev, dtype=dt)
        out = torch.empty_like(q)
        table = torch.arange(0, a.kv, device=dev, dtype=torch.int32).view(1, a.kv)
        cs = torch.tensor([a.kv], device=dev, dtype=torch.int32)
        cu = torch.tensor([0, chunk], device=dev, dtype=torch.int32)

        def fa3_chunk():
            flash_attn_with_kvcache(q=q, k_cache=k_cache, v_cache=v_cache, page_table=table,
                                    cache_seqlens=cs, cu_seqlens_q=cu, max_seqlen_q=chunk,
                                    out=out, softmax_scale=scale, causal=True)

        def serial():
            fa3_chunk(); fa2_rows()

        def streams():
            cur = torch.cuda.current_stream()
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                fa3_chunk()
            fa2_rows()
            cur.wait_stream(side)

        t3 = time_it(fa3_chunk, a.iters)
        t2 = time_it(fa2_rows, a.iters)
        ts = time_it(serial, a.iters)
        tp = time_it(streams, a.iters)
        ideal = max(t3, t2)
        gap = (ts - ideal)
        rec = 100 * min(max((ts - tp) / gap, 0), 1) if gap > 1e-6 else 0.0
        print(f"{chunk:>6}{t3:>9.4f}{t2:>8.4f}{ts:>9.4f}{tp:>9.4f}{ideal:>8.4f}{tp/ts:>8.4f}x{rec:>14.0f}%")
    print("Q64_OVERLAP_DONE")


if __name__ == "__main__":
    main()
