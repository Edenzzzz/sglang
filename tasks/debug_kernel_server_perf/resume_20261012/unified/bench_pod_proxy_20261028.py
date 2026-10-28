"""POD-Attention's regime on Hopper, measured rather than asserted (user, 2026-10-28).

POD fuses the prefill chunk and the decode rows into ONE FA2 kernel. On Hopper that kernel still runs -- FA2 is
portable -- but its prefill half issues Ampere-style mma.sync instead of wgmma. So we time POD's structure (a single
stock flashinfer FA2 call covering the chunk AND the rows) against ours (stock FA3 for the chunk, stock FA2 for the
rows) on the same shapes. Any CTA-interleaving POD adds is an optimisation on top of that kernel and cannot recover
the instruction gap, so this is an upper bound on POD on Hopper.
  python bench_pod_proxy.py [--kv 131072] [--chunk 2048] [--rows 32]
"""
import argparse
import torch
import flashinfer
from sgl_kernel.flash_attn import flash_attn_with_kvcache


def t(fn, iters=20, warm=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) / iters)
    return best


ap = argparse.ArgumentParser()
ap.add_argument("--kv", type=int, default=131072)
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--rows", type=int, default=32)
ap.add_argument("--row-kv", type=int, default=4096)
a = ap.parse_args()
dev, dt = torch.device("cuda"), torch.bfloat16
H, KVH, D = 28, 4, 128
scale = D ** -0.5
torch.manual_seed(0)
pages = a.kv + a.rows * a.row_kv + 16
kc = torch.randn(pages, 1, KVH, D, device=dev, dtype=dt)
vc = torch.randn_like(kc)
wrk = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=dev)

# --- ours: FA3 for the chunk, FA2 for the rows (two calls)
q_chunk = torch.randn(a.chunk, H, D, device=dev, dtype=dt)
out_chunk = torch.empty_like(q_chunk)
tbl = torch.arange(0, a.kv, device=dev, dtype=torch.int32).view(1, a.kv)
cs = torch.tensor([a.kv], device=dev, dtype=torch.int32)
cu = torch.tensor([0, a.chunk], device=dev, dtype=torch.int32)
q_rows = torch.randn(a.rows, H, D, device=dev, dtype=dt)
out_rows = torch.empty_like(q_rows)
rtbl = torch.arange(a.kv, a.kv + a.rows * a.row_kv, device=dev, dtype=torch.int32)
w_rows = flashinfer.BatchPrefillWithPagedKVCacheWrapper(wrk, kv_layout="NHD", backend="fa2")
w_rows.plan(torch.arange(a.rows + 1, device=dev, dtype=torch.int32),
            torch.arange(a.rows + 1, device=dev, dtype=torch.int32) * a.row_kv,
            rtbl, torch.full((a.rows,), a.row_kv, dtype=torch.int32, device=dev),
            H, KVH, D, 1, causal=False, sm_scale=scale, q_data_type=dt)

def ours():
    flash_attn_with_kvcache(q=q_chunk, k_cache=kc, v_cache=vc, page_table=tbl, cache_seqlens=cs,
                            cu_seqlens_q=cu, max_seqlen_q=a.chunk, out=out_chunk, softmax_scale=scale, causal=True)
    w_rows.run(q_rows, (kc, vc), out=out_rows)

# --- POD's structure: ONE FA2 call covering chunk + rows together
q_all = torch.cat([q_chunk, q_rows], 0)
out_all = torch.empty_like(q_all)
qi = torch.tensor([0, a.chunk] + [a.chunk + i + 1 for i in range(a.rows)], device=dev, dtype=torch.int32)
kv_lens = [a.kv] + [a.row_kv] * a.rows
pi = torch.tensor([0] + list(torch.tensor(kv_lens).cumsum(0).tolist()), device=dev, dtype=torch.int32)
pages_all = torch.cat([torch.arange(0, a.kv, device=dev, dtype=torch.int32), rtbl])
w_pod = flashinfer.BatchPrefillWithPagedKVCacheWrapper(wrk, kv_layout="NHD", backend="fa2")
w_pod.plan(qi, pi, pages_all, torch.tensor(kv_lens, dtype=torch.int32, device=dev),
           H, KVH, D, 1, causal=True, sm_scale=scale, q_data_type=dt)

def pod():
    w_pod.run(q_all, (kc, vc), out=out_all)

t_ours, t_pod = t(ours), t(pod)
print(f"kv={a.kv} chunk={a.chunk} rows={a.rows}")
print(f"  ours (FA3 chunk + FA2 rows) : {t_ours:8.3f} ms")
print(f"  POD structure (one FA2 call): {t_pod:8.3f} ms")
print(f"  ours is {t_pod / t_ours:.2f}x faster than POD's structure on Hopper")
print("POD_PROXY_DONE")
