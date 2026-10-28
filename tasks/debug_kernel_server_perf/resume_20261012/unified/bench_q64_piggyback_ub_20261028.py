"""Upper bound on q64xq16 piggybacking: if every row tile rides free inside a chunk CTA, the gain is the share of
attention time the rows cost today. Measured per model (grouped-query ratio) and per mode, plus the piggyback
capacity implied by the per-tile work ratio."""
import torch, flashinfer
from sgl_kernel.flash_attn import flash_attn_with_kvcache

def t(fn, iters=20, warm=6):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); best = float("inf")
    for _ in range(4):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters): fn()
        b.record(); torch.cuda.synchronize(); best = min(best, a.elapsed_time(b)/iters)
    return best

dev, dt = torch.device("cuda"), torch.bfloat16
D, scale = 128, 128 ** -0.5
MODELS = [("Qwen2.5-7B", 28, 4), ("Llama-3.3-70B", 64, 8), ("GLM-4.5-Air", 96, 8)]
CHUNK, ROWS, ROWKV = 2048, 32, 4096
print(f"chunk={CHUNK} rows={ROWS} row_kv={ROWKV}  (row tile = 16; chunk tiles = {CHUNK//64} at q64)")
print(f"{'model':16}{'ctx':>8}{'mode':>7}{'chunk ms':>10}{'rows ms':>9}{'row share':>11}{'= UPPER BOUND':>15}")
for name, H, KVH in MODELS:
    for kv in (32768, 131072):
        pages = kv + ROWS*ROWKV + 16
        kc = torch.randn(pages, 1, KVH, D, device=dev, dtype=dt); vc = torch.randn_like(kc)
        q = torch.randn(CHUNK, H, D, device=dev, dtype=dt); o = torch.empty_like(q)
        tbl = torch.arange(0, kv, device=dev, dtype=torch.int32).view(1, kv)
        cs = torch.tensor([kv], device=dev, dtype=torch.int32)
        cu = torch.tensor([0, CHUNK], device=dev, dtype=torch.int32)
        def chunk_fn():
            flash_attn_with_kvcache(q=q, k_cache=kc, v_cache=vc, page_table=tbl, cache_seqlens=cs,
                                    cu_seqlens_q=cu, max_seqlen_q=CHUNK, out=o, softmax_scale=scale, causal=True)
        tc = t(chunk_fn)
        wrk = torch.empty(512*1024*1024, dtype=torch.uint8, device=dev)
        rtbl = torch.arange(kv, kv+ROWS*ROWKV, device=dev, dtype=torch.int32)
        for mode, qlen in (("2-mode", 1), ("4-mode", 4)):
            qr = torch.randn(ROWS*qlen, H, D, device=dev, dtype=dt); orr = torch.empty_like(qr)
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(wrk, kv_layout="NHD", backend="fa2")
            w.plan(torch.arange(ROWS+1, device=dev, dtype=torch.int32)*qlen,
                   torch.arange(ROWS+1, device=dev, dtype=torch.int32)*ROWKV, rtbl,
                   torch.full((ROWS,), ROWKV, dtype=torch.int32, device=dev), H, KVH, D, 1,
                   causal=False, sm_scale=scale, q_data_type=dt)
            tr = t(lambda: w.run(qr, (kc, vc), out=orr))
            share = tr/(tc+tr)
            print(f"{name:16}{kv//1024:>6}K{mode:>7}{tc:>10.3f}{tr:>9.3f}{100*share:>10.2f}%{100*share:>14.2f}%")
print("UB_DONE")
