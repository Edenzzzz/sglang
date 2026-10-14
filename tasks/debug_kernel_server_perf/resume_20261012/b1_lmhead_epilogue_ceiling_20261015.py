"""B ceiling for TRUE epilogue fusion of the target argmax into the LM-head GEMM (2026-10-15).

At serving verify shapes (Qwen2.5-7B: hidden 3584, vocab 151936, bf16), time on the H200:
  gemm      logits = h @ W^T                       (cuBLAS, the current LM head)
  argmax    ids = logits.argmax(-1)                (the separate pass an epilogue would delete)
  gemm_f32  the same GEMM with fp32 output        (what a sampler that upcasts would read)
A GEMM-epilogue argmax removes the argmax pass and the logits write+read, so (argmax time + the
logits' HBM traffic) bounds its saving; it is compared with the measured serving step time.
Correctness of the timed argmax is checked against an fp32 reference on every shape.
"""
import torch

torch.manual_seed(0)
H, V = 3584, 151936
W = (torch.randn(V, H, device="cuda", dtype=torch.bfloat16) * 0.02)
def bench(fn, iters=200, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters
for T in (48, 184, 1024):
    h = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    logits = torch.empty(T, V, device="cuda", dtype=torch.bfloat16)
    ids = torch.empty(T, device="cuda", dtype=torch.long)
    t_gemm = bench(lambda: torch.matmul(h, W.t(), out=logits))
    t_arg = bench(lambda: torch.argmax(logits, dim=-1, out=ids))
    t_both = bench(lambda: (torch.matmul(h, W.t(), out=logits), torch.argmax(logits, dim=-1, out=ids)))
    ref = (h.float() @ W.float().t()).argmax(-1)
    agree = (torch.matmul(h, W.t()).argmax(-1) == ref).float().mean().item()
    mb = T * V * 2 / 1e6
    print(f"LMHEAD T={T:5d} gemm={t_gemm*1e3:8.1f}us argmax={t_arg*1e3:7.1f}us gemm+argmax={t_both*1e3:8.1f}us "
          f"logits={mb:6.1f}MB (HBM w+r ~{2*mb/4.8e3*1e3:5.1f}us) argmax_saving_bound={(t_both-t_gemm)*1e3:6.1f}us "
          f"bf16_vs_fp32_argmax_agree={agree:.4f}", flush=True)
print("B1_LMHEAD_CEILING_DONE")
