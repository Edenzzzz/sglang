#!/usr/bin/env python3
"""At what context length is attention more than half of a decode step?

Every attention-kernel win is multiplied by attention's share of the step, so this bounds what
the kernel work can do end to end, per context length. Two measurements of the SAME shapes:

  e2e   sglang.bench_one_batch median decode latency (whole step: 28 layers, GEMMs, norms,
        sampling, CUDA graph), Qwen2.5-7B, yarn x4 to 131072 as in serving
  attn  the decode kernel sglang runs for those steps (flashinfer BatchDecodeWithPagedKVCache,
        NHD, page 1, tensor cores as sglang picks for GQA 7), x 28 layers

  python context_share.py <bench_one_batch result.jsonl>
"""
import json, sys
import numpy as np, torch, flashinfer
from flashinfer.testing.utils import bench_gpu_time

H_QO, H_KV, D, LAYERS = 28, 4, 128, 28
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)
max_tok = max(r["batch_size"] * (r["input_len"] + r["output_len"]) for r in rows)
kv = torch.empty(max_tok, 2, 1, H_KV, D, device="cuda", dtype=torch.bfloat16).normal_()
print(f"{'batch':>5} {'ctx':>7} {'e2e step ms':>11} {'attn ms':>8} {'attn share':>10}")
for r in sorted(rows, key=lambda r: (r["batch_size"], r["input_len"])):
    B, L = r["batch_size"], r["input_len"] + r["output_len"] // 2  # mid-decode KV length
    indptr = torch.arange(0, (B + 1) * L, L, dtype=torch.int32, device="cuda")
    idx = torch.arange(B * L, dtype=torch.int32, device="cuda")
    last = torch.ones(B, dtype=torch.int32, device="cuda")
    w.plan(indptr, idx, last, H_QO, H_KV, D, 1, q_data_type=torch.bfloat16,
           kv_data_type=torch.bfloat16)
    q = torch.randn(B, H_QO, D, device="cuda", dtype=torch.bfloat16)
    t_us = float(np.median(bench_gpu_time(lambda: w.run(q, kv[: B * L])))) * 1e3
    attn_ms = t_us * LAYERS / 1e3
    e2e_ms = r["median_decode_latency"] * 1e3
    print(f"{B:5d} {L:7d} {e2e_ms:11.2f} {attn_ms:8.2f} {attn_ms / e2e_ms * 100:9.1f}%")
