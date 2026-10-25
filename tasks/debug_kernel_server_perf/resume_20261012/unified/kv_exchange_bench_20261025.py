"""FA3 multicast step 1: paged K-tile delivery to a 2-CTA pair, full gather per CTA (mode 0) vs half gather + DSMEM push
(mode 1), kv_exchange_bench_20261025.cu. Checks the exchanged tile against the gathered rows first, then times both
modes with every pair reading the SAME KV (L2-hot, as FA3's q tiles of one row) and with distinct KV per pair (cold).
  python kv_exchange_bench_20261025.py
Claude, 2026-10-25.
"""
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

here = Path(__file__).resolve().parent
load(name="kv_exchange", sources=[str(here / "kv_exchange_bench_20261025.cu")],
     extra_cuda_cflags=["-O3", "-arch=sm_90a", "-std=c++17"], is_python_module=False, verbose=False)
ops = torch.ops.kv_exchange
dev = torch.device("cuda")
sms = torch.cuda.get_device_properties(dev).multi_processor_count
pairs = sms // 2
n_blocks = int(os.environ.get("KVX_BLOCKS", "64"))
tokens = 1 << 20
k = torch.randn(tokens, 128, device=dev, dtype=torch.bfloat16)
sink = torch.zeros(1, device=dev)
per = n_blocks * 128
g = torch.Generator(device="cpu").manual_seed(0)
for scenario in ("hot", "cold"):
    if scenario == "hot":
        one = torch.randperm(tokens, generator=g)[:per].to(torch.int32)
        pt = one.repeat(pairs).to(dev)
    else:
        pt = torch.randint(0, tokens, (pairs * per,), generator=g, dtype=torch.int32).to(dev)
    bad = ops.check(k, pt, pairs)
    t0 = ops.run(k, pt, n_blocks, pairs, 0, 20, sink)
    t1 = ops.run(k, pt, n_blocks, pairs, 1, 20, sink)
    gb = pairs * 2 * n_blocks * 128 * 256 / 1e9
    print(f"KVX {scenario} pairs={pairs} n_blocks={n_blocks} check_bad={bad} full_gather={t0:.4f}ms "
          f"({gb / t0 * 1e3:.0f} GB/s delivered) half+push={t1:.4f}ms ({gb / t1 * 1e3:.0f} GB/s) "
          f"speedup={t0 / t1:.3f}x", flush=True)
print("KVX_DONE")
