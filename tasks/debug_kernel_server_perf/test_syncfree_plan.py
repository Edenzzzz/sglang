"""Validate sync-free host-metadata plan for BatchAttention.

1. Correctness: host-tensor plan output == device-tensor plan output.
2. Pipeline: with ~40ms of queued GPU work, host plan() must not stall.
3. Double-buffer safety: alternating shapes, many iterations, outputs stay
   correct while the GPU runs behind.
"""

import time

import numpy as np
import torch

import flashinfer

torch.manual_seed(0)
device = "cuda"
num_qo_heads, num_kv_heads, head_dim = 28, 4, 128


def make_case(d_bs, d_kv, p_qo, p_kv):
    q_lens = torch.tensor([1] * d_bs + [p_qo], dtype=torch.int32)
    kv_lens = torch.tensor([d_kv] * d_bs + [p_kv], dtype=torch.int32)
    q_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0).int()])
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(kv_lens, 0).int()])
    nb = int(kv_indptr[-1])
    q = torch.rand(int(q_indptr[-1]), num_qo_heads, head_dim, device=device).to(torch.bfloat16)
    kv = torch.randn(nb, 2, 1, num_kv_heads, head_dim, device=device).to(torch.bfloat16)
    kv_indices = torch.arange(nb, dtype=torch.int32, device=device)
    return q_indptr, kv_indptr, kv_lens, q, kv, kv_indices


def plan_wrapper(pa, case, host):
    q_indptr, kv_indptr, kv_lens, q, kv, kv_indices = case
    if host:
        pa.plan(q_indptr, kv_indptr, kv_indices, kv_lens,
                num_qo_heads, num_kv_heads, head_dim, head_dim, 1, causal=True,
                q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    else:
        pa.plan(q_indptr.to(device), kv_indptr.to(device), kv_indices, kv_lens.to(device),
                num_qo_heads, num_kv_heads, head_dim, head_dim, 1, causal=True,
                q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)


# --- 1. correctness ---
pa = flashinfer.BatchAttention(kv_layout="NHD")
caseA = make_case(32, 8192, 4096, 8192)
caseB = make_case(8, 65536, 1024, 66560)

for name, case in [("A", caseA), ("B", caseB)]:
    plan_wrapper(pa, case, host=False)
    o_ref, lse_ref = pa.run(case[3], case[4])
    plan_wrapper(pa, case, host=True)
    o_host, lse_host = pa.run(case[3], case[4])
    assert torch.equal(o_ref, o_host), f"case {name}: outputs differ"
    print(f"correctness case {name}: OK (bitwise equal)")

# --- 2. pipeline stall ---
a = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)

def queue_work(n=40):
    for _ in range(n):
        a @ b

for host in (False, True):
    plan_wrapper(pa, caseA, host=host)  # warm
    torch.cuda.synchronize()
    queue_work()
    t0 = time.perf_counter()
    plan_wrapper(pa, caseA, host=host)
    dt = (time.perf_counter() - t0) * 1e3
    torch.cuda.synchronize()
    print(f"plan wall with ~40ms queued GPU work, host={host}: {dt:.3f} ms")

# --- 3. double-buffer safety under deep pipelines ---
pa2 = flashinfer.BatchAttention(kv_layout="NHD")
refs = []
for case in (caseA, caseB):
    plan_wrapper(pa2, case, host=False)
    o, _ = pa2.run(case[3], case[4])
    refs.append(o.clone())

outs = {0: [], 1: []}
for it in range(20):
    queue_work(10)  # keep GPU busy so plans run far ahead
    for i, case in enumerate((caseA, caseB)):
        plan_wrapper(pa2, case, host=True)
        o, _ = pa2.run(case[3], case[4])
        outs[i].append(o)
torch.cuda.synchronize()
for i in range(2):
    for it, o in enumerate(outs[i]):
        assert torch.equal(o, refs[i]), f"pipelined mismatch case {i} iter {it}"
print("double-buffer pipelined correctness: OK (20 iters x 2 shapes, bitwise)")
