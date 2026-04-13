import torch
from triton.testing import do_bench


torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

M = 8192
N = 8192
K = 8192

a = torch.randn(M, K)
b = torch.randn(N, K)
c = torch.randn(M, N)

def fn():
    torch.matmul(a, b.t())

TEST_COUNT = 20
REPS = [50, 100, 200]
import time

for rep in REPS:
    time.sleep(1)
    ms = do_bench(fn, warmup=5, rep=rep, return_mode="median")
    print(f"REP:{rep} ms; SHAPE:[{M},{N},{K}] TIME:{ms:.3f} ms; {2*M*N*K / ms * 1e-9:.3f} TFLOPS")
