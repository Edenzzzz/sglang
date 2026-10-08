# Populate ~/.cache/flashinfer/*/generated/batch_attention_with* by driving one tiny
# BatchAttention plan+run. The harness globs for those generated sources at load() time,
# so on a fresh pod it must be run once before any arm.
import numpy as np, torch, glob, os
from flashinfer.attention import BatchAttention

HQ, HKV, Dh, PAGE = 28, 4, 128, 128
qo = np.array([0, 512], dtype=np.int32)
kvl = np.array([512], dtype=np.int32)
npages = -(-int(kvl[0]) // PAGE)
kv_c = np.array([0, npages], dtype=np.int32)
idx = torch.arange(npages, dtype=torch.int32, device="cuda")

wk = BatchAttention(torch.empty(512 << 20, dtype=torch.uint8, device="cuda"), "NHD")
wk.plan(torch.from_numpy(qo), torch.from_numpy(kv_c), idx,
        torch.from_numpy(kvl), HQ, HKV, Dh, Dh, PAGE, causal=False,
        q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
q = torch.empty(512, HQ, Dh, device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
kv = torch.empty(npages, 2, PAGE, HKV, Dh, device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
out = wk.run(q, (kv[:, 0], kv[:, 1]))
torch.cuda.synchronize()
print("warm run ok, out", tuple(out[0].shape) if isinstance(out, (tuple, list)) else tuple(out.shape))
g = glob.glob(os.path.expanduser("~/.cache/flashinfer/*/generated/batch_attention_with*"))
print("generated dirs found:", len(g))
for p in g[:3]:
    print("  ", p)
assert g, "cache still empty -- the harness glob will still fail"
print("WARMOK")
