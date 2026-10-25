"""Synthetic mixed steps in the trace-sampled capture format, shaped after MELD's decode-attention evaluation (SOSP '26,
Table 4: ShareGPT / OpenR1-Math traces, batch 16-1024, H100), so every A6 arm, native v3 and the FP32 gate run on them
unchanged. Each step: one 512-token prefill chunk on a 4K context (row 0, FA3; the native v3 harness needs a prefill
row) plus `bs` decode (q=1, "two") or verify (q=4, "four") rows. Qwen2.5-7B geometry (28 / 4 heads, d 128), page 1.
Sets (KV length of a row = prompt + generated-so-far, generated ~ U(0, response)):
  sharegpt      prompt / response lognormal with ShareGPT's mean / std (929 / 1264, 466 / 652)
  math          OpenR1-Math (87 / 66, 11311 / 7401): long, spread KV
  longtail      90% ShareGPT rows + 10% agent rows of 64K-200K: the imbalanced case
  sharedprefix  ShareGPT rows behind one 4K prefix whose pages all rows share (MELD Fig. 11)
  sharedlong    ShareGPT rows behind one 48K shared prefix (agent fan-out; the prefix's K+V, 96 MB, exceeds L2)
  python gen_synth_meld_20261025.py <out root> [set ...]
Claude, 2026-10-25.
"""
import math
import sys
from pathlib import Path

import torch

HEADS, KV_HEADS, D = 28, 4, 128
PREFILL_Q, PREFILL_KV = 512, 4096
root = Path(sys.argv[1])
g = torch.Generator().manual_seed(20261025)


def lognormal(mean, std, n):
    s2 = math.log(1 + (std / mean) ** 2)
    mu = math.log(mean) - s2 / 2
    return torch.exp(mu + math.sqrt(s2) * torch.randn(n, generator=g))


def kv_lengths(kind, n):
    if kind in ("sharegpt", "sharedprefix", "sharedlong"):
        p, r = lognormal(929, 1264, n), lognormal(466, 652, n)
    elif kind == "math":
        p, r = lognormal(87, 66, n), lognormal(11311, 7401, n)
    else:  # longtail
        p, r = lognormal(929, 1264, n), lognormal(466, 652, n)
    kv = (p + r * torch.rand(n, generator=g)).clamp(64, 32768).long()
    if kind == "longtail":
        agent = torch.rand(n, generator=g) < 0.1
        agent[0] = True
        kv[agent] = torch.randint(65536, 200000, (int(agent.sum()),), generator=g)
    return kv.tolist()


def make(kind, bs, q):
    kv = [PREFILL_KV] + kv_lengths(kind, bs)
    shared = {"sharedprefix": 4096, "sharedlong": 49152}.get(kind, 0)
    if shared:
        kv = [PREFILL_KV] + [shared + k for k in kv[1:]]
    ql = [PREFILL_Q] + [q] * bs
    kv = [max(k, qq) for k, qq in zip(kv, ql)]
    private = [k - (shared if i > 0 else 0) for i, k in enumerate(kv)]
    live = sum(private) + shared
    pool = int(live * 1.25) + 1024
    phys = torch.randperm(pool, generator=g)[:live]
    page_table = torch.zeros(len(kv), max(kv), dtype=torch.int32)
    pref = phys[:shared]
    off = shared
    for i, k in enumerate(kv):
        if i > 0 and shared:
            page_table[i, :shared] = pref.int()
            n = k - shared
            page_table[i, shared:k] = phys[off:off + n].int()
        else:
            n = k
            page_table[i, :k] = phys[off:off + n].int()
        off += n
    total_q = sum(ql)
    cu = torch.tensor([0] + ql).cumsum(0).int()
    geometry = dict(
        ordinal=bs, mode="four" if q == 4 else "two", category=f"synth_{kind}", layer=0, query_lengths=ql,
        prefix_lengths=[k - qq for k, qq in zip(kv, ql)], kv_lengths=kv, prefix_rows=1, prefix_tokens=PREFILL_Q,
        row_query_max=q, row_context_max=max(kv[1:]), verify_tokens=q * bs if q > 1 else 0,
        verify_rows=bs if q > 1 else 0, small_prefill_rows_in_fa2=0, heads=HEADS, kv_heads=KV_HEADS,
        rank=0, packed_query_limit=64, scale=1 / math.sqrt(D), key_shape=[pool, 1, KV_HEADS, D],
        key_stride=[KV_HEADS * D, KV_HEADS * D, D, 1], value_shape=[pool, 1, KV_HEADS, D],
        value_stride=[KV_HEADS * D, KV_HEADS * D, D, 1], live_pages=live, query_stride=[HEADS * D, D, 1],
        source=f"synthetic MELD-shaped step ({kind}, bs {bs}, q {q}), gen_synth_meld_20261025.py")
    return dict(
        geometry=geometry,
        query=torch.randn(total_q, HEADS, D, generator=g).to(torch.bfloat16),
        key=torch.randn(live, 1, KV_HEADS, D, generator=g).to(torch.bfloat16),
        value=torch.randn(live, 1, KV_HEADS, D, generator=g).to(torch.bfloat16),
        physical_ids=phys.long(), page_table=page_table, cache_lengths=torch.tensor(kv, dtype=torch.int32),
        cumulative_query=cu)


PLAN = {"sharegpt": (64, 128, 256, 512, 1024), "math": (16, 64, 128, 256), "longtail": (16, 32, 64, 128),
        "sharedprefix": (64, 128, 256), "sharedlong": (16, 32, 64, 128)}
only = set(sys.argv[2:]) or {k for k in PLAN if k != "sharedlong"}
for kind, sizes in PLAN.items():
    if kind not in only:
        continue
    for q, mode in ((1, "two"), (4, "four")):
        d = root / f"{kind}_{mode}" / "captures"
        d.mkdir(parents=True, exist_ok=True)
        for bs in sizes:
            c = make(kind, bs, q)
            torch.save(c, d / f"s{bs:05d}_{kind}.pt")
            kv = c["geometry"]["kv_lengths"][1:]
            print(f"SYNTH {kind}_{mode} bs={bs} kv min/med/max={min(kv)}/{sorted(kv)[len(kv) // 2]}/{max(kv)} "
                  f"live={c['geometry']['live_pages']}", flush=True)
print("SYNTH_DONE")
