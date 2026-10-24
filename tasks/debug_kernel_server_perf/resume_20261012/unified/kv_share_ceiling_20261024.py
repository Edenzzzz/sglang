"""Shared-KV ceiling for multicast (A5.5), CPU only: for every trace-sampled capture, how many of each role's KV page
reads another tile of the same step also reads. Rows are routed as in stage 1: > 32 packed queries -> FA3 (128-row
tiles), otherwise FA2 (one 16/32-row tile per row and kv head).
  inter-row: 1 - unique physical pages / total pages, over the role's rows (shared radix prefixes)
  intra-row: 1 - 1/tiles of the q tiles that read the same row's KV (counted without causal trimming: upper bound)
Upper bounds on load BYTES; most of these loads already hit L2. Pages are per kv head, so the fractions are per head.
  python kv_share_ceiling_20261024.py <captures dir> [...]
Claude, 2026-10-24.
"""
import sys
from pathlib import Path

import torch

for d in sys.argv[1:]:
    agg = {"fa3": [0, 0, 0], "fa2": [0, 0, 0]}  # [page reads incl. q tiles, page reads once per row, unique pages]
    for path in sorted(Path(d).glob("*.pt")):
        c = torch.load(path, map_location="cpu", weights_only=False)
        g = c["geometry"]
        G = g["heads"] // g["kv_heads"]
        for role in ("fa3", "fa2"):
            seen = set()
            for i, (ql, kv) in enumerate(zip(g["query_lengths"], g["kv_lengths"])):
                if (ql * G > 32) != (role == "fa3"):
                    continue
                pages = c["page_table"][i, :kv].tolist()  # original physical page ids
                tiles = -(-ql * G // 128) if role == "fa3" else 1
                agg[role][0] += tiles * kv
                agg[role][1] += kv
                seen.update(pages)
            agg[role][2] += len(seen)
    for role, (reads, once, uniq) in agg.items():
        if reads == 0:
            print(f"KVSHARE {Path(d).parent.name} {role}: no rows")
            continue
        print(f"KVSHARE {Path(d).parent.name} {role}: page reads {reads}, inter-row shared {1 - uniq / once:.4f}, "
              f"intra-row (q tiles) shared {1 - once / reads:.4f}, total shared {1 - uniq / reads:.4f}", flush=True)
