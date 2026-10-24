"""Python side of the unified tile dispatcher (A6): one plan per step, two launches, no merge launch.
  ex = UnifiedExecutor(heads, kv_heads, device)
  plan = ex.plan(query_lengths, kv_lengths, page_table)   # host plan; once per step, shared by layers
  plan.run(q, k_cache, v_cache, scale, out)              # FA3 (task 3) -> PDL -> FA2 + merges
Claude, 2026-10-24.
"""
import os

import torch


class UnifiedExecutor:
    def __init__(self, heads, kv_heads, device, fa3_min_packed=49, fa2_ctas_per_sm=2,
                 int_bytes=256 << 20, pdl=True):
        self.heads, self.kv_heads, self.device = heads, kv_heads, device
        self.num_sm = torch.cuda.get_device_properties(device).multi_processor_count
        self.fa2_ctas = fa2_ctas_per_sm * self.num_sm
        self.fa3_ctas = self.num_sm
        self.fa3_min_packed = fa3_min_packed
        self.int_ws = torch.empty(int_bytes, dtype=torch.uint8, device=device)
        self.pinned = torch.empty(int_bytes, dtype=torch.uint8, pin_memory=True)
        self.float_ws = torch.empty(1 << 20, dtype=torch.float32, device=device)
        self.pdl = pdl

    def plan(self, query_lengths, kv_lengths, page_table):
        q = torch.tensor([0, *query_lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
        kv = torch.tensor(kv_lengths, dtype=torch.int32)
        kvp = torch.cat([torch.zeros(1, dtype=torch.int32), kv.cumsum(0, dtype=torch.int32)])
        plan, stats = torch.ops.unified_dispatch.build_unified_plan(
            self.int_ws, self.pinned, self.float_ws, q, kvp, kv, self.heads, self.kv_heads, 1,
            self.num_sm, self.fa2_ctas, self.fa3_ctas, self.fa3_min_packed)
        return UnifiedPlan(self, plan, stats, q, kvp, kv, page_table)


class UnifiedPlan:
    STATS = ("max_splits_fa3", "fa3_requests", "items48", "items32", "items16", "items_fa3",
             "merge_rows", "merge_groups", "fa3_merge_reqs", "fa2_chunk", "fa3_chunk", "int_bytes", "float_bytes")

    def __init__(self, ex, plan, stats, q, kvp, kv, page_table):
        self.ex, self.plan_vec = ex, list(plan)
        self.stats = dict(zip(self.STATS, stats))
        dev = ex.device
        self.q_indptr = q.to(dev)
        self.kv_indptr = kvp.to(dev)
        self.kv_lens = kv.to(dev)
        rows = [page_table[i, :int(kv[i])] for i in range(len(kv))]
        self.kv_indices = torch.cat(rows).to(torch.int32).contiguous()
        self.total_q = int(q[-1])
        self.max_q = int((q[1:] - q[:-1]).max())
        self.max_pages = int(kv.max())
        n = len(kv)
        splits = max(1, self.stats["max_splits_fa3"])
        h = ex.heads
        self.pt = torch.full((n * self.max_pages,), -1, dtype=torch.int32, device=dev)
        self.po = torch.empty(splits * h * self.total_q * 128, dtype=torch.float32, device=dev)
        self.plse = torch.empty(splits * h * self.total_q, dtype=torch.float32, device=dev)
        self.lse = torch.empty(self.total_q * h, dtype=torch.float32, device=dev)
        self.filled = False
        # UNIFIED_FA3_STOCK=1 (must match a -DUNIFIED_FA3_STOCK=1 build): FA3 rows run as stock flash-attention with its
        # own tile scheduler, launched the way native v3 launches it, with the unified plan's split count. Its split
        # partials are reduced by the plan's FA3 merge items in the FA2 kernel (no combine launch).
        self.stock = os.environ.get("UNIFIED_FA3_STOCK") == "1"
        if self.stock:
            if not hasattr(torch.ops, "stock_fa3_pdl") or not hasattr(torch.ops.stock_fa3_pdl, "run"):
                torch.ops.load_library(os.environ["NATIVE_PDL_FA3_LIBRARY"])
            g = ex.heads // ex.kv_heads
            ql = (q[1:] - q[:-1]).tolist()
            fa3 = [n * g >= ex.fa3_min_packed for n in ql]
            rows = sum(fa3)
            if fa3 != [True] * rows + [False] * (len(ql) - rows):
                raise ValueError("stock FA3 needs the FA3 rows to be a prefix of the batch")
            self.fa3_rows, self.fa3_tokens = rows, int(q[rows])
            if rows:
                kvl = kv[:rows]
                self.pre_indptr = q[:rows + 1].to(dev)
                self.pre_lengths = kv[:rows].to(dev)
                self.pre_pages = page_table[:rows, :int(kvl.max())].contiguous()
                self.pre_max_q = int(max(ql[:rows]))
                self.po = torch.empty((splits, h, self.fa3_tokens, 128), dtype=torch.float32, device=dev)
                self.plse = torch.empty((splits, h, self.fa3_tokens), dtype=torch.float32, device=dev)
                self.pre_lse = torch.empty((h, self.fa3_tokens), dtype=torch.float32, device=dev)
                self.split_counts = torch.full((rows,), splits, dtype=torch.int32, device=dev)
                self.tile_counter = torch.zeros(1, dtype=torch.int32, device=dev)
                self.timeline = torch.empty(0, dtype=torch.int64, device=dev)

    def run(self, q, k_cache, v_cache, scale, out):
        ex = self.ex
        if self.stock and self.fa3_rows:
            t = self.fa3_tokens
            torch.ops.stock_fa3_pdl.run(
                q[:t], k_cache, v_cache, out[:t], self.pre_lse, self.pre_indptr, self.pre_lengths, self.pre_pages,
                self.po, self.plse, self.split_counts, self.tile_counter, max(1, self.stats["max_splits_fa3"]),
                self.pre_max_q, ex.num_sm, float(scale), 2 if ex.pdl else 1, self.timeline)
        torch.ops.unified_dispatch.unified_run(
            q, k_cache, v_cache, self.kv_indices, out, ex.int_ws, ex.float_ws, self.plan_vec,
            ex.heads, ex.kv_heads, 1, float(scale), self.q_indptr, self.kv_indptr, self.kv_lens,
            self.max_q, self.max_pages, self.pt, self.po, self.plse, self.lse, not self.filled, ex.pdl)
        self.filled = True
        return out
