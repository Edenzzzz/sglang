"""Python side of the unified tile dispatcher (A6): one plan per step, two launches, no merge launch.
  ex = UnifiedExecutor(heads, kv_heads, device)
  plan = ex.plan(query_lengths, kv_lengths, page_table)   # host plan; once per step, shared by layers
  plan.run(q, k_cache, v_cache, scale, out)              # FA3 (task 3) -> PDL -> FA2 + merges
Claude, 2026-10-24.
"""
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

    def run(self, q, k_cache, v_cache, scale, out):
        ex = self.ex
        torch.ops.unified_dispatch.unified_run(
            q, k_cache, v_cache, self.kv_indices, out, ex.int_ws, ex.float_ws, self.plan_vec,
            ex.heads, ex.kv_heads, 1, float(scale), self.q_indptr, self.kv_indptr, self.kv_lens,
            self.max_q, self.max_pages, self.pt, self.po, self.plse, self.lse, not self.filled, ex.pdl)
        self.filled = True
        return out
