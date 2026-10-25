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
        # Persistent FA2 grid: UNIFIED_FA2_CTAS_PER_SM resident CTAs per SM (the plan's LPT slots).
        # UNIFIED_FA2_CTAS_PER_SM16: the grid for plans whose FA2 rows all fit the 16-row tile (the kernel launches its
        # UNIFIED_FA2_MIN_BLOCKS16 instantiation for those).
        self.fa2_ctas = int(os.environ.get("UNIFIED_FA2_CTAS_PER_SM", fa2_ctas_per_sm)) * self.num_sm
        self.fa2_ctas16 = int(os.environ.get("UNIFIED_FA2_CTAS_PER_SM16", self.fa2_ctas // self.num_sm)) * self.num_sm
        self.fa3_ctas = self.num_sm
        self.fa3_min_packed = fa3_min_packed
        self.int_ws = torch.empty(int_bytes, dtype=torch.uint8, device=device)
        # Two pinned staging buffers, alternated per plan: the host writes plan N+1 while plan N's copy may still be
        # queued. Each buffer's event marks its copy done; the host waits on it only before reusing that buffer.
        self.pinned = [torch.empty(int_bytes // 2, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        self.pinned_done = [None, None]
        self.pinned_next = 0
        self.float_ws = torch.empty(1 << 20, dtype=torch.float32, device=device)
        self.pdl = pdl and os.environ.get("UNIFIED_PDL", "1") != "0"  # UNIFIED_PDL=0: serial launches (diagnostic)

    def plan(self, query_lengths, kv_lengths, page_table, q_indptr_dev=None, kv_lens_dev=None):
        """q_indptr_dev / kv_lens_dev: the step's cumulative query lengths and KV lengths already on the device (int32,
        e.g. the FA3 backend's cu_seqlens_q / cache_seqlens_int32). With them the plan issues no blocking host-to-device
        copy, so building it does not wait for the previous step's kernels."""
        q = torch.tensor([0, *query_lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
        kv = torch.tensor(kv_lengths, dtype=torch.int32)
        kvp = torch.cat([torch.zeros(1, dtype=torch.int32), kv.cumsum(0, dtype=torch.int32)])
        g = self.heads // self.kv_heads
        only16 = all(n * g <= 16 for n in query_lengths if n * g < self.fa3_min_packed)
        b = self.pinned_next
        self.pinned_next ^= 1
        if self.pinned_done[b] is not None:
            self.pinned_done[b].synchronize()
        plan, stats = torch.ops.unified_dispatch.build_unified_plan(
            self.int_ws, self.pinned[b], self.float_ws, q, kvp, kv, self.heads, self.kv_heads, 1,
            self.num_sm, self.fa2_ctas16 if only16 else self.fa2_ctas, self.fa3_ctas, self.fa3_min_packed)
        done = torch.cuda.Event()
        done.record()
        self.pinned_done[b] = done
        return UnifiedPlan(self, plan, stats, q, kvp, kv, page_table, q_indptr_dev, kv_lens_dev)


class UnifiedPlan:
    STATS = ("max_splits_fa3", "fa3_requests", "items48", "items32", "items16", "items_fa3",
             "merge_rows", "merge_groups", "fa3_merge_reqs", "fa2_chunk", "fa3_chunk", "int_bytes", "float_bytes")

    def __init__(self, ex, plan, stats, q, kvp, kv, page_table, q_indptr_dev=None, kv_lens_dev=None):
        self.ex, self.plan_vec = ex, list(plan)
        self.stats = dict(zip(self.STATS, stats))
        dev = ex.device
        on_device = q_indptr_dev is not None and kv_lens_dev is not None
        if on_device:
            self.q_indptr = q_indptr_dev
            self.kv_lens = kv_lens_dev
            self.kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=dev),
                                        kv_lens_dev.cumsum(0, dtype=torch.int32)])
        else:
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
            # SM pre-assignment: the FA3 grid's SM budget (UNIFIED_FA3_SM; default every SM). The FA2 launch fills the
            # SMs FA3 leaves free from the start of the step instead of only the gaps FA3 opens.
            self.fa3_sm = int(os.environ.get("UNIFIED_FA3_SM", str(ex.num_sm)))
            if not 0 < self.fa3_sm <= ex.num_sm:
                raise ValueError("FA3 SM budget outside the device")
            # no FA3 rows: correctly shaped placeholders (the FA2 launch checks the stock partial layout)
            self.po = torch.empty((1, h, 1, 128), dtype=torch.float32, device=dev)
            self.plse = torch.empty((1, h, 1), dtype=torch.float32, device=dev)
            if rows:
                kvl = kv[:rows]
                self.pre_indptr = self.q_indptr[:rows + 1]
                self.pre_lengths = self.kv_lens[:rows]
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
            # stock FA3 takes page-1 KV as [tokens, kv_heads, d]; a [pages, 1, kv_heads, d] cache is the same storage
            k3 = k_cache[:, 0] if k_cache.dim() == 4 and k_cache.size(1) == 1 else k_cache
            v3 = v_cache[:, 0] if v_cache.dim() == 4 and v_cache.size(1) == 1 else v_cache
            torch.ops.stock_fa3_pdl.run(
                q[:t], k3, v3, out[:t], self.pre_lse, self.pre_indptr, self.pre_lengths, self.pre_pages,
                self.po, self.plse, self.split_counts, self.tile_counter, max(1, self.stats["max_splits_fa3"]),
                self.pre_max_q, self.fa3_sm, float(scale), 2 if ex.pdl else 1, self.timeline)
        torch.ops.unified_dispatch.unified_run(
            q, k_cache, v_cache, self.kv_indices, out, ex.int_ws, ex.float_ws, self.plan_vec,
            ex.heads, ex.kv_heads, 1, float(scale), self.q_indptr, self.kv_indptr, self.kv_lens,
            self.max_q, self.max_pages, self.pt, self.po, self.plse, self.lse, not self.filled, ex.pdl)
        self.filled = True
        return out
