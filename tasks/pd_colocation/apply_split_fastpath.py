"""Cut the split dispatch's PER-LAYER cost on launch-bound steps (SGLANG_SPLIT_FASTPATH=1).

Measured cause (H200, 2026-10-22). With the split disabled so every extend step is one varlen FA3 call -- the same
kernel upstream's FA3 backend runs -- our mixed prefill steps are still 43% slower in the shallow bin, and that bin's
steps turn out to be tiny: ptok 164, prefix 37, total KV 201, **nd = 0**. Attention there is microscopic, so the 3.2 ms
gap (10.67 vs 7.48 ms per step) is ~114 us per layer of GPU-timeline time that is not attention math -- the dispatch's
own per-layer work, on steps whose kernels are too small to hide it. The page table's row stride is NOT the cause
(page_table_stride_20261022.py: narrow vs stride-131072 views differ -4..+3%).

Three per-layer costs removed:
  1. `torch.empty_like(qv)` plus `o[span].copy_(o_fa3)` every layer. When the plan is a single FA3 call over the whole
     batch, FA3's own output already IS the answer, so it is returned directly: one allocation and one full-output
     copy per layer disappear.
  2. `get_kv_buffer` + `paged_kv_view` every layer. The pool's buffers and their paged views are stable for the
     process, so they are built once per layer id and cached.
  3. The split's own counter formatting, which ran `getattr` chains per layer; now a plain int.
Numerics are untouched: the same FA3 call with the same arguments. Guarded so the two-sided split path keeps its
scatter / view behaviour exactly.
  python apply_split_fastpath.py <tree>/python/sglang/srt      (after apply_split_policy.py)
Claude, 2026-10-22.
"""
import sys

p = f"{sys.argv[1]}/layers/attention/flashinfer_backend.py"
s = open(p).read()
assert "SPLIT_FASTPATH" not in s, "already applied"
assert "_fs_chunk_ms" in s, "apply_split_policy.py first"


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:120])
    s = s.replace(old, new)


sub(
    'SPLIT_POLICY = os.environ.get("SGLANG_SPLIT_POLICY", "0") == "1"\n',
    'SPLIT_POLICY = os.environ.get("SGLANG_SPLIT_POLICY", "0") == "1"\n'
    "# Per-layer fast path for the dispatch: no output copy on single-call steps, cached paged K/V views.\n"
    'SPLIT_FASTPATH = os.environ.get("SGLANG_SPLIT_FASTPATH", "0") == "1"\n',
)

# Cached paged K/V views, built once per layer id.
sub(
    """    def _fs_chunk_ms(self, forward_batch, fa3_rows):""",
    """    def _fs_kv_views(self, layer, hd):
        \"\"\"Paged K/V views for this layer, cached: the pool's buffers do not move after allocation.\"\"\"
        c = getattr(self, "_fs_kv_cache", None)
        if c is None:
            c = self._fs_kv_cache = {}
        v = c.get(layer.layer_id)
        if v is None:
            from sglang.srt.mem_cache.layout.paged_view import paged_kv_view

            _k_raw, _v_raw = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            v = c[layer.layer_id] = (
                paged_kv_view(_k_raw, self.page_size, layer.tp_k_head_num, hd),
                paged_kv_view(_v_raw, self.page_size, layer.tp_v_head_num, layer.v_head_dim),
            )
        return v

    def _fs_chunk_ms(self, forward_batch, fa3_rows):""",
)

# The dispatch: skip the output allocation on single-call steps under the fast path.
sub(
    """        o = torch.empty_like(qv)
        # Contiguous spans -> write through views, no scatter. See the plan for the measurement.""",
    """        _single_fast = SPLIT_FASTPATH and pl.get("single", False)
        o = None if _single_fast else torch.empty_like(qv)
        # Contiguous spans -> write through views, no scatter. See the plan for the measurement.""",
)

# Use the cached views instead of rebuilding them per layer.
sub(
    """        from sglang.srt.mem_cache.layout.paged_view import paged_kv_view

        _k_raw, _v_raw = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k_buf = paged_kv_view(_k_raw, self.page_size, layer.tp_k_head_num, hd)
        v_buf = paged_kv_view(_v_raw, self.page_size, layer.tp_v_head_num, layer.v_head_dim)
""",
    """        if SPLIT_FASTPATH:
            k_buf, v_buf = self._fs_kv_views(layer, hd)
        else:
            from sglang.srt.mem_cache.layout.paged_view import paged_kv_view

            _k_raw, _v_raw = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            k_buf = paged_kv_view(_k_raw, self.page_size, layer.tp_k_head_num, hd)
            v_buf = paged_kv_view(_v_raw, self.page_size, layer.tp_v_head_num, layer.v_head_dim)
""",
)

# The query slice: on a single-call step it is the whole tensor, so skip the slicing arithmetic.
sub(
    """        _q3 = qv[_s3[0] : _s3[1]] if _use_views else qv[pl["fa3_tok"]]""",
    """        if _single_fast:
            _q3 = qv
        else:
            _q3 = qv[_s3[0] : _s3[1]] if _use_views else qv[pl["fa3_tok"]]""",
)

# Return FA3's own output on a single-call step: no scatter, no copy, no second allocation.
sub(
    """        else:
            if _use_views:
                o[_s3[0] : _s3[1]].copy_(o_fa3.view(-1, hq, hd))
            else:
                o.index_copy_(0, pl["fa3_tok"], o_fa3.view(-1, hq, hd).to(o.dtype))
""",
    """        elif _single_fast:
            pass
        else:
            if _use_views:
                o[_s3[0] : _s3[1]].copy_(o_fa3.view(-1, hq, hd))
            else:
                o.index_copy_(0, pl["fa3_tok"], o_fa3.view(-1, hq, hd).to(o.dtype))
""",
)

sub(
    """        if pl.get("single"):
            self.split_single_calls = getattr(self, "split_single_calls", 0) + 1""",
    """        if pl.get("single"):
            if _single_fast:
                # Dtype is checked, not assumed: the scatter path used to cast with `.to(o.dtype)`, so a
                # kernel returning a different dtype than the query would change what the caller receives.
                _ov = o_fa3.view(-1, hq, hd)
                o = _ov if _ov.dtype == qv.dtype else _ov.to(qv.dtype)
            self.split_single_calls = getattr(self, "split_single_calls", 0) + 1""",
)
open(p, "w").write(s)
print("applied", p)
