"""SGLANG_SPLIT_ROWS_FA3=1: the split's decode / verify rows go to FA3 (sgl-kernel flash_attn_with_kvcache, varlen q,
GQA packed by FA3's own heuristic) instead of the flashinfer FA2 BatchPrefill wrapper.

rows_kernels_20261021.py (H200, Qwen2.5-7B heads, 48 rows at 4-64K KV, ms per layer): q4 verify rows take 8.19 on FA2
BatchPrefill -- the executor the split has used for these rows -- and 0.92 on FA3, against a KV-read floor of 0.78; q1
rows 0.91 on FA3. The fusion_decompose_20261021.py decomposition found these rows, not the prefill chunk, set the
mixed step's attention time.

The rows get their own FA3 plan (page table, cache_seqlens, cu_seqlens_q, max_seqlen_q), built once per step in
_fused_split_plan beside the prefill side's; with the flag on, the FA2 wrapper is neither created nor planned. Under
SPLIT_OVERLAP the rows run on the main stream while the prefill chunk runs on the side stream with sm_margin
SGLANG_FUSED_SM_MARGIN, so the rows take exactly those margin SMs (sm_margin = num_SMs - margin); sequential, they take
the whole GPU.

Also fixes the prefill side's page table for page_size > 1: `page_table[:, :max_k] // page_size` kept one column per
TOKEN, while FA3 reads column j as page j, so every column past the first named the wrong page. Both sides now take one
column per page (`[:, ::page_size] // page_size`, upstream flashattention_backend's construction).
  python apply_split_rows_fa3.py <tree>/python/sglang/srt
Claude, 2026-10-21.
"""
import sys

p = f"{sys.argv[1]}/layers/attention/flashinfer_backend.py"
s = open(p).read()
assert "SPLIT_ROWS_FA3" not in s, "already applied"


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:120]
    s = s.replace(old, new)


sub(
    'SPLIT_OVERLAP = os.environ.get("SGLANG_SPLIT_OVERLAP", "0") == "1"\n',
    'SPLIT_OVERLAP = os.environ.get("SGLANG_SPLIT_OVERLAP", "0") == "1"\n'
    "# Decode / verify rows of the split on FA3 instead of the FA2 BatchPrefill wrapper.\n"
    'SPLIT_ROWS_FA3 = os.environ.get("SGLANG_SPLIT_ROWS_FA3", "0") == "1"\n',
)

# Prefill side: one page-table column per page.
sub(
    """        torch.index_select(r2t, 0, fa3_t, out=_pt)
        page_table = _pt[:, :max_k]
        if self.page_size > 1:
            page_table = page_table // self.page_size
""",
    """        torch.index_select(r2t, 0, fa3_t, out=_pt)
        page_table = self._fs_page_table(_pt, max_k)
""",
)

# Rows side: skip the FA2 gather and wrapper, build the rows' FA3 plan instead.
sub(
    """        n2 = len(fa2_rows)
        # One flat gather""",
    """        n2 = len(fa2_rows)
        if SPLIT_ROWS_FA3:
            assert not FUSED_KERNEL_EXEC and not FUSED_SPLIT_NO_FA2_WRAPPER
            return self._fs_rows_fa3_plan(
                fa3_rows, fa2_rows, qo_lens, seq_lens_cpu, rp_cpu, tok_off, _tok_total,
                page_table, tok_idx, indptr, dev,
            )
        # One flat gather""",
)

sub(
    """    def _fused_split_plan(self, forward_batch: ForwardBatch) -> bool:
""",
    """    def _fs_page_table(self, rows_table, max_k):
        \"\"\"FA3 page table from req_to_token rows: one column per page of the first max_k tokens.\"\"\"
        if self.page_size == 1:
            return rows_table[:, :max_k]
        return rows_table[:, : max_k : self.page_size] // self.page_size

    def _fs_rows_fa3_plan(
        self, fa3_rows, fa2_rows, qo_lens, seq_lens_cpu, rp_cpu, tok_off, tok_total,
        page_table, tok_idx, indptr, dev,
    ):
        \"\"\"The split plan with the decode / verify rows on FA3 (SPLIT_ROWS_FA3).\"\"\"
        import numpy as np

        r2t = self.req_to_token_pool.req_to_token
        _cols = int(r2t.shape[1])
        n2 = len(fa2_rows)
        rows_t = self._fs_upload(
            "rows_ids", np.asarray([rp_cpu[r] for r in fa2_rows], dtype=np.int64), torch.int64, dev
        )
        _pt = self._fs_dev_buf("rows_page_table", n2 * _cols, r2t.dtype, dev).view(n2, _cols)
        torch.index_select(r2t, 0, rows_t, out=_pt)
        # Same span test as the FA2-wrapper path: two adjacent token ranges covering the batch.
        _f3, _f2 = fa3_rows, fa2_rows
        _fa3_span = _fa2_span = None
        if (
            int(_f3[-1]) - int(_f3[0]) == len(_f3) - 1
            and int(_f2[-1]) - int(_f2[0]) == len(_f2) - 1
        ):
            a3, b3 = int(tok_off[int(_f3[0])]), int(tok_off[int(_f3[-1]) + 1])
            a2, b2 = int(tok_off[int(_f2[0])]), int(tok_off[int(_f2[-1]) + 1])
            if min(a3, a2) == 0 and max(b3, b2) == tok_total and (b3 == a2 or b2 == a3):
                _fa3_span, _fa2_span = (a3, b3), (a2, b2)
        self.fused_split_plan = dict(
            fa3_span=_fa3_span,
            fa2_span=_fa2_span,
            tok_total=tok_total,
            fa3_tok=tok_idx("fa3_tok", fa3_rows),
            fa2_tok=tok_idx("fa2_tok", fa2_rows),
            cu_q=indptr("fa3_cu_q", qo_lens[fa3_rows]),
            max_q=int(qo_lens[fa3_rows].max()),
            page_table=page_table,
            cache_seqlens=self._fs_upload(
                "fa3_kvlen", seq_lens_cpu[fa3_rows].numpy().astype(np.int32), torch.int32, dev
            ),
            rows_fa3=dict(
                cu_q=indptr("rows_cu_q", qo_lens[fa2_rows]),
                max_q=int(qo_lens[fa2_rows].max()),
                page_table=self._fs_page_table(_pt, int(seq_lens_cpu[fa2_rows].max())),
                cache_seqlens=self._fs_upload(
                    "rows_kvlen", seq_lens_cpu[fa2_rows].numpy().astype(np.int32), torch.int32, dev
                ),
            ),
            n_fa3=len(fa3_rows),
            n_fa2=n2,
        )
        if FUSED_SPLIT_DEBUG:
            self._fused_split_debug_check("plan")
        return True

    def _fused_split_plan(self, forward_batch: ForwardBatch) -> bool:
""",
)

# Dispatch: the rows' executor.
sub(
    """        o_fa2 = self.fused_fa2_wrapper.forward(
            qv[_s2[0] : _s2[1]] if _use_views else qv[pl["fa2_tok"]],
            # Already resolved by the caller, including the fp4-dequant workspace case.
            kv_cache,
            causal=causal,
            sm_scale=layer.scaling,
            logits_soft_cap=logits_soft_cap,
        )
""",
    """        _q2 = qv[_s2[0] : _s2[1]] if _use_views else qv[pl["fa2_tok"]]
        _rp = pl.get("rows_fa3")
        if _rp is not None:
            if getattr(self, "_fs_num_sms", None) is None:
                self._fs_num_sms = torch.cuda.get_device_properties(qv.device).multi_processor_count
            o_fa2 = flash_attn_with_kvcache(
                q=_q2,
                k_cache=k_buf,
                v_cache=v_buf,
                page_table=_rp["page_table"],
                cache_seqlens=_rp["cache_seqlens"],
                cu_seqlens_q=_rp["cu_q"],
                max_seqlen_q=_rp["max_q"],
                softmax_scale=layer.scaling,
                causal=causal,
                softcap=logits_soft_cap if logits_soft_cap else 0.0,
                # Overlapped, the rows take the SMs the prefill chunk leaves free.
                sm_margin=(
                    self._fs_num_sms - self.fused_sm_margin
                    if _overlap and self.fused_sm_margin > 0
                    else 0
                ),
            )
            self.split_rows_fa3_calls = getattr(self, "split_rows_fa3_calls", 0) + 1
            if self.split_rows_fa3_calls % 5000 == 1:
                logger.info(
                    "SPLIT_ROWS_FA3 calls=%d rows=%d max_q=%d overlap=%d",
                    self.split_rows_fa3_calls, pl["n_fa2"], _rp["max_q"], int(_overlap),
                )
        else:
            o_fa2 = self.fused_fa2_wrapper.forward(
                _q2,
                # Already resolved by the caller, including the fp4-dequant workspace case.
                kv_cache,
                causal=causal,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )
""",
)
open(p, "w").write(s)
print("applied", p)
