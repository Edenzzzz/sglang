"""SGLANG_VERIFY_FA3=1: MTP target-verify steps run attention on FA3 (sgl-kernel flash_attn_with_kvcache, varlen q,
causal) instead of the flashinfer BatchPrefill verify wrapper.

Stock sglang on Hopper verifies on FA3 (its default backend for MHA models with no spec or topk 1); ours ran the
flashinfer wrapper, whose BatchPrefill takes 48 q4 rows at 4-64K KV in ~8 ms per layer vs FA3's 0.93
(rows_layout_check_20261021.py). EAGLE topk 1 only: its verify mask is the chain's lower triangle, which is FA3's
bottom-right causal mask over kv = seq_lens + draft_token_num (eagle_info.generate_attn_arg_prefill). topk > 1 raises.

Plan outside the graph, run inside it: the page table, cache_seqlens and cu_seqlens_q live in persistent buffers
filled in init_forward_metadata / init_forward_metadata_out_graph (which runs before every replay), and the captured
FA3 call reads them. With the flag on, the flashinfer verify wrapper is not planned, unless SGLANG_VERIFY_FA3_CHECK=1,
which plans it and compares every verify layer's FA3 output to it (needs --disable-cuda-graph; VERIFY_FA3_CHECK lines).
  python apply_verify_fa3.py <tree>/python/sglang/srt
Claude, 2026-10-21.
"""
import sys

p = f"{sys.argv[1]}/layers/attention/flashinfer_backend.py"
s = open(p).read()
assert "VERIFY_FA3" not in s, "already applied"
assert "SPLIT_ROWS_FA3" in s, "apply_split_rows_fa3.py first (shares _fs_page_table)"


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:120])
    s = s.replace(old, new)


sub(
    'SPLIT_ROWS_FA3 = os.environ.get("SGLANG_SPLIT_ROWS_FA3", "0") == "1"\n',
    'SPLIT_ROWS_FA3 = os.environ.get("SGLANG_SPLIT_ROWS_FA3", "0") == "1"\n'
    "# MTP target-verify attention on FA3; _CHECK also plans the flashinfer wrapper and compares.\n"
    'VERIFY_FA3 = os.environ.get("SGLANG_VERIFY_FA3", "0") == "1"\n'
    'VERIFY_FA3_CHECK = os.environ.get("SGLANG_VERIFY_FA3_CHECK", "0") == "1"\n',
)

# Reset with the split state at the top of both metadata hooks.
sub(
    """        self.fused_split_mask = None
        self.fused_split_plan = None
""",
    """        self.fused_split_mask = None
        self.fused_split_plan = None
        self._vf3 = None
""",
    count=3,
)
sub(
    '        self.fused_sm_margin = int(os.environ.get("SGLANG_FUSED_SM_MARGIN", "0"))\n',
    '        self.fused_sm_margin = int(os.environ.get("SGLANG_FUSED_SM_MARGIN", "0"))\n'
    "        self._vf3_bufs = None\n",
)

# Non-graph verify metadata.
sub(
    """        elif forward_batch.forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_wrappers_verify,
                use_ragged=False,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
                plan=forward_batch.kv_loc_plan,
            )
""",
    """        elif forward_batch.forward_mode.is_target_verify():
            if VERIFY_FA3:
                self._verify_fa3_plan(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.batch_size,
                    forward_batch.spec_info,
                )
            if not VERIFY_FA3 or VERIFY_FA3_CHECK:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_cpu,
                    forward_batch.seq_lens_sum,
                    prefix_lens=None,
                    prefill_wrappers=self.prefill_wrappers_verify,
                    use_ragged=False,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=forward_batch.spec_info,
                    plan=forward_batch.kv_loc_plan,
                )
""",
)

# Graph verify metadata (runs at capture and before every replay).
sub(
    """            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
                plan=forward_batch.kv_loc_plan,
            )
        elif forward_mode.is_dllm_extend():""",
    """            if VERIFY_FA3:
                self._verify_fa3_plan(req_pool_indices, seq_lens, bs, spec_info)
            if not VERIFY_FA3 or VERIFY_FA3_CHECK:
                self.indices_updater_prefill.update(
                    req_pool_indices[:bs],
                    seq_lens[:bs],
                    seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                    seq_lens_sum,
                    prefix_lens=None,
                    prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                    use_ragged=False,
                    encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                    spec_info=spec_info,
                    plan=forward_batch.kv_loc_plan,
                )
        elif forward_mode.is_dllm_extend():""",
)

sub(
    """    def _fs_page_table(self, rows_table, max_k):""",
    """    def _verify_fa3_plan(self, req_pool_indices, seq_lens, bs, spec_info):
        \"\"\"Fill the verify step's FA3 buffers. Persistent and full-width, so a captured FA3 call reads them.\"\"\"
        topk = getattr(spec_info, "topk", 1)
        if topk != 1:
            raise NotImplementedError(
                f"SGLANG_VERIFY_FA3 covers the EAGLE chain (topk 1) only; topk={topk} needs its tree mask"
            )
        n = int(spec_info.draft_token_num)
        r2t = self.req_to_token_pool.req_to_token
        dev = r2t.device
        if self._vf3_bufs is None:
            rows = int(r2t.shape[0])
            cols = (int(r2t.shape[1]) + self.page_size - 1) // self.page_size
            self._vf3_bufs = dict(
                pt=torch.zeros(rows, cols, dtype=torch.int32, device=dev),
                kvlen=torch.zeros(rows, dtype=torch.int32, device=dev),
                cu=torch.zeros(rows + 1, dtype=torch.int32, device=dev),
            )
        b = self._vf3_bufs
        rp = req_pool_indices[:bs].to(torch.int64)
        if self.page_size == 1:
            torch.index_select(r2t, 0, rp, out=b["pt"][:bs])
        else:
            b["pt"][:bs].copy_(r2t[rp, :: self.page_size] // self.page_size)
        b["kvlen"][:bs].copy_(seq_lens[:bs] + n)
        torch.arange(0, (bs + 1) * n, n, dtype=torch.int32, device=dev, out=b["cu"][: bs + 1])
        self._vf3 = dict(bs=bs, n=n)

    def _verify_fa3_forward(self, q, layer, causal, logits_soft_cap):
        from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
        from sglang.srt.mem_cache.layout.paged_view import paged_kv_view

        if layer.sliding_window_size not in (None, -1):
            raise NotImplementedError("SGLANG_VERIFY_FA3 on a sliding-window layer needs its window and SWA pool")
        bs, n = self._vf3["bs"], self._vf3["n"]
        b = self._vf3_bufs
        hq, hd = layer.tp_q_head_num, layer.head_dim
        _k_raw, _v_raw = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        o = flash_attn_with_kvcache(
            q=q.view(-1, hq, hd),
            k_cache=paged_kv_view(_k_raw, self.page_size, layer.tp_k_head_num, hd),
            v_cache=paged_kv_view(_v_raw, self.page_size, layer.tp_v_head_num, layer.v_head_dim),
            page_table=b["pt"][:bs],
            cache_seqlens=b["kvlen"][:bs],
            cu_seqlens_q=b["cu"][: bs + 1],
            max_seqlen_q=n,
            softmax_scale=layer.scaling,
            causal=causal,
            softcap=logits_soft_cap if logits_soft_cap else 0.0,
        )
        o = o[0] if isinstance(o, tuple) else o
        self.verify_fa3_calls = getattr(self, "verify_fa3_calls", 0) + 1
        if self.verify_fa3_calls % 5000 == 1:
            logger.info("VERIFY_FA3 calls=%d bs=%d draft_tokens=%d", self.verify_fa3_calls, bs, n)
        return o.view(-1, hq, hd)

    def _fs_page_table(self, rows_table, max_k):""",
)

# forward_extend: verify steps on FA3, optionally checked against the planned wrapper.
sub(
    """            if o is None:
                o = prefill_wrapper_paged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    kv_cache,
                    causal=causal,""",
    """            if (
                o is None
                and self._vf3 is not None
                and forward_batch.forward_mode.is_target_verify()
            ):
                o = self._verify_fa3_forward(q, layer, causal, logits_soft_cap)
                if VERIFY_FA3_CHECK and not torch.cuda.is_current_stream_capturing():
                    ref = prefill_wrapper_paged.forward(
                        q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        kv_cache,
                        causal=causal,
                        sm_scale=layer.scaling,
                        logits_soft_cap=logits_soft_cap,
                        k_scale=layer.k_scale_float,
                        v_scale=layer.v_scale_float,
                    ).view(o.shape)
                    err = ((o.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-6)).item()
                    self.verify_fa3_worst = max(getattr(self, "verify_fa3_worst", 0.0), err)
                    if err > FUSED_SPLIT_TOL:
                        logger.error(
                            "VERIFY_FA3_CHECK FAIL layer=%d err=%.5f bs=%d", layer.layer_id, err, self._vf3["bs"]
                        )
                    elif self.verify_fa3_calls % 500 == 1:
                        logger.info(
                            "VERIFY_FA3_CHECK ok layer=%d err=%.5f worst=%.5f",
                            layer.layer_id, err, self.verify_fa3_worst,
                        )
            if o is None:
                o = prefill_wrapper_paged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    kv_cache,
                    causal=causal,""",
)
open(p, "w").write(s)
print("applied", p)
