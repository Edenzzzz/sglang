"""SGLANG_DECODE_FA3=1: plain decode steps (no speculative input) run attention on FA3 instead of the flashinfer
BatchDecode wrapper -- stock sglang's Hopper default (FA3 backend) decodes on FA3, so without MTP ours should too.

Built on apply_verify_fa3.py: the same persistent buffers and FA3 call, with one query per request and
cache_seqlens = seq_lens (the decode token is already in seq_lens, as in upstream flashattention_backend). Only
batches with spec_info None: the draft model's multi-step decode is this same backend class, and its KV lengths carry
per-step offsets from spec_info, so it keeps the flashinfer wrapper (one draft layer). SGLANG_VERIFY_FA3_CHECK=1 also
checks decode layers against the planned BatchDecode wrapper (needs --disable-cuda-graph).
  python apply_decode_fa3.py <tree>/python/sglang/srt
Claude, 2026-10-21.
"""
import sys

p = f"{sys.argv[1]}/layers/attention/flashinfer_backend.py"
s = open(p).read()
assert "DECODE_FA3" not in s, "already applied"
assert "_verify_fa3_plan" in s, "apply_verify_fa3.py first"


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:120])
    s = s.replace(old, new)


sub(
    'VERIFY_FA3_CHECK = os.environ.get("SGLANG_VERIFY_FA3_CHECK", "0") == "1"\n',
    'VERIFY_FA3_CHECK = os.environ.get("SGLANG_VERIFY_FA3_CHECK", "0") == "1"\n'
    "# Plain decode (spec_info None) on FA3; checked under VERIFY_FA3_CHECK.\n"
    'DECODE_FA3 = os.environ.get("SGLANG_DECODE_FA3", "0") == "1"\n',
)

# The plan helper: decode = one query per request, KV length = seq_lens.
sub(
    """    def _verify_fa3_plan(self, req_pool_indices, seq_lens, bs, spec_info):
        \"\"\"Fill the verify step's FA3 buffers. Persistent and full-width, so a captured FA3 call reads them.\"\"\"
        topk = getattr(spec_info, "topk", 1)
        if topk != 1:
            raise NotImplementedError(
                f"SGLANG_VERIFY_FA3 covers the EAGLE chain (topk 1) only; topk={topk} needs its tree mask"
            )
        n = int(spec_info.draft_token_num)
""",
    """    def _verify_fa3_plan(self, req_pool_indices, seq_lens, bs, spec_info, decode=False):
        \"\"\"Fill the verify (or, decode=True, plain decode) step's FA3 buffers. Persistent and full-width, so a
        captured FA3 call reads them.\"\"\"
        if decode:
            n = 1
        else:
            topk = getattr(spec_info, "topk", 1)
            if topk != 1:
                raise NotImplementedError(
                    f"SGLANG_VERIFY_FA3 covers the EAGLE chain (topk 1) only; topk={topk} needs its tree mask"
                )
            n = int(spec_info.draft_token_num)
""",
)
sub(
    """        b["kvlen"][:bs].copy_(seq_lens[:bs] + n)
        torch.arange(0, (bs + 1) * n, n, dtype=torch.int32, device=dev, out=b["cu"][: bs + 1])
        self._vf3 = dict(bs=bs, n=n)
""",
    """        b["kvlen"][:bs].copy_(seq_lens[:bs] if decode else seq_lens[:bs] + n)
        torch.arange(0, (bs + 1) * n, n, dtype=torch.int32, device=dev, out=b["cu"][: bs + 1])
        self._vf3 = dict(bs=bs, n=n, decode=decode)
""",
)

# Non-graph decode metadata.
sub(
    """        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                decode_wrappers=self.decode_wrappers,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
                fixed_split_size=self.decode_split_tile_size,
                disable_split_kv=False,
                req_pool_indices=forward_batch.req_pool_indices,
                plan=forward_batch.kv_loc_plan,
            )
""",
    """        if forward_batch.forward_mode.is_decode_or_idle():
            _dfa3 = (
                DECODE_FA3
                and forward_batch.spec_info is None
                and forward_batch.forward_mode.is_decode()
                and forward_batch.batch_size > 0
            )
            if _dfa3:
                self._verify_fa3_plan(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.batch_size,
                    None,
                    decode=True,
                )
            if not _dfa3 or VERIFY_FA3_CHECK:
                self.indices_updater_decode.update(
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_cpu,
                    forward_batch.seq_lens_sum,
                    decode_wrappers=self.decode_wrappers,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=forward_batch.spec_info,
                    fixed_split_size=self.decode_split_tile_size,
                    disable_split_kv=False,
                    req_pool_indices=forward_batch.req_pool_indices,
                    plan=forward_batch.kv_loc_plan,
                )
""",
)

# Graph decode metadata. The capture-time fast_decode_plan swap below needs the initial plan, so at capture the
# wrapper is always planned; at replay it is skipped when FA3 serves the step.
sub(
    """            self.indices_updater_decode.update(
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                decode_wrappers=self.decode_cuda_graph_metadata[bs],
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
                fixed_split_size=None,
                disable_split_kv=self.disable_cuda_graph_kv_split,
                req_pool_indices=req_pool_indices,
                plan=forward_batch.kv_loc_plan,
            )
""",
    """            _dfa3 = DECODE_FA3 and spec_info is None and forward_mode.is_decode() and bs > 0
            if _dfa3:
                self._verify_fa3_plan(req_pool_indices, seq_lens, bs, None, decode=True)
            if not _dfa3 or VERIFY_FA3_CHECK or in_capture:
                self.indices_updater_decode.update(
                    seq_lens[:bs],
                    seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                    seq_lens_sum,
                    decode_wrappers=self.decode_cuda_graph_metadata[bs],
                    encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                    spec_info=spec_info,
                    fixed_split_size=None,
                    disable_split_kv=self.disable_cuda_graph_kv_split,
                    req_pool_indices=req_pool_indices,
                    plan=forward_batch.kv_loc_plan,
                )
""",
)

sub(
    """        # Call the wrapped function
        o = decode_wrapper.forward(""",
    """        if self._vf3 is not None and self._vf3.get("decode"):
            o = self._verify_fa3_forward(q, layer, True, layer.logit_cap)
            if VERIFY_FA3_CHECK and not torch.cuda.is_current_stream_capturing():
                ref = decode_wrapper.forward(
                    q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    kv_cache,
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                    k_scale=layer.k_scale_float,
                    v_scale=layer.v_scale_float,
                ).view(o.shape)
                err = ((o.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-6)).item()
                self.verify_fa3_worst = max(getattr(self, "verify_fa3_worst", 0.0), err)
                if err > FUSED_SPLIT_TOL:
                    logger.error("DECODE_FA3_CHECK FAIL layer=%d err=%.5f bs=%d", layer.layer_id, err, self._vf3["bs"])
                elif self.verify_fa3_calls % 500 == 1:
                    logger.info(
                        "DECODE_FA3_CHECK ok layer=%d err=%.5f worst=%.5f", layer.layer_id, err, self.verify_fa3_worst
                    )
            return o.view(-1, layer.tp_q_head_num * layer.head_dim)

        # Call the wrapped function
        o = decode_wrapper.forward(""",
)
open(p, "w").write(s)
print("applied", p)
