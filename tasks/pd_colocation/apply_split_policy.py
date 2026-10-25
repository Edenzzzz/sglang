"""SGLANG_SPLIT_POLICY=1: per extend step, split or not, and every extend step on FA3.

serving_mix_corun_20261021.py (H200, serving-shaped mixed steps vs ONE varlen FA3 call over all rows, stock sglang's
Hopper default):
  * a short chunk (512-1024 tokens) beside long-context decode / verify rows: split -20..-77%, co-run on shared SMs
    -22..-77% -- one call sized for the chunk leaves each long row on a single serial pass;
  * a long chunk (2048-4096 @ 32K): split 0..+21%, co-run 0..+11% -- the rows hide under the chunk and the second
    launch only adds;
  * rows on a reserved SM partition lose whenever the rows are long (rows on 8 SMs: +20..+158%), and sharing all SMs
    (margin 0) matches or beats the best partition in 15 of 20 steps.
So the step is split (chunk on FA3, rows on FA3, co-run on shared SMs under SPLIT_OVERLAP) only when the chunk's
predicted FA3 time is below SGLANG_SPLIT_SINGLE_CHUNK_MS (default 1.0 ms per layer; same cost constants as
apply_split_auto_margin.py); otherwise -- and for every one-sided step, which before this went to flashinfer's
ragged + paged prefill -- the whole step is one FA3 call, as stock runs it. SPLIT_POLICY counters log both kinds.
  python apply_split_policy.py <tree>/python/sglang/srt     (after apply_split_rows_fa3.py, apply_split_auto_margin.py)
Claude, 2026-10-21.
"""
import sys

p = f"{sys.argv[1]}/layers/attention/flashinfer_backend.py"
s = open(p).read()
assert "SPLIT_POLICY" not in s, "already applied"
assert "_fs_auto_margin" in s, "apply_split_auto_margin.py first"


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:120])
    s = s.replace(old, new)


sub(
    'SPLIT_MARGIN_AUTO = os.environ.get("SGLANG_SPLIT_MARGIN_AUTO", "0") == "1"\n',
    'SPLIT_MARGIN_AUTO = os.environ.get("SGLANG_SPLIT_MARGIN_AUTO", "0") == "1"\n'
    "# Per-step split-or-one-call policy; every extend step on FA3 (see apply_split_policy.py).\n"
    'SPLIT_POLICY = os.environ.get("SGLANG_SPLIT_POLICY", "0") == "1"\n'
    'SPLIT_SINGLE_CHUNK_MS = float(os.environ.get("SGLANG_SPLIT_SINGLE_CHUNK_MS", "1.0"))\n',
)

# Decide before the one-sided early return.
sub(
    """        fa3_rows, fa2_rows = np.nonzero(mask)[0], np.nonzero(~mask)[0]
        self.fused_split_plans += 1
""",
    """        fa3_rows, fa2_rows = np.nonzero(mask)[0], np.nonzero(~mask)[0]
        self.fused_split_plans += 1
        _single = False
        if SPLIT_POLICY:
            assert SPLIT_ROWS_FA3, "SGLANG_SPLIT_POLICY needs SGLANG_SPLIT_ROWS_FA3"
            _single = (
                len(fa3_rows) == 0
                or len(fa2_rows) == 0
                or self._fs_chunk_ms(forward_batch, fa3_rows) >= SPLIT_SINGLE_CHUNK_MS
            )
            if _single:
                fa3_rows, fa2_rows = np.arange(len(mask)), np.arange(0)
            self.split_policy_n = getattr(self, "split_policy_n", [0, 0])
            self.split_policy_n[int(_single)] += 1
            if sum(self.split_policy_n) % 500 == 1:
                logger.info(
                    "SPLIT_POLICY split=%d one_call=%d", self.split_policy_n[0], self.split_policy_n[1]
                )
""",
)

# One-call steps skip the one-sided early return.
sub(
    """        if len(fa3_rows) == 0 or len(fa2_rows) == 0:
            # One-sided partition""",
    """        if (len(fa3_rows) == 0 or len(fa2_rows) == 0) and not _single:
            # One-sided partition""",
)

# The single plan: FA3 over every row, built after the FA3 side's page table.
sub(
    """        n2 = len(fa2_rows)
        if SPLIT_ROWS_FA3:""",
    """        n2 = len(fa2_rows)
        if _single:
            self.fused_split_plan = dict(
                single=True,
                fa3_span=(0, _tok_total),
                fa2_span=None,
                tok_total=_tok_total,
                fa3_tok=None,
                fa2_tok=None,
                cu_q=indptr("fa3_cu_q", qo_lens[fa3_rows]),
                max_q=int(qo_lens[fa3_rows].max()),
                page_table=page_table,
                cache_seqlens=self._fs_upload(
                    "fa3_kvlen", seq_lens_cpu[fa3_rows].numpy().astype(np.int32), torch.int32, dev
                ),
                n_fa3=len(fa3_rows),
                n_fa2=0,
            )
            return True
        if SPLIT_ROWS_FA3:""",
)

sub(
    """    def _fs_auto_margin(self, qo_lens, seq_lens_cpu, fa3_rows, fa2_rows, dev):""",
    """    def _fs_chunk_ms(self, forward_batch, fa3_rows):
        \"\"\"Predicted FA3 ms per layer of the FA3-class rows (the prefill chunk): causal FLOPs / peak.\"\"\"
        import numpy as np

        if getattr(self, "_fs_cost", None) is None:
            props = torch.cuda.get_device_properties(forward_batch.seq_lens.device)
            h20 = "H20" in props.name
            self._fs_cost = dict(
                nsm=props.multi_processor_count,
                tflops=float(os.environ.get("SGLANG_SPLIT_FA3_TFLOPS", "120" if h20 else "600")) * 1e12,
                tbps=float(os.environ.get("SGLANG_SPLIT_KV_TBPS", "3.6" if h20 else "4.0")) * 1e12,
                scale=float(os.environ.get("SGLANG_SPLIT_ROWS_SM_SCALE", "0.75")),
            )
        iu = self.indices_updater_prefill
        q = np.asarray(forward_batch.extend_seq_lens_cpu, dtype=np.int64)[fa3_rows]
        k = forward_batch.seq_lens_cpu.numpy().astype(np.int64)[fa3_rows]
        flops = float((4 * iu.num_qo_heads * iu.head_dim * q * ((k - q) + (q + 1) / 2)).sum())
        return flops / self._fs_cost["tflops"] * 1e3

    def _fs_auto_margin(self, qo_lens, seq_lens_cpu, fa3_rows, fa2_rows, dev):""",
)

# Dispatch: the single plan is one FA3 call written straight into the output.
sub(
    """        _s3, _s2 = pl.get("fa3_span"), pl.get("fa2_span")
        _use_views = _s3 is not None and _s2 is not None
""",
    """        _s3, _s2 = pl.get("fa3_span"), pl.get("fa2_span")
        _use_views = _s3 is not None and (_s2 is not None or pl.get("single", False))
""",
)
sub(
    """        _overlap = (
            SPLIT_OVERLAP
            and not pl.get("no_fa2_wrapper")""",
    """        _overlap = (
            SPLIT_OVERLAP
            and not pl.get("single", False)
            and not pl.get("no_fa2_wrapper")""",
)
sub(
    """        if pl.get("no_fa2_wrapper"):
            if ref_wrapper is None or kv_cache is None:""",
    """        if pl.get("single"):
            self.split_single_calls = getattr(self, "split_single_calls", 0) + 1
            if FUSED_SPLIT_VERIFY and ref_wrapper is not None and kv_cache is not None:
                ref = ref_wrapper.forward(
                    qv, kv_cache, causal=causal, sm_scale=layer.scaling, logits_soft_cap=logits_soft_cap,
                    k_scale=layer.k_scale_float, v_scale=layer.v_scale_float,
                ).view(-1, hq, hd)
                err = ((o - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
                self.fused_split_err = max(self.fused_split_err, err)
                if err > FUSED_SPLIT_TOL:
                    logger.error("SPLIT_POLICY one_call VERIFY FAIL layer=%d err=%.5f", layer.layer_id, err)
                elif self.split_single_calls % 500 == 1:
                    logger.info("SPLIT_POLICY one_call VERIFY ok err=%.5f worst=%.5f", err, self.fused_split_err)
            return o

        if pl.get("no_fa2_wrapper"):
            if ref_wrapper is None or kv_cache is None:""",
)
open(p, "w").write(s)
print("applied", p)
