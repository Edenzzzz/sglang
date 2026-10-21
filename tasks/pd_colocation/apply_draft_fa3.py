"""SGLANG_DRAFT_FA3=1: the draft model's draft-extend steps (EAGLE spec v2, DRAFT_EXTEND_V2) run attention on FA3
instead of the flashinfer BatchPrefill wrapper.

Stock prefill-prioritized + MTP on its own FA3 backend serves 161 tok/s agent throughput (H200, chat 8 req/s); the same
schedule on our backend -- every target attention call already on FA3 -- 142.5 (`_h200_pp_c8_ct_ourbe`,
h200_mtp_gap_20261021.sh). What still runs flashinfer there is the draft model, and its draft extend is BatchPrefill
over num_tokens_per_req query rows per request at full context: the decode-shaped rows on which BatchPrefill is ~8x
slower than FA3 (rows_layout_check_20261021.py), once per verify cycle.

Layout (eagle_info.generate_attn_arg_prefill, draft extend): a constant num_tokens_per_req queries per request (required
for cuda-graph capture), KV length = the seq_lens handed to the planner (they already include the draft-extend window),
no custom mask -> FA3 causal over cache_seqlens = seq_lens. Planned in init_forward_metadata_out_graph (capture and
every replay) into the persistent buffers of apply_verify_fa3.py; checked under SGLANG_VERIFY_FA3_CHECK=1 (graphs off).
  python apply_draft_fa3.py <tree>/python/sglang/srt      (after apply_verify_fa3.py and apply_decode_fa3.py)
Claude, 2026-10-21.
"""
import sys

p = f"{sys.argv[1]}/layers/attention/flashinfer_backend.py"
s = open(p).read()
assert "DRAFT_FA3" not in s, "already applied"
assert "decode=decode" in s, "apply_decode_fa3.py first"


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:120])
    s = s.replace(old, new)


sub(
    'DECODE_FA3 = os.environ.get("SGLANG_DECODE_FA3", "0") == "1"\n',
    'DECODE_FA3 = os.environ.get("SGLANG_DECODE_FA3", "0") == "1"\n'
    "# Draft-model draft-extend (spec v2) on FA3; checked under VERIFY_FA3_CHECK.\n"
    'DRAFT_FA3 = os.environ.get("SGLANG_DRAFT_FA3", "0") == "1"\n',
)

sub(
    """    def _verify_fa3_plan(self, req_pool_indices, seq_lens, bs, spec_info, decode=False):""",
    """    def _verify_fa3_plan(self, req_pool_indices, seq_lens, bs, spec_info, decode=False, draft=False):""",
)
sub(
    """        if decode:
            n = 1
        else:""",
    """        if decode:
            n = 1
        elif draft:
            n = int(spec_info.num_tokens_per_req)
        else:""",
)
sub(
    """        b["kvlen"][:bs].copy_(seq_lens[:bs] if decode else seq_lens[:bs] + n)""",
    """        b["kvlen"][:bs].copy_(seq_lens[:bs] if (decode or draft) else seq_lens[:bs] + n)""",
)
sub(
    """        self._vf3 = dict(bs=bs, n=n, decode=decode)""",
    """        self._vf3 = dict(bs=bs, n=n, decode=decode, draft=draft)""",
)

# Graph draft-extend metadata.
sub(
    """        elif forward_mode.is_draft_extend_v2():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.draft_extend_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
                plan=forward_batch.kv_loc_plan,
            )
""",
    """        elif forward_mode.is_draft_extend_v2():
            _dr = DRAFT_FA3 and bs > 0
            if _dr:
                self._verify_fa3_plan(req_pool_indices, seq_lens, bs, spec_info, draft=True)
            # The capture-time fast_prefill_plan swap below needs the initial plan.
            if not _dr or VERIFY_FA3_CHECK or in_capture:
                self.indices_updater_prefill.update(
                    req_pool_indices[:bs],
                    seq_lens[:bs],
                    seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                    seq_lens_sum,
                    prefix_lens=None,
                    prefill_wrappers=self.draft_extend_cuda_graph_metadata[bs],
                    use_ragged=False,
                    encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                    spec_info=spec_info,
                    plan=forward_batch.kv_loc_plan,
                )
""",
)

# forward_extend: draft-extend steps take the same FA3 path as verify.
sub(
    """            if (
                o is None
                and self._vf3 is not None
                and forward_batch.forward_mode.is_target_verify()
            ):""",
    """            if (
                o is None
                and self._vf3 is not None
                and (
                    forward_batch.forward_mode.is_target_verify()
                    or (self._vf3.get("draft") and forward_batch.forward_mode.is_draft_extend_v2())
                )
            ):""",
)
open(p, "w").write(s)
print("applied", p)
