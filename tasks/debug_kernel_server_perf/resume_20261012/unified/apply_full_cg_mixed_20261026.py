"""SGLANG_FULL_CG_MIXED=1: allow the full prefill CUDA-graph backend with --enable-mixed-chunk (upstream asserts the
breakable backend). A mixed batch's decode rows are 1-token extends with a prefix, the same attention shape as prefill
rows, so a full graph captured per token bucket with request-slot padding can replay prefill, mixed and 4-mode merged
steps alike. Step 1 of making mixed steps replay full graphs: lift the gate (opt-in), then fix what breaks. Idempotent.
  python apply_full_cg_mixed_20261026.py <run root>
Claude, 2026-10-26."""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py"
s = p.read_text()
if "SGLANG_FULL_CG_MIXED" not in s:
    old = """            assert backend == Backend.BREAKABLE, ("""
    new = """            assert backend == Backend.BREAKABLE or (
                backend == Backend.FULL and __import__("os").environ.get("SGLANG_FULL_CG_MIXED") == "1"
            ), ("""
    assert s.count(old) == 1, "layout changed"
    s = s.replace(old, new)
    # The merged 4-mode step keeps each chunk row's last token plus EVERY verify token (logits_processor,
    # mixed_verify_tokens), so the static logits buffer needs batch_size - verify_rows + verify_tokens rows, not one per
    # request; with one per request the verify rows' logits were truncated (eagle_sample: shape [5, 4] for 5 values).
    old = """    def _prefill_logits_buffer_rows(self, forward_batch: ForwardBatch) -> int:
        if not forward_batch.return_logprob:
            return forward_batch.batch_size"""
    new = """    def _prefill_logits_buffer_rows(self, forward_batch: ForwardBatch) -> int:
        mvt = int(getattr(forward_batch, "mixed_verify_tokens", 0) or 0)
        if mvt and not forward_batch.return_logprob:
            covered = verify_rows = 0
            for length in reversed(forward_batch.extend_seq_lens_cpu):
                if covered >= mvt:
                    break
                covered += int(length)
                verify_rows += 1
            assert covered == mvt, "verify rows must be the batch's trailing rows"
            return forward_batch.batch_size - verify_rows + mvt
        if not forward_batch.return_logprob:
            return forward_batch.batch_size"""
    assert s.count(old) == 1, "logits-buffer layout changed"
    s = s.replace(old, new)
    # ... and the replay output must keep those rows: the full backend trimmed next_token_logits to one row per
    # request (raw_bs), cutting a merged step's verify logits to one per verify row (shape [4, 4] for 4 values).
    old = """        bs = forward_batch.batch_size
        self.raw_bs = bs
"""
    new = """        bs = forward_batch.batch_size
        self.raw_bs = bs
        self.raw_logits_rows = (
            self._prefill_logits_buffer_rows(forward_batch)
            if int(getattr(forward_batch, "mixed_verify_tokens", 0) or 0)
            and not forward_batch.return_logprob
            else bs
        )
"""
    assert s.count(old) == 1, "load_batch layout changed"
    s = s.replace(old, new)
    old = """        logits_rows = self.raw_bs if self._is_full_backend else self.raw_num_tokens"""
    new = """        logits_rows = (
            getattr(self, "raw_logits_rows", self.raw_bs) if self._is_full_backend else self.raw_num_tokens
        )"""
    assert s.count(old) == 1, "trim layout changed"
    s = s.replace(old, new)
    compile(s, str(p), "exec")
    p.write_text(s)
print("FULL_CG_MIXED", "SGLANG_FULL_CG_MIXED" in p.read_text())
