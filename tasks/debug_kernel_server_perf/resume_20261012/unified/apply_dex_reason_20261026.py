"""Diagnostic: SGLANG_PCG_REASON=1 also logs why EAGLEDraftExtendCudaGraphRunner.can_run_graph refuses a batch (per-request
width vs captured width, bs vs max_bs, padding mode), the first 30 refusals and then every 500th. Idempotent.
  python apply_dex_reason_20261026.py <run root>
Claude, 2026-10-26."""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py"
s = p.read_text()
if "DEX_REASON" not in s:
    old = """    def can_run_graph(self, forward_batch: ForwardBatch):
        # Uniform-width replay invariant"""
    new = """    def can_run_graph(self, forward_batch: ForwardBatch):
        ok = self._can_run_graph_impl(forward_batch)
        if not ok and __import__("os").environ.get("SGLANG_PCG_REASON") == "1":
            n = self.__dict__["_dex_no"] = self.__dict__.get("_dex_no", 0) + 1
            if n <= 30 or n % 500 == 0:
                si = forward_batch.spec_info
                __import__("logging").getLogger(__name__).info(
                    "DEX_REASON n=%d width=%s captured=%s bs=%d max_bs=%s disable_padding=%s mode=%s",
                    n, getattr(si, "num_tokens_per_req", None), self.captured_req_width,
                    forward_batch.seq_lens.numel(), self.max_bs, self.disable_padding, forward_batch.forward_mode.name)
        return ok

    def _can_run_graph_impl(self, forward_batch: ForwardBatch):
        # Uniform-width replay invariant"""
    assert s.count(old) == 1, "layout changed"
    s = s.replace(old, new)
    compile(s, str(p), "exec")
    p.write_text(s)
print("DEX_REASON", "DEX_REASON" in p.read_text())
