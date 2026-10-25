"""SGLANG_MIXED_SPEC_QUEUE_LONG_ROWS=1: the waiting-queue trigger of the 4-mode mixed step (SGLANG_MIXED_SPEC_MIN_QUEUE)
applies only when some running request's context is >= SGLANG_MIXED_SPEC_MIN_CTX. A short prefill co-scheduled with
short decode / verify rows saves no long-row attention, and the mixed step runs eager (the prefill CUDA-graph runner
refuses MIXED batches): on ShareGPT at 32 req/s that cost 7% throughput and grew the queue (114 vs 3). With long agent
rows running, the trigger is worth keeping (GLM-4.5-Air overloaded four-mode: +28% throughput with it, +14% without).
Off by default. Idempotent.
  python apply_mixed_spec_queue_long_20261026.py <run root>
Claude, 2026-10-26."""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "python/sglang/srt/managers/scheduler.py"
s = p.read_text()
if "SGLANG_MIXED_SPEC_QUEUE_LONG_ROWS" not in s:
    old = """            < MIXED_SPEC_MIN_CTX
            and len(self.waiting_queue) < MIXED_SPEC_MIN_QUEUE
        ):"""
    new = """            < MIXED_SPEC_MIN_CTX
            and (
                len(self.waiting_queue) < MIXED_SPEC_MIN_QUEUE
                # SGLANG_MIXED_SPEC_QUEUE_LONG_ROWS=1: the backlog trigger needs a long-context running row; with only
                # short rows the merged step saves no long-row attention and loses the prefill CUDA graph.
                or (
                    os.environ.get("SGLANG_MIXED_SPEC_QUEUE_LONG_ROWS") == "1"
                    and max(
                        (len(r.origin_input_ids) + len(r.output_ids) for r in running_batch.reqs), default=0
                    )
                    < MIXED_SPEC_MIN_CTX
                )
            )
        ):"""
    assert s.count(old) == 1, "4-mode gate layout changed"
    s = s.replace(old, new)
    p.write_text(s)
print("QUEUE_LONG_ROWS", "SGLANG_MIXED_SPEC_QUEUE_LONG_ROWS" in p.read_text())
