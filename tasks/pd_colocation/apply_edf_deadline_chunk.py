"""Per-step prefill chunk from EDF deadlines for the 4-mode MTP schedule.

On 8 H20s the 4-mode step's agent pair moved along a frontier with the static chunk size (2048: TTFT -33% / ITL -5%;
512: -21% / -45% vs stock + MTP): small chunks shorten every mixed step that decoding requests wait through, large
chunks finish cold agent prefills sooner. SGLANG_EDF_DEADLINE_CHUNK="small:large:tok_s": each step uses the small
chunk unless the in-progress chunked request or a waiting request (deadline order, cumulative uncached tokens) would
miss its TTFT deadline at tok_s -- the prefill rate the small chunk sustains -- in which case it uses the large one.
  python apply_edf_deadline_chunk.py <tree>/python/sglang/srt
"""
import sys

srt = sys.argv[1]
p = f"{srt}/managers/scheduler.py"
s = open(p).read()
old = '''        chunked_prefill_size = self._edf_limit_chunk(running_batch, chunked_prefill_size)
        if chunked_prefill_size == 0:'''
new = '''        chunked_prefill_size = self._edf_deadline_chunk(running_batch, chunked_prefill_size)
        chunked_prefill_size = self._edf_limit_chunk(running_batch, chunked_prefill_size)
        if chunked_prefill_size == 0:'''
assert s.count(old) == 1 or new in s
if new not in s:
    s = s.replace(old, new, 1)
    old2 = '''    def _edf_rescore_and_park(self):'''
    new2 = '''    def _edf_deadline_chunk(self, running_batch, chunk_size):
        """Small chunk unless the deadline-ordered prefill backlog would miss a TTFT deadline at its pace."""
        spec = os.environ.get("SGLANG_EDF_DEADLINE_CHUNK", "")
        if not spec or not self.edf_enabled or not chunk_size or running_batch is None or not running_batch.reqs:
            return chunk_size
        small, large, tok_s = (float(x) for x in spec.split(":"))
        now = time.monotonic()
        tokens = 0
        late = False
        if self.chunked_req is not None:
            r = self.chunked_req
            tokens += max(0, len(r.origin_input_ids) - r.prefix_len)
            late = r.priority is not None and r.priority < now + tokens / tok_s
        if not late:
            for r in sorted((r for r in self.waiting_queue if r.priority is not None), key=lambda r: r.priority):
                tokens += max(0, len(r.origin_input_ids) - getattr(r, "_edf_cached_len", 0))
                if r.priority < now + tokens / tok_s:
                    late = True
                    break
        self.edf_deadline_chunk_calls = getattr(self, "edf_deadline_chunk_calls", 0) + 1
        self.edf_deadline_chunk_large = getattr(self, "edf_deadline_chunk_large", 0) + int(late)
        if self.edf_deadline_chunk_calls % 500 == 1:
            logger.info("EDF_DEADLINE_CHUNK calls=%d large=%d backlog=%d", self.edf_deadline_chunk_calls,
                        self.edf_deadline_chunk_large, tokens)
        return int(min(chunk_size, large if late else small))

    def _edf_rescore_and_park(self):'''
    assert s.count(old2) == 1
    s = s.replace(old2, new2, 1)
    open(p, "w").write(s)
    print("patched")
else:
    print("already")
