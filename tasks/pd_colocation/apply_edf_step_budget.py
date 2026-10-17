"""Step-time budget on the per-step prefill chunk (4-mode MTP schedule), on top of apply_edf_deadline_chunk.py.

On the H200 best setting (deadline chunk 1024/2048) chat ITL is the mean step interval, and the intervals that set it
are the mixed steps whose prefill chunk sits on a deep context: 16% of chat ITL time is in intervals > 80 ms
(mean 110 ms, a 2048 chunk over a 32-64K prefix), 21% in 40-80 ms. A fixed token chunk makes the step length grow with
the prefix the chunk attends to. SGLANG_EDF_STEP_BUDGET="budget_ms:a:b" sizes the chunk to a step time instead: the
largest chunk c with a*c + b*c*(prefix + c/2) <= budget_ms (linear layers + causal attention over the chunk's
prefix), for the request this step prefills (the in-progress chunked request, else the earliest-deadline waiting
one); never below the deadline rule's small chunk and never above the chunk it chose. With a large deadline chunk this
gives large chunks while a cold prompt's prefix is short and smaller ones as it deepens.
  python apply_edf_step_budget.py <tree>/python/sglang/srt
"""
import sys

p = f"{sys.argv[1]}/managers/scheduler.py"
s = open(p).read()
old = '''        chunked_prefill_size = self._edf_deadline_chunk(running_batch, chunked_prefill_size)
'''
new = '''        chunked_prefill_size = self._edf_deadline_chunk(running_batch, chunked_prefill_size)
        chunked_prefill_size = self._edf_step_budget_chunk(running_batch, chunked_prefill_size)
'''
old2 = '''    def _edf_deadline_chunk(self, running_batch, chunk_size):'''
new2 = '''    def _edf_step_budget_chunk(self, running_batch, chunk_size):
        """Largest chunk whose predicted mixed-step time over the next prefill's prefix fits the step budget."""
        spec = os.environ.get("SGLANG_EDF_STEP_BUDGET", "")
        if not spec or not chunk_size or running_batch is None or not running_batch.reqs:
            return chunk_size
        budget, a, b = (float(x) for x in spec.split(":"))
        dl = os.environ.get("SGLANG_EDF_DEADLINE_CHUNK", "")
        floor = int(float(dl.split(":")[0])) if dl else 512
        if self.chunked_req is not None:
            # Tokens of this prompt already in KV (prefix_len is refreshed at prefix match; the committed length
            # covers chunks computed since).
            r = self.chunked_req
            prefix = max(r.prefix_len, r.kv.kv_committed_len if getattr(r, "kv", None) is not None else 0)
        else:
            waiting = [r for r in self.waiting_queue if r.priority is not None]
            if not waiting:
                return chunk_size
            prefix = getattr(min(waiting, key=lambda r: r.priority), "_edf_cached_len", 0)
        # a*c + b*c*prefix + (b/2)*c^2 = budget
        qa, qb = b / 2, a + b * prefix
        c = (budget / qb) if qa == 0 else ((-qb + (qb * qb + 4 * qa * budget) ** 0.5) / (2 * qa))
        out = int(min(chunk_size, max(floor, c)))
        self.edf_step_budget_calls = getattr(self, "edf_step_budget_calls", 0) + 1
        self.edf_step_budget_capped = getattr(self, "edf_step_budget_capped", 0) + int(out < chunk_size)
        if self.edf_step_budget_calls % 500 == 1 or (out < chunk_size and self.edf_step_budget_capped % 100 == 1):
            logger.info("EDF_STEP_BUDGET calls=%d capped=%d prefix=%d chunk=%d->%d", self.edf_step_budget_calls,
                        self.edf_step_budget_capped, prefix, chunk_size, out)
        return out

    def _edf_deadline_chunk(self, running_batch, chunk_size):'''
if "def _edf_step_budget_chunk" in s:
    print("already")
else:
    assert s.count(old) == 1 and s.count(old2) == 1, (s.count(old), s.count(old2))
    s = s.replace(old, new, 1).replace(old2, new2, 1)
    open(p, "w").write(s)
    print("patched")
