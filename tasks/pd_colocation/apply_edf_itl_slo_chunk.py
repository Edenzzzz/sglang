"""SLO-sized prefill chunk for the colocated step: admit the largest chunk that keeps the running decodes within their
ITL SLO, unless the prefill itself is already past its TTFT deadline (EDF: the earlier deadline wins).

A decode row emits `tokens_per_step` tokens per step (1 without speculation, the acceptance length with MTP), so its
per-token ITL stays within SLO while the step takes <= slo_ms * tokens_per_step. The step is modelled as
base_ms + a*c + b*c*(prefix + c/2): the decode-only step, linear layers over the chunk, and causal attention of the
chunk over its prefix (the same form and coefficients as apply_edf_step_budget.py's model). The chunk is the largest c
within that budget, at least min_chunk, and never above the chunk the deadline rule chose. The request whose chunk is
sized is the in-progress chunked request, else the earliest-deadline waiting one; when its deadline (req.priority,
monotonic seconds) has passed, the deadline rule's chunk is kept. Applied after apply_edf_deadline_chunk.py.

  SGLANG_EDF_ITL_SLO="slo_ms:tokens_per_step:base_ms:a:b:min_chunk"
  python apply_edf_itl_slo_chunk.py <tree>/python/sglang/srt
Logs `EDF_ITL_SLO calls= capped= late= prefix= chunk=a->b` every 500 calls and on every 100th cap.
Claude, 2026-10-19.
"""
import sys

p = f"{sys.argv[1]}/managers/scheduler.py"
s = open(p).read()
if "def _edf_itl_slo_chunk" in s:
    print("already")
    sys.exit(0)
old = "        chunked_prefill_size = self._edf_deadline_chunk(running_batch, chunked_prefill_size)\n"
new = old + "        chunked_prefill_size = self._edf_itl_slo_chunk(running_batch, chunked_prefill_size)\n"
old2 = "    def _edf_deadline_chunk(self, running_batch, chunk_size):"
new2 = '''    def _edf_itl_slo_chunk(self, running_batch, chunk_size):
        """Largest chunk whose predicted step keeps the running decodes within their ITL SLO."""
        spec = os.environ.get("SGLANG_EDF_ITL_SLO", "")
        if not spec or not chunk_size or running_batch is None or not running_batch.reqs:
            return chunk_size
        slo_ms, tps, base_ms, a, b, min_chunk = (float(x) for x in spec.split(":"))
        if self.chunked_req is not None:
            r = self.chunked_req
            prefix = max(r.prefix_len, r.kv.kv_committed_len if getattr(r, "kv", None) is not None else 0)
        else:
            waiting = [w for w in self.waiting_queue if w.priority is not None]
            if not waiting:
                return chunk_size
            r = min(waiting, key=lambda w: w.priority)
            prefix = getattr(r, "_edf_cached_len", 0)
        self.edf_itl_slo_calls = getattr(self, "edf_itl_slo_calls", 0) + 1
        late = r.priority is not None and r.priority < time.monotonic()
        if late:
            self.edf_itl_slo_late = getattr(self, "edf_itl_slo_late", 0) + 1
            return chunk_size
        budget = slo_ms * tps - base_ms
        if budget <= 0:
            c = min_chunk
        else:
            # a*c + b*c*prefix + (b/2)*c^2 = budget
            qa, qb = b / 2, a + b * prefix
            c = (budget / qb) if qa == 0 else ((-qb + (qb * qb + 4 * qa * budget) ** 0.5) / (2 * qa))
        out = int(min(chunk_size, max(min_chunk, c)))
        self.edf_itl_slo_capped = getattr(self, "edf_itl_slo_capped", 0) + int(out < chunk_size)
        if self.edf_itl_slo_calls % 500 == 1 or (out < chunk_size and self.edf_itl_slo_capped % 100 == 1):
            logger.info("EDF_ITL_SLO calls=%d capped=%d late=%d prefix=%d chunk=%d->%d", self.edf_itl_slo_calls,
                        self.edf_itl_slo_capped, getattr(self, "edf_itl_slo_late", 0), prefix, chunk_size, out)
        return out

''' + old2
assert s.count(old) == 1 and s.count(old2) == 1, (s.count(old), s.count(old2))
s = s.replace(old, new, 1).replace(old2, new2, 1)
open(p, "w").write(s)
print("patched")
