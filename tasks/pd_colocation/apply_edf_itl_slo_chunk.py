"""SLO-sized prefill chunk for the colocated step: admit the largest chunk that keeps the running decodes within their
ITL SLO. No exception: a late prefill does not get a larger chunk, and there is no minimum chunk above the budget.

A decode row emits `tokens_per_step` tokens per step (1 without speculation, the acceptance length with MTP), so its
per-token ITL stays within SLO while the step takes <= slo_ms * tokens_per_step. The step is modelled as
base_ms + a*c + b*c*(prefix + c/2): the decode-only step, linear layers over the chunk, and causal attention of the
chunk over its prefix (the same form and coefficients as apply_edf_step_budget.py's model). The chunk is the largest c
within that budget and never above the chunk the deadline rule chose. When less than one KV page fits (or the decode
step alone fills the budget) the step admits no prefill: it returns 0, which get_new_batch_prefill turns into "no
prefill batch this step" (a sub-page chunk rounds down to a zero-token extend, which flashinfer's planner rejects:
kv_len[0]-1 should be non-negative, 2026-10-20 H20). Under mixed chunk the scheduler charges one token per running
decode row against the chunk, so the returned chunk is the prefill cap plus the decode rows; returning the bare cap
left a chunk smaller than the decode rows, a zero-token extend for the in-progress chunked request and the same
planner abort. The request whose chunk is sized is the in-progress chunked request, else the earliest-deadline
waiting one. Applied after apply_edf_deadline_chunk.py.

  SGLANG_EDF_ITL_SLO="slo_ms:tokens_per_step:base_ms:a:b"            fixed decode-step base
  SGLANG_EDF_ITL_SLO="slo_ms:tokens_per_step:c0:c_bs:c_kv:a:b"       live base c0 + c_bs*rows + c_kv*KV/1e3 from the
                                                                     running batch (the router's step model)
  a trailing ":measured" on either form: the decode base is this node's measured decode-step time (EWMA over the decode-step
                                  cadence sglang already accumulates in decode_moment_totals); the model base is used
                                  only until the first decode samples arrive. The grid-fit model over-predicted H20
                                  fan-out decode steps ~2-3x (shared-prefix rows), capping prefill to ~10 tokens.
  python apply_edf_itl_slo_chunk.py <tree>/python/sglang/srt
Logs `EDF_ITL_SLO calls= capped= prefix= chunk=a->b` every 500 calls and on every 100th cap.
Claude, 2026-10-19.
"""
import sys

p = f"{sys.argv[1]}/managers/scheduler.py"
s = open(p).read()
OLD_CLAMPS = [
    "        out = int(min(chunk_size, max(1, c)))\n",
    ("        out = int(min(chunk_size, c))\n"
     "        if out < max(1, self.page_size):\n"
     "            # Less than one page fits beside the running decodes: no prefill this step.\n"
     "            out = 0\n"),
]
NEW_CLAMP = (
    "        rows = len(running_batch.reqs) if self.is_mixed_chunk else 0\n"
    "        out = int(min(chunk_size - rows, c))\n"
    "        if out < max(1, self.page_size):\n"
    "            # Less than one page fits beside the running decodes: no prefill this step.\n"
    "            out = 0\n"
    "        elif self.is_mixed_chunk:\n"
    "            # Mixed chunk charges one token per running decode row against the chunk (PrefillAdder subtracts\n"
    "            # num_mixed_decode_tokens), so the prefill share is what is left: add the rows back so prefill gets c.\n"
    "            out += rows\n"
)
old = "        chunked_prefill_size = self._edf_deadline_chunk(running_batch, chunked_prefill_size)\n"
new = old + "        chunked_prefill_size = self._edf_itl_slo_chunk(running_batch, chunked_prefill_size)\n"
old2 = "    def _edf_deadline_chunk(self, running_batch, chunk_size):"
new2 = '''    def _edf_measured_decode_ms(self):
        """EWMA of this node's measured decode-step time (ms) from the decode_moment_totals deltas since the last
        call; None unless SGLANG_EDF_ITL_SLO ends in ":measured" and a decode step has been timed."""
        if not os.environ.get("SGLANG_EDF_ITL_SLO", "").endswith(":measured"):
            return None
        totals = getattr(self, "decode_moment_totals", None)
        if not totals:
            return None
        last = getattr(self, "_edf_dm_last", None)
        self._edf_dm_last = (totals[0], totals[2])
        if last is not None and totals[0] > last[0]:
            step_ms = (totals[2] - last[1]) / (totals[0] - last[0]) / 1000.0
            prev = getattr(self, "_edf_dm_ewma", None)
            self._edf_dm_ewma = step_ms if prev is None else 0.8 * prev + 0.2 * step_ms
        return getattr(self, "_edf_dm_ewma", None)

    def _edf_itl_slo_chunk(self, running_batch, chunk_size):
        """Largest chunk whose predicted step keeps the running decodes within their ITL SLO."""
        spec = os.environ.get("SGLANG_EDF_ITL_SLO", "")
        if not spec or not chunk_size or running_batch is None or not running_batch.reqs:
            return chunk_size
        vals = [float(x) for x in spec.split(":") if x != "measured"]
        if len(vals) == 7:
            slo_ms, tps, c0, c_bs, c_kv, a, b = vals
            reqs = running_batch.reqs
            base_ms = c0 + c_bs * len(reqs) + c_kv * sum(
                len(q.origin_input_ids) + len(q.output_ids) for q in reqs) / 1e3
        else:
            slo_ms, tps, base_ms, a, b = vals
        measured = self._edf_measured_decode_ms()
        if measured is not None:
            base_ms = measured
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
        budget = slo_ms * tps - base_ms
        if budget <= 0:
            c = 0
        else:
            # a*c + b*c*prefix + (b/2)*c^2 = budget
            qa, qb = b / 2, a + b * prefix
            c = (budget / qb) if qa == 0 else ((-qb + (qb * qb + 4 * qa * budget) ** 0.5) / (2 * qa))
        rows = len(running_batch.reqs) if self.is_mixed_chunk else 0
        out = int(min(chunk_size - rows, c))
        if out < max(1, self.page_size):
            # Less than one page fits beside the running decodes: no prefill this step.
            out = 0
        elif self.is_mixed_chunk:
            # Mixed chunk charges one token per running decode row against the chunk (PrefillAdder subtracts
            # num_mixed_decode_tokens), so the prefill share is what is left: add the rows back so prefill gets c.
            out += rows
        self.edf_itl_slo_capped = getattr(self, "edf_itl_slo_capped", 0) + int(out < chunk_size)
        if self.edf_itl_slo_calls % 500 == 1 or (out < chunk_size and self.edf_itl_slo_capped % 100 == 1):
            logger.info("EDF_ITL_SLO calls=%d capped=%d prefix=%d chunk=%d->%d", self.edf_itl_slo_calls,
                        self.edf_itl_slo_capped, prefix, chunk_size, out)
        return out

''' + old2
MEASURED_HELPER_ANCHOR = "    def _edf_itl_slo_chunk(self, running_batch, chunk_size):\n"
MEASURED_OVERRIDE_OLD = "        else:\n            slo_ms, tps, base_ms, a, b = vals\n"
MEASURED_OVERRIDE_NEW = MEASURED_OVERRIDE_OLD + (
    "        measured = self._edf_measured_decode_ms()\n"
    "        if measured is not None:\n"
    "            base_ms = measured\n")
ENV_FORM = 'if os.environ.get("SGLANG_EDF_ITL_SLO_MEASURED") != "1":'
SUFFIX_FORM = 'if not os.environ.get("SGLANG_EDF_ITL_SLO", "").endswith(":measured"):'
VALS_OLD = 'vals = [float(x) for x in spec.split(":")]'
VALS_NEW = 'vals = [float(x) for x in spec.split(":") if x != "measured"]'
if ENV_FORM in s:
    s = s.replace(ENV_FORM, SUFFIX_FORM).replace(VALS_OLD, VALS_NEW)
    open(p, "w").write(s)
    print("measured switch -> ':measured' suffix")
if "def _edf_itl_slo_chunk" in s and "def _edf_measured_decode_ms" not in s:
    helper = new2[: new2.index("    def _edf_itl_slo_chunk(self, running_batch, chunk_size):")]
    assert s.count(MEASURED_HELPER_ANCHOR) == 1 and s.count(MEASURED_OVERRIDE_OLD) == 1
    s = s.replace(MEASURED_HELPER_ANCHOR, helper + MEASURED_HELPER_ANCHOR).replace(
        MEASURED_OVERRIDE_OLD, MEASURED_OVERRIDE_NEW)
    open(p, "w").write(s)
    print("measured base added")
if "def _edf_itl_slo_chunk" in s:
    if NEW_CLAMP in s:
        print("already")
    else:
        old = [o for o in OLD_CLAMPS if s.count(o) == 1 and s.count(NEW_CLAMP) == 0]
        assert old, "no known clamp to upgrade"
        open(p, "w").write(s.replace(old[-1], NEW_CLAMP))
        print("upgraded")
    sys.exit(0)
assert s.count(old) == 1 and s.count(old2) == 1, (s.count(old), s.count(old2))
s = s.replace(old, new, 1).replace(old2, new2, 1)
open(p, "w").write(s)
print("patched")
