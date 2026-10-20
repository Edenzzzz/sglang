"""SLO-checked admission for deadline-ordered decode rows (applied after apply_decode_rows_edf.py).

With rows sitting out steps, the running-request cap no longer bounds per-row speed: at cap 96 the resident rows each
ran half the steps and chat TPOT rose to 57.6 ms against its 20 ms SLO (2026-10-20, H200, rep 1). A request is admitted
only while every resident row can still meet its mean-TPOT SLO. A row with SLO s needs a step every s * a / dt steps,
where dt is the measured step interval and a the resident rows' tokens per verify (sum of output tokens / sum of
sglang's per-request spec_verify_ct; 1 without speculation), so it
takes a fraction f = dt / (s * a) of one of the K row slots per step. Admission stops once sum(f) over resident rows
plus the newcomer exceeds util * K; the newcomer is charged as a chat row (the tightest SLO). The running batch at
admission time holds every resident row (the parked ones are merged back first).

  SGLANG_DECODE_ROWS="K:slo_ms:long_tok:long_ms:util"     e.g. 48:20:24000:80:0.9 (util omitted: no admission check)
  python apply_decode_rows_admit.py <tree>/python/sglang/srt
Logs `DECODE_ROWS_ADMIT dt= a= load= budget=` every 500 admission checks that bind.
Claude, 2026-10-20.
"""
import sys

p = f"{sys.argv[1]}/managers/scheduler.py"
s = open(p).read()
assert "_decode_rows_select" in s, "apply_decode_rows_edf.py first"
assert "_decode_rows_budget" not in s, "already applied"

# 1. Measure the step interval and per-row tokens per step at every row selection (one per decode / 4-mode step).
old = """        k, slo_ms, long_tok, long_ms = (float(x) for x in spec.split(":"))
        k = int(k)
        running_batch.filter_batch()
        n = len(running_batch.reqs)
"""
assert s.count(old) == 1
s = s.replace(old, """        k, slo_ms, long_tok, long_ms = (float(x) for x in spec.split(":")[:4])
        k = int(k)
        running_batch.filter_batch()
        n = len(running_batch.reqs)
        self._decode_rows_measure(running_batch)
""")
old = "    def _decode_rows_select(self, running_batch):"
assert s.count(old) == 1
s = s.replace(old, '''    def _decode_rows_measure(self, running_batch):
        """EWMA of the step interval (ms) between row selections (one per decode / 4-mode step)."""
        now = time.monotonic()
        last = getattr(self, "_decode_rows_t", None)
        self._decode_rows_t = now
        if last is not None and now - last < 1.0:
            dt = (now - last) * 1000.0
            ew = getattr(self, "_decode_rows_dt", None)
            self._decode_rows_dt = dt if ew is None else 0.9 * ew + 0.1 * dt

    def _decode_rows_budget(self, running_batch):
        """How many more requests fit with every resident row still within its mean-TPOT SLO; None = no limit."""
        spec = os.environ.get("SGLANG_DECODE_ROWS", "").split(":")
        if len(spec) < 5 or running_batch is None:
            return None
        dt = getattr(self, "_decode_rows_dt", None)
        if dt is None or not running_batch.reqs:
            return None
        ver = sum(r.spec_verify_ct for r in running_batch.reqs)
        a = sum(len(r.output_ids) for r in running_batch.reqs) / ver if ver else 1.0
        k, slo_ms, long_tok, long_ms, util = (float(x) for x in spec[:5])
        load = sum(
            dt / ((long_ms if len(r.origin_input_ids) >= long_tok else slo_ms) * a)
            for r in running_batch.reqs
        )
        budget = max(0, int((util * k - load) / (dt / (slo_ms * a))))
        st = getattr(self, "_decode_rows_admit_stats", 0) + (budget == 0)
        self._decode_rows_admit_stats = st
        if budget == 0 and st % 500 == 1:
            logger.info(
                "DECODE_ROWS_ADMIT dt=%.1f a=%.2f load=%.1f rows=%d budget=0 (x%d)",
                dt, a, load, len(running_batch.reqs), st,
            )
        return budget

''' + old)

# 2. Admission: cap the allocatable count by the SLO budget.
old = """        active_batch = running_batch or self.running_batch
        available = max(
            available - self.beam_coordinator.pending_member_rows(active_batch), 0
        )
        res = min(pp_budget, available)
"""
assert s.count(old) == 1
s = s.replace(old, old + """        slo_budget = self._decode_rows_budget(active_batch)
        if slo_budget is not None:
            res = min(res, slo_budget)
""")
open(p, "w").write(s)
print("applied", p)
