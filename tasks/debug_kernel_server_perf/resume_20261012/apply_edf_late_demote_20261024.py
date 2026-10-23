"""Overload handling for the EDF scheduler (2026-10-24): demote prefills that can no longer meet their deadline.

Plain EDF serves the most overdue request first. Under overload (the 32-64K agent burst: every queued agent prefill
already past its deadline) each late request therefore also makes the requests behind it late -- chat arriving
during the burst waited 1.5-6.7 s (p50) behind already-failed agent prefills. With SGLANG_EDF_LATE_DEMOTE=1, a
waiting or chunked prefill whose deadline is earlier than now + (its uncached tokens / SGLANG_EDF_LATE_DEMOTE_TOK_S)
ranks behind every request that can still make its deadline; demoted requests keep deadline order among themselves,
so they finish in EDF order once the feasible work is served. Decodes returning from retraction (output already
started) are never demoted. Patches <root>/python/sglang/srt/managers/scheduler.py in place; idempotent.
  python apply_edf_late_demote_20261024.py <root>
Claude.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "python/sglang/srt/managers/scheduler.py"
s = path.read_text()
if "SGLANG_EDF_LATE_DEMOTE" in s:
    sys.exit(0)
old = """        candidates = self.waiting_queue[:64]
        for req in candidates:
            arrival = getattr(req, "_edf_arrival_monotonic", None)
            if arrival is None or req.priority is None or getattr(req, "_edf_parked", False):
                continue
            match_kv_cache(req, self.tree_cache)
            req.priority = min(req.priority, arrival + compute_ttft_slo_ms(uncached(req)) / 1000.0)

        chunked = self.chunked_req
"""
new = """        candidates = self.waiting_queue[:64]
        for req in candidates:
            arrival = getattr(req, "_edf_arrival_monotonic", None)
            if (arrival is None or req.priority is None or getattr(req, "_edf_parked", False)
                    or getattr(req, "_edf_late", False)):
                continue
            match_kv_cache(req, self.tree_cache)
            req.priority = min(req.priority, arrival + compute_ttft_slo_ms(uncached(req)) / 1000.0)

        if os.environ.get("SGLANG_EDF_LATE_DEMOTE", "0") == "1":
            tok_s = float(os.environ.get("SGLANG_EDF_LATE_DEMOTE_TOK_S", "6000"))
            now = time.monotonic()
            pending = [(r, uncached(r)) for r in candidates]
            if self.chunked_req is not None:
                pending.append((self.chunked_req,
                                max(1, len(self.chunked_req.full_untruncated_fill_ids) - self.chunked_req.prefix_len)))
            for req, need in pending:
                if req.priority is None or req.output_ids or getattr(req, "_edf_late", False):
                    continue
                if req.priority < now + need / tok_s:
                    req._edf_late = True
                    req.priority += EDF_LATE_OFFSET_S
                    self.edf_late_demoted = getattr(self, "edf_late_demoted", 0) + 1
                    if self.edf_late_demoted == 1 or self.edf_late_demoted % 50 == 0:
                        logger.info("EDF_LATE_DEMOTE count=%d need=%d", self.edf_late_demoted, need)

        chunked = self.chunked_req
"""
assert s.count(old) == 1, "rescore anchor not found"
s = s.replace(old, new)
anchor = "class Scheduler("
assert s.count(anchor) == 1
s = s.replace(anchor, "# Added to a demoted request's deadline: later than every feasible deadline, order among demoted kept.\n"
              "EDF_LATE_OFFSET_S = 1.0e6\n\n\n" + anchor)
path.write_text(s)
print("patched", path)
