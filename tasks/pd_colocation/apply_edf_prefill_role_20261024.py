"""Deadline-ordered admission on PD-disaggregation PREFILL workers (2026-10-24), for a stronger disagg baseline.

The EDF scheduler sets a request's priority (its TTFT deadline) only on the colocated path
(DisaggregationMode.NULL); a prefill-role worker queues every request with the default priority, so a
`disagg_*_edf` arm still ran FCFS on its prefill pods -- chat waited behind 64K-token agent prefills. This sets the
same deadline (uncached-token based when SGLANG_EDF_UNCACHED_DEADLINE=1) when a request enters the prefill
bootstrap queue; the waiting queue it later joins is sorted by priority under --enable-priority-scheduling. Order
only: chunk preemption stays off on prefill workers (its KV is being shipped to decode). Inactive unless
SGLANG_EDF_SCHEDULING=1. Patches <tree>/sglang/srt/managers/scheduler.py in place; idempotent.
  python apply_edf_prefill_role_20261024.py <tree python dir>
Claude.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "sglang/srt/managers/scheduler.py"
s = path.read_text()
if "EDF_PREFILL_ROLE_QUEUE" in s:
    print("already patched", path)
    sys.exit(0)
old = """        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._prefetch_kvcache(req)
            self.disagg_prefill_bootstrap_queue.add(
"""
new = """        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._prefetch_kvcache(req)
            if self.edf_enabled:
                from sglang.srt.managers import edf_scheduler as _edf

                if _edf.EDF_UNCACHED_DEADLINE:
                    req._edf_cached_len = self._edf_cached_prefix_len(req)
                _edf.edf_set_priority(req)
                self.edf_queue_calls += 1
                if self.edf_queue_calls == 1 or self.edf_queue_calls % 200 == 0:
                    logger.info("EDF_PREFILL_ROLE_QUEUE calls=%d priority=%.6f", self.edf_queue_calls, req.priority)
            self.disagg_prefill_bootstrap_queue.add(
"""
assert s.count(old) == 1, "prefill-role anchor not found"
path.write_text(s.replace(old, new))
print("patched", path)
