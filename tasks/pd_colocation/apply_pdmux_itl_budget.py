"""pdmux deadline sharing with an ITL term (PD item 5's other half). Builds on apply_pdmux_deadline_share.py.

The deadline rule weighed only TTFT, so it always took the TTFT side of the frontier. The decode loop now keeps an EWMA
of the wall interval between consecutive decode steps (what ITL sees on a non-speculative server);
SGLANG_PDMUX_ITL_BUDGET_MS=B lets a TTFT-late prefill batch share the whole GPU only while that EWMA is below B,
otherwise the batch keeps pdmux's decode-isolating split. 0 = no ITL term.
  python apply_pdmux_itl_budget.py <tree>/python/sglang/srt
"""
import sys

srt = sys.argv[1]
p = f"{srt}/multiplex/multiplexing_mixin.py"
s = open(p).read()


def patch(old, new):
    global s
    if new in s:
        print("already"); return
    assert s.count(old) == 1, (s.count(old), old[:60])
    s = s.replace(old, new, 1); print("patched")


patch('''                    decode_result = self.run_batch(running_batch)''',
'''                    _now = time.perf_counter()
                    _last = getattr(self, "_pdmux_last_decode_t", None)
                    if _last is not None and _now - _last < 1.0:
                        _gap_ms = (_now - _last) * 1e3
                        _ew = getattr(self, "pdmux_itl_ewma_ms", None)
                        self.pdmux_itl_ewma_ms = _gap_ms if _ew is None else 0.9 * _ew + 0.1 * _gap_ms
                    self._pdmux_last_decode_t = _now
                    decode_result = self.run_batch(running_batch)''')
patch('''                if late:
                    stream_idx = 0
                    self.pdmux_shared_batches = getattr(self, "pdmux_shared_batches", 0) + 1''',
'''                itl_budget = float(os.environ.get("SGLANG_PDMUX_ITL_BUDGET_MS", "0"))
                if late and itl_budget > 0 and getattr(self, "pdmux_itl_ewma_ms", 0.0) >= itl_budget:
                    late = False
                    self.pdmux_itl_held = getattr(self, "pdmux_itl_held", 0) + 1
                if late:
                    stream_idx = 0
                    self.pdmux_shared_batches = getattr(self, "pdmux_shared_batches", 0) + 1''')
patch('''                        "PDMUX_SHARE_DEADLINE batches=%d shared=%d tokens=%d split_rate=%.0f",
                        self.pdmux_split_batches,
                        getattr(self, "pdmux_shared_batches", 0),''',
'''                        "PDMUX_SHARE_DEADLINE batches=%d shared=%d itl_held=%d itl_ewma=%.1f tokens=%d split_rate=%.0f",
                        self.pdmux_split_batches,
                        getattr(self, "pdmux_shared_batches", 0),
                        getattr(self, "pdmux_itl_held", 0),
                        getattr(self, "pdmux_itl_ewma_ms", 0.0),''')
if "\nimport time\n" not in s:
    s = s.replace("\nimport os\n", "\nimport os\nimport time\n", 1)
open(p, "w").write(s)
