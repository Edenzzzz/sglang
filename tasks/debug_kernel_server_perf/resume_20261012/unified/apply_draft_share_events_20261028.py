"""Measure the draft head's share of a speculative step with in-step CUDA events (2026-10-28).

Gates the draft/verify pipelining item (the user's idea: overlap one group's draft with another group's verify).
Its ceiling is how much of a step the draft forwards occupy, and that cannot be read from STEPTRACE: `gpu=` already
spans the whole step, so draft and verify are indistinguishable there (measured on 18,075 decode-step pairs: wall
gap p50 30.08 ms against a traced forward of p50 30.98 ms, i.e. only 0.2% outside the forward).

Appends a wrapper at module scope -- no edit to any method body -- that times `EagleDraftWorker.draft_forward` and
`EAGLEWorkerV2.verify` with CUDA events and logs the split every N verify calls. Events are recorded on the current
stream and synchronised only at report time, so the steady-state cost is two event records per call.

  SGLANG_DRAFT_SHARE=1            turn it on (off by default, so no arm changes unless asked)
  SGLANG_DRAFT_SHARE_EVERY=200    report every N verify calls
  python apply_draft_share_events_20261028.py <tree>/python/sglang/srt
Logs: DRAFT_SHARE draft_ms=.. verify_ms=.. share=..% nd=.. nv=..
Claude.
"""
import sys

p = f"{sys.argv[1]}/speculative/eagle_worker_v2.py"
s = open(p).read()
if "DRAFT_SHARE" in s:
    print("already")
    sys.exit(0)
assert "class EAGLEWorkerV2(" in s and "class EagleDraftWorker(" in s, "unexpected module layout"

s += '''

# --- draft-share measurement (apply_draft_share_events_20261028.py), off unless SGLANG_DRAFT_SHARE=1 ----------
import os as _ds_os

if _ds_os.environ.get("SGLANG_DRAFT_SHARE") == "1":
    import logging as _ds_logging

    import torch as _ds_torch

    _ds_store = {"draft": [], "verify": []}
    _ds_log = _ds_logging.getLogger(__name__)

    def _ds_report():
        _ds_torch.cuda.synchronize()
        draft_ms = sum(a.elapsed_time(b) for a, b in _ds_store["draft"])
        verify_ms = sum(a.elapsed_time(b) for a, b in _ds_store["verify"])
        n_d, n_v = len(_ds_store["draft"]), len(_ds_store["verify"])
        _ds_log.info(
            "DRAFT_SHARE draft_ms=%.3f verify_ms=%.3f step_ms=%.3f share=%.1f%% nd=%d nv=%d",
            draft_ms / max(n_d, 1), verify_ms / max(n_v, 1),
            (draft_ms + verify_ms) / max(n_v, 1),
            100.0 * draft_ms / max(draft_ms + verify_ms, 1e-9), n_d, n_v)
        _ds_store["draft"].clear()
        _ds_store["verify"].clear()

    def _ds_wrap(cls, name, bucket):
        original = getattr(cls, name)

        def wrapper(self, *args, **kwargs):
            start = _ds_torch.cuda.Event(enable_timing=True)
            end = _ds_torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                return original(self, *args, **kwargs)
            finally:
                end.record()
                _ds_store[bucket].append((start, end))
                if bucket == "verify" and len(_ds_store["verify"]) >= int(
                        _ds_os.environ.get("SGLANG_DRAFT_SHARE_EVERY", "200")):
                    _ds_report()

        wrapper.__name__ = name
        setattr(cls, name, wrapper)

    # draft_forward runs once per speculative step (3 per verify at --speculative-num-steps 3), so the draft bucket
    # accumulates every draft forward and nd/nv shows the ratio actually used.
    _ds_wrap(EagleDraftWorker, "draft_forward", "draft")
    _ds_wrap(EAGLEWorkerV2, "verify", "verify")
    _ds_log.info("DRAFT_SHARE instrumentation active")
'''
open(p, "w").write(s)
print("patched")
