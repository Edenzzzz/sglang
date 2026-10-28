"""Measure the draft head's share of a speculative step with in-step CUDA events (2026-10-28).

Gates the draft/verify pipelining item (the user's idea: overlap one group's draft with another group's verify).
Its ceiling is how much of a step the draft forwards occupy, and that cannot be read from STEPTRACE: `gpu=` already
spans the whole step, so draft and verify are indistinguishable there (measured on 18,075 decode-step pairs: wall
gap p50 30.08 ms against a traced forward of p50 30.98 ms, i.e. only 0.2% outside the forward).

Appends a wrapper at module scope -- no edit to any method body -- that times `EagleDraftWorker.draft_forward` and
`verify` on EVERY speculative worker class this tree defines (v1 and v2 -- the factory picks between
them on enable_overlap) with CUDA events, and logs the split every N verify calls. Events are recorded on the current
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
    # BOTH worker classes must be wrapped, which is why the first three attempts reported nothing.
    # `_create_eagle_worker` (speculative/spec_info.py) returns EAGLEWorkerV2 only when enable_overlap is true and
    # the v1 EAGLEWorker otherwise, and BOTH define `draft_forward` and `verify`. Wrapping v2 alone therefore
    # instruments a class the arm may never construct, and the failure is silent: the module-scope patch installs,
    # logs "active", and no report ever fires. Wrap whichever classes this tree defines, and name them.
    _ds_installed = []
    for _ds_modname, _ds_clsname in (("eagle_worker_v2", "EAGLEWorkerV2"), ("eagle_worker", "EAGLEWorker")):
        try:
            _ds_mod = __import__(f"sglang.srt.speculative.{_ds_modname}", fromlist=[_ds_clsname])
            _ds_cls = getattr(_ds_mod, _ds_clsname)
        except (ImportError, AttributeError):
            continue
        if "verify" in vars(_ds_cls):
            _ds_wrap(_ds_cls, "verify", "verify")
            _ds_installed.append(_ds_clsname)
    _ds_log.info("DRAFT_SHARE instrumentation active on %s", ", ".join(_ds_installed) or "NOTHING")
    assert _ds_installed, "no speculative worker class carried a `verify` to wrap"

    # And report which class the live worker actually is, so a zero-report run is diagnosable in one line instead
    # of three driver attempts.
    try:
        from sglang.srt.managers import tp_worker as _ds_tp
        _ds_sched_init = _ds_tp.TpModelWorker.__init__

        def _ds_init(self, *a, **k):
            _ds_sched_init(self, *a, **k)
            _ds_sw = getattr(self, "spec_worker", None)
            _ds_log.info("DRAFT_SHARE live spec worker: %s", type(_ds_sw).__name__ if _ds_sw else "none")

        _ds_tp.TpModelWorker.__init__ = _ds_init
    except Exception as _ds_exc:  # the probe must never break a run
        _ds_log.info("DRAFT_SHARE could not install the live-worker probe: %r", _ds_exc)
'''
open(p, "w").write(s)
print("patched")
