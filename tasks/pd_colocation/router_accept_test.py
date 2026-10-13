"""Acceptance-aware TPOT in the colocation router (CPU only; point 3)."""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router import PerfModel, Worker

w = Worker("http://x", 1_000_000, 256)
assert w.accept is None
w.observe_accept(300, 100, 0.5)            # first sample is taken as is
assert abs(w.accept - 3.0) < 1e-9
w.observe_accept(200, 100, 0.5)            # EWMA: 0.5*3 + 0.5*2
assert abs(w.accept - 2.5) < 1e-9
w.observe_accept(0, 10, 0.5); w.observe_accept(10, 0, 0.5)   # incomplete metadata is ignored
assert abs(w.accept - 2.5) < 1e-9

pm = PerfModel(20000, 15.0, 0.3)
base = pm.tpot_ms(10, 0.0)
assert abs(base - 18.0) < 1e-9                       # unchanged default: per-step
assert abs(pm.tpot_ms(10, 0.0, None) - base) < 1e-9
assert abs(pm.tpot_ms(10, 0.0, 3.0) - base / 3.0) < 1e-9   # per-token under MTP

pm.step = {"c0": 7.0, "c_tok": 0.02, "c_attn": 1.4, "c_bs": 0.32, "c_kv": 0.077, "c_acc": 0.5}
step = 7.0 + 0.32 * 10 + 0.077 * 400_000 / 1e3 + 0.5 * 2.0
assert abs(pm.tpot_ms(10, 400_000, 2.0) - step / 2.0) < 1e-9
step1 = 7.0 + 0.32 * 10 + 0.077 * 400_000 / 1e3 + 0.5 * 1.0
assert abs(pm.tpot_ms(10, 400_000, None) - step1) < 1e-9   # no EWMA yet: acceptance 1
print("router_accept_test: PASS")
