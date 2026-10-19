"""CPU check of router.py's LoadStat subscriber and report-driven prediction: a fake worker PUB socket sends the
patched LoadStat wire array (apply_load_stat_4mode.py); the router's view of that worker must follow it, fall back to
its own bookkeeping once the report goes stale, and price a decode-heavy node's TTFT as infinite under --slo-chunk."""
import os
import sys
import threading
import time
import types

import msgspec
import zmq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import router as R

ep = f"tcp://[::1]:{__import__("random").randint(40000, 59000)}"
pub = zmq.Context.instance().socket(zmq.PUB)
pub.setsockopt(zmq.IPV6, 1)
pub.bind(ep.replace("[::1]", "[::]"))
a = types.SimpleNamespace(slo_chunk="20:0.02:5e-7", chunk_tokens=2048, accept_ewma_alpha=0, slo_chunk_max_steps=20000,
                          load_max_age_s=0.5)
pm = R.PerfModel(8300, 9.0, 0.3, 0.05)
pm.curve = [(4096, 490), (49152, 11200)]
w = R.Worker("http://x", 1_000_000, 64)
me = R.Router.__new__(R.Router)
me.a, me.pm, me.workers = a, pm, [w]
threading.Thread(target=R.subscribe_loads, args=(me, [ep]), daemon=True).start()
time.sleep(0.5)


def send(rows, kv, acc, queue):
    stat = ["LoadStat", rows, len(queue), kv, 1_000_000, 0, 0, 0, 0, 0, 0, rows, kv, acc, queue]
    pub.send_multipart([b"load", (1).to_bytes(8, "big"), msgspec.msgpack.encode(stat)])


for _ in range(20):
    send(2, 32000, 1.0, [[500.0, 16000, 0], [9000.0, 8000, 4000]])
    time.sleep(0.05)
    if w.load:
        break
assert w.load and w.load["rows"] == 2 and len(w.load["queue"]) == 2, w.load
dl = time.time() + 1.0
light = R.Router.drain_ms(me, w, dl)
print("light node: rows=2, drain ahead of a 1 s deadline =", round(light), "ms (only the 500 ms-deadline entry)")
send(40, 40 * 30000, 1.0, [[500.0, 16000, 0]])
time.sleep(0.2)
heavy = R.Router.loaded_prefill_ms(me, w, 8000, 8000)
print("heavy node: rows=40 -> loaded prefill", heavy)
assert heavy == float("inf")
time.sleep(0.7)
assert R.Router.fresh_load(me, w) is None
print("stale after", a.load_max_age_s, "s -> router bookkeeping:", R.Router.decode_state(me, w))
print("LOAD_SUB_TEST_PASS")
