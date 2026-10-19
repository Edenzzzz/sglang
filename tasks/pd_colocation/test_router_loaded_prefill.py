import sys, types
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import router as R
cls = [c for n, c in vars(R).items() if isinstance(c, type) and hasattr(c, "loaded_prefill_ms")][0]
pm = R.PerfModel(8300, 9.0, 0.3, 0.05)
pm.curve = [(4096, 490), (49152, 11200)]
a = types.SimpleNamespace(slo_chunk="20:0.02:5e-7", chunk_tokens=2048, accept_ewma_alpha=0, slo_chunk_max_steps=20000)
me = types.SimpleNamespace(a=a, pm=pm)
def w(n, kv):
    return types.SimpleNamespace(accept=None, inflight={i: dict(t_first=1.0, decode_left=100, kv=kv) for i in range(n)})
f = lambda ww, u, c: cls.loaded_prefill_ms(me, ww, u, c)
print("no decodes  32K cold :", round(f(w(0, 0), 32768, 32768)), "ms (cold curve)")
for n in (2, 8, 16, 24, 32):
    t = f(w(n, 16000), 32768, 32768)
    d = pm.tpot_ms(n, n * 16000)
    print(f"{n:2d} rows x 16K: decode step {d:5.1f} ms -> 32K prefill {t if t == float('inf') else round(t)} ms")
a.slo_chunk = ""
print("off:", round(f(w(8, 16000), 32768, 32768)))
