"""CPU check of apply_kv_copy_source.py on a patched copy of mooncake_store.py with a fake store."""
import importlib.util, os, sys, types, random
import torch
os.environ.update(SGLANG_KV_COPY_NWORKERS="3", SGLANG_KV_COPY_WORKER="1")
spec = importlib.util.spec_from_file_location("ms", sys.argv[1]); ms = importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)

class FakeStore:
    def __init__(self): self.d = {}; self.reads = []
    def get_hostname(self): return "10.0.0.2:1234"
    def batch_is_exist(self, ks): return [1 if k in self.d else 0 for k in ks]
    def batch_get_into(self, ks, ptrs, sizes):
        self.reads.append(list(ks)); return [sizes[i] if k in self.d else -1 for i, k in enumerate(ks)]
    def batch_put_from(self, ks, ptrs, sizes, cfg=None):
        for k in ks: self.d[k] = cfg.preferred_segments if cfg else "default"
        return [0] * len(ks)

class Cfg:
    preferred_segments = None

def make(policy):
    os.environ["SGLANG_KV_COPY_POLICY"] = policy
    st = ms.MooncakeStore.__new__(ms.MooncakeStore)
    st.store = FakeStore(); st.config_prefix = None; st.is_mla_backend = False; st.should_split_heads = False
    st.mha_suffix = "r0"; st.enable_storage_metrics = False; st._replicate_config_cls = Cfg
    st.mem_pool_host = types.SimpleNamespace(page_size=4, kv_buffer=object())
    st._batch_preprocess = lambda keys, hi: ([f"{k}_r0_{x}" for k in keys for x in "kv"], [0] * 2 * len(keys), [8] * 2 * len(keys))
    st.pp_rank = 0
    return st

keys = [f"h{i}" for i in range(5)]; hi = torch.arange(20)
for pol in ("first", "random", "least_latency"):
    random.seed(0)
    st = make(pol)
    for k in keys:  # base copy written by worker 0 (source 0)
        for x in "kv": st.store.d[f"{k}_r0_{x}"] = "default"
    for k in keys[:3]:  # worker 2 holds a copy of the first 3 pages only
        for x in "kv": st.store.d[f"{k}@w2_r0_{x}"] = ["w2"]
    assert st.batch_exists(keys) == 5
    ok = st.batch_get_v1(keys, hi)
    assert all(ok), ok
    assert st.store.reads[-1][0] == "h0_r0_k", st.store.reads[-1]  # only the base covers all 5 pages
    assert all(st.store.d.get(f"{k}@w1_r0_{x}") == ["10.0.0.2:1234"] for k in keys for x in "kv")  # own copy written
    ok = st.batch_get_v1(keys, hi)
    assert st.store.reads[-1][0] == "h0@w1_r0_k", st.store.reads[-1]  # own copy preferred
    # 3-page request: base, w2 and own all cover it -> own wins
    st.batch_get_v1(keys[:3], hi[:12]); assert st.store.reads[-1][0] == "h0@w1_r0_k"
    # drop own copy; 3-page request now chooses between base (0) and w2 (3)
    st.store.d = {k: v for k, v in st.store.d.items() if "@w1" not in k}
    picks = set()
    for _ in range(20):
        st.store.d = {k: v for k, v in st.store.d.items() if "@w1" not in k}
        st.batch_get_v1(keys[:3], hi[:12]); picks.add(st.store.reads[-1][0])
    print(pol, "picks", sorted(picks), "lat", st._kv_copy["lat"], "reads", st._kv_copy["reads"])
    if pol == "first": assert picks == {"h0_r0_k"}
    if pol == "random": assert picks == {"h0_r0_k", "h0@w2_r0_k"}
    if pol == "least_latency": assert "h0@w2_r0_k" in picks  # untried source explored
# off: unpatched path, no copy written
st = make("off")
for k in keys:
    for x in "kv": st.store.d[f"{k}_r0_{x}"] = "default"
assert st.batch_exists(keys) == 5 and all(st.batch_get_v1(keys, hi)) and not any("@w" in k for k in st.store.d)
print("KV_COPY_TEST_PASS")
