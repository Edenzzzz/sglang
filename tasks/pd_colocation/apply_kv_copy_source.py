"""Point 4 as proposed: choose the SOURCE of a KV pull among copies of the prefix, with the serving node fixed.

sglang's mooncake store writes each page once (ReplicateConfig.replica_num = 1) and mooncake's client reads the first
complete replica in the master's order, so a pull has exactly one source. This patch adds copies and a source choice:

  copies   after a worker pulls pages from another source it re-exports them under its own key suffix (`@w<worker>`)
           on its own segment (ReplicateConfig.preferred_segments = [this client's hostname]). A hot prefix then has
           one copy per worker that has served it.
  sources  source 0 is the original key (written by whoever computed the prefix); source j + 1 is worker j's copy.
           A worker's own copy is always read first (local segment). Otherwise, among the sources holding the whole
           requested range: `first` (lowest index -- the stock single-source behaviour, plus copies), `random`
           (point 4's rule), `least_latency` (EWMA of observed seconds per page per source, untried sources first;
           learned from this worker's own pulls, no router query).

  SGLANG_KV_COPY_POLICY=off|first|random|least_latency   (off = unpatched behaviour)
  SGLANG_KV_COPY_WORKER=<global worker index>  SGLANG_KV_COPY_NWORKERS=<worker count>
  python apply_kv_copy_source.py <tree>/python/sglang/srt
Logs `KV_COPY_STATS` every 200 pulls: reads and pages per source, EWMA latency per source, copies written.
Claude, 2026-10-19.
"""
import sys

p = f"{sys.argv[1]}/mem_cache/storage/mooncake_store/mooncake_store.py"
s = open(p).read()
if "_batch_get_v1_orig" in s:
    print("already")
    sys.exit(0)

for old, new in (("    def batch_get_v1(\n", "    def _batch_get_v1_orig(\n"),
                 ("    def batch_exists(\n", "    def _batch_exists_orig(\n")):
    assert s.count(old) == 1, old
    s = s.replace(old, new)

methods = '''
    def _kv_copy_state(self):
        st = getattr(self, "_kv_copy", None)
        if st is None:
            n = int(os.getenv("SGLANG_KV_COPY_NWORKERS", "0"))
            st = self._kv_copy = dict(
                policy=os.getenv("SGLANG_KV_COPY_POLICY", "off"),
                me=int(os.getenv("SGLANG_KV_COPY_WORKER", "-1")),
                n=n,
                lat=[None] * (n + 1),
                reads=[0] * (n + 1),
                pages=[0] * (n + 1),
                pulls=0,
                copies=0,
                cfg=None,
            )
            if st["policy"] != "off":
                assert n > 0 and 0 <= st["me"] < n, (n, st["me"])
                cfg = self._replicate_config_cls()
                cfg.preferred_segments = [self.store.get_hostname()]
                st["cfg"] = cfg
                logger.info(
                    "KV_COPY policy=%s worker=%d/%d segment=%s",
                    st["policy"], st["me"], n, cfg.preferred_segments[0],
                )
        return st

    @staticmethod
    def _kv_copy_keys(keys, src):
        return keys if src == 0 else [f"{k}@w{src - 1}" for k in keys]

    def _kv_copy_coverage(self, keys, extra_info):
        st = self._kv_copy_state()
        return [
            self._batch_exists_orig(self._kv_copy_keys(keys, src), extra_info)
            for src in range(st["n"] + 1)
        ]

    def batch_exists(
        self, keys, extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        if self._kv_copy_state()["policy"] == "off":
            return self._batch_exists_orig(keys, extra_info)
        return max(self._kv_copy_coverage(keys, extra_info))

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        st = self._kv_copy_state()
        if st["policy"] == "off" or self.mem_pool_host.kv_buffer is None:
            return self._batch_get_v1_orig(keys, host_indices, extra_info)
        cover = self._kv_copy_coverage(keys, extra_info)
        cands = [src for src, c in enumerate(cover) if c >= len(keys)]
        if not cands:
            # Pages were evicted between exists and get: read what the best source still holds.
            cands = [max(range(len(cover)), key=lambda src: cover[src])]
        own = st["me"] + 1
        if own in cands:
            src = own
        elif st["policy"] == "first":
            src = cands[0]
        elif st["policy"] == "random":
            src = random.choice(cands)
        else:
            src = min(cands, key=lambda c: (st["lat"][c] is not None, st["lat"][c] or 0.0, random.random()))
        t0 = time.perf_counter()
        ok = self._batch_get_v1_orig(self._kv_copy_keys(keys, src), host_indices, extra_info)
        dt = time.perf_counter() - t0
        got = 0
        while got < len(ok) and ok[got]:
            got += 1
        if got:
            per_page = dt / got
            prev = st["lat"][src]
            st["lat"][src] = per_page if prev is None else 0.8 * prev + 0.2 * per_page
        st["reads"][src] += 1
        st["pages"][src] += got
        st["pulls"] += 1
        if src != own and got:
            page = self.mem_pool_host.page_size
            res = self._kv_copy_put(keys[:got], host_indices[: got * page], own)
            st["copies"] += sum(1 for r in res if r)
        if st["pulls"] % 200 == 0:
            logger.info(
                "KV_COPY_STATS pulls=%d copies=%d reads=%s pages=%s lat_ms_per_page=%s",
                st["pulls"], st["copies"], st["reads"], st["pages"],
                [None if v is None else round(v * 1e3, 3) for v in st["lat"]],
            )
        return ok

    def _kv_copy_put(self, keys, host_indices, src):
        st = self._kv_copy_state()
        keys = self._tag_keys(self._kv_copy_keys(keys, src))
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)
        exist = self._batch_exist(key_strs)
        todo = [i for i in range(len(key_strs)) if exist[i] != 1]
        results = [0] * len(key_strs)
        if todo:
            args = ([key_strs[i] for i in todo], [buffer_ptrs[i] for i in todo], [buffer_sizes[i] for i in todo])
            if self._uses_multi_buffer(buffer_ptrs):
                put = self.store.batch_put_from_multi_buffers(*args, st["cfg"])
            else:
                put = self.store.batch_put_from(*args, st["cfg"])
            for i, r in zip(todo, put):
                results[i] = r
        return self._batch_postprocess(results, is_set_operate=True)

'''
anchor = "    def close(self):\n"
assert s.count(anchor) == 1
s = s.replace(anchor, methods + anchor)
if "\nimport random\n" not in s:
    s = s.replace("import os\n", "import os\nimport random\n", 1)
open(p, "w").write(s)
print("patched")
