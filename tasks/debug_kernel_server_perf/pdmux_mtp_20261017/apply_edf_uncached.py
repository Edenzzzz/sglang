"""EDF TTFT deadline on UNCACHED prompt tokens (SGLANG_EDF_UNCACHED_DEADLINE=1), applied to tree up_pdmux_mtp.

The length-scaled TTFT deadline used the full prompt length, so a follow-up agent turn whose 32-64K context is already in
the radix tree (a few hundred new tokens) got the same late deadline as a cold prompt of that length, though its prefill
costs a fraction. The cached length is read at enqueue -- the previous turn's KV is in the tree by then -- with a
read-only match (no prefix adopted onto the request), and the deadline scales with the uncached remainder.
  python apply_edf_uncached.py <tree>/python/sglang/srt
"""
import sys

srt = sys.argv[1]


def patch(rel, old, new):
    p = f"{srt}/{rel}"
    s = open(p).read()
    if new in s:
        print("already", rel); return
    assert s.count(old) == 1, (rel, s.count(old), old[:70])
    open(p, "w").write(s.replace(old, new)); print("patched", rel)


patch("managers/edf_scheduler.py",
'''TTFT_SLO_MS = float(os.environ.get("SGLANG_TTFT_SLO_MS", "2000"))''',
'''TTFT_SLO_MS = float(os.environ.get("SGLANG_TTFT_SLO_MS", "2000"))
# Scale the TTFT deadline with the prompt tokens NOT already in the prefix cache (req._edf_cached_len, set at enqueue).
EDF_UNCACHED_DEADLINE = os.environ.get("SGLANG_EDF_UNCACHED_DEADLINE", "0") == "1"''')
patch("managers/edf_scheduler.py",
'''    input_len = len(req.origin_input_ids)
    slo_ms = compute_ttft_slo_ms(input_len)
    # Use the monotonic arrival time we set in edf_set_priority''',
'''    input_len = len(req.origin_input_ids)
    if EDF_UNCACHED_DEADLINE:
        input_len = max(1, input_len - getattr(req, "_edf_cached_len", 0))
    slo_ms = compute_ttft_slo_ms(input_len)
    # Use the monotonic arrival time we set in edf_set_priority''')
patch("managers/scheduler.py",
'''                if is_retracted:
                    edf_set_retracted_priority(req)
                else:
                    edf_set_priority(req)
                self.edf_queue_calls += 1''',
'''                if is_retracted:
                    edf_set_retracted_priority(req)
                else:
                    from sglang.srt.managers import edf_scheduler as _edf

                    if _edf.EDF_UNCACHED_DEADLINE:
                        req._edf_cached_len = self._edf_cached_prefix_len(req)
                    edf_set_priority(req)
                self.edf_queue_calls += 1''')
patch("managers/scheduler.py",
'''    def _init_edf_runtime(self):''',
'''    def _edf_cached_prefix_len(self, req) -> int:
        """Prompt tokens already in the prefix cache (device + host), by a read-only match."""
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        result = self.tree_cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(token_ids=req.origin_input_ids, extra_key=req.extra_key,
                             cache_salt=req.cache_salt)
            )
        )
        cached = int(result.device_prefix_len) + int(result.host_hit_length or 0)
        self.edf_uncached_samples = getattr(self, "edf_uncached_samples", 0) + 1
        if self.edf_uncached_samples % 200 == 1:
            logger.info("EDF_UNCACHED calls=%d prompt=%d cached=%d", self.edf_uncached_samples,
                        len(req.origin_input_ids), cached)
        return cached

    def _init_edf_runtime(self):''')
print("done")
