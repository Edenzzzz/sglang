"""Checkpoint a chunked prefill into the radix tree every N computed tokens instead of after every chunk.

Profiling the H200 4-mode MTP server (prof/prof_mixed_20261018.py, GPU-only trace) found the GPU idle 25% of a 60-step
mixed window vs 8% for stock MTP; the largest host cost the GPU waited on was the per-chunk radix checkpoint of the
chunked request (checkpoint_into_tree -> _add_new_node / collect_full_device_indices), which walks the request's whole
computed prefix. At chunk 2048 it runs 4x as often per prompt token as stock's 8192, on prefixes up to 64K.
SGLANG_CHECKPOINT_EVERY_TOKENS=N: between checkpoints the request keeps holding its KV (req_to_token) and the tree lags
by < N tokens; the next checkpoint inserts from cache_protected_len, so it covers the skipped chunks. Parking (EDF chunk
preemption) and the final insert do their own checkpoint/release, so they are unaffected.
  python apply_checkpoint_every.py <tree>/python/sglang/srt
"""
import sys

p = f"{sys.argv[1]}/managers/scheduler.py"
s = open(p).read()
old = '''    def stash_chunked_request(self, req: Req):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.checkpoint_disagg_prefill(req)
        else:
            checkpoint_kv_cache(req, self.tree_cache)
'''
new = '''    def stash_chunked_request(self, req: Req):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.checkpoint_disagg_prefill(req)
        else:
            every = int(os.environ.get("SGLANG_CHECKPOINT_EVERY_TOKENS", "0"))
            if every > 0 and req.extend_end - getattr(req, "_ckpt_at", req.kv.cache_protected_len) < every:
                # What checkpoint_kv_cache does after publishing: the next extend resumes here.
                req.prefix_len = req.extend_end
                self.checkpoint_skipped = getattr(self, "checkpoint_skipped", 0) + 1
                if self.checkpoint_skipped % 200 == 1:
                    logger.info("CHECKPOINT_EVERY skipped=%d every=%d", self.checkpoint_skipped, every)
                return
            checkpoint_kv_cache(req, self.tree_cache)
            req._ckpt_at = req.extend_end
'''
if new in s:
    print("already")
else:
    assert s.count(old) == 1, s.count(old)
    open(p, "w").write(s.replace(old, new, 1))
    print("patched")
