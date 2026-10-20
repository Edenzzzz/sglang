"""Deadline-ordered decode rows: each step decodes the K running requests whose next token is due soonest.

With MTP, sglang caps running requests at the speculative CUDA-graph batch size (48): above it every verify step slows
for every row (cap 64 and 128 measured worse for all systems), so new chats queue for a slot -- 36-39% of our chat
misses on the standard mix are TTFT misses behind that cap while the running chats decode at 13 ms/token against a
20 ms SLO. Here the running set may grow to --max-running-requests (resident KV), but a step carries at most K rows
(the graph batch size), chosen by deadline: a request that has emitted o tokens since it started decoding at t0 is due
its next token at t0 + slo * (o + 1); rows with the least time left run, the rest sit out this step and keep their KV
and draft state (the overlap relay is keyed by req_pool_idx, so a skipped row's inputs stay in place). Every row still
meets its mean-TPOT SLO while slack exists; the slack lets new requests start instead of waiting for a slot.

Rows are merged back before admission (so the running-request cap, KV reservation and EDF chunk rules see all of
them) and split right before the decode / 4-mode verify batch is formed. Per-class SLOs as in apply_edf_itl_slo_chunk:
prompts >= long_tok use long_ms (agents), the rest slo_ms (chat).

  SGLANG_DECODE_ROWS="K:slo_ms:long_tok:long_ms"     e.g. 48:20:24000:80
  python apply_decode_rows_edf.py <tree>/python/sglang/srt
Logs `DECODE_ROWS steps= split= rows= parked=` every 500 split steps.
Claude, 2026-10-20.
"""
import sys

p = f"{sys.argv[1]}/managers/scheduler.py"
s = open(p).read()
assert "_decode_rows_select" not in s, "already applied"

if "\nimport copy\n" not in s:
    s = s.replace("\nimport os\n", "\nimport copy\nimport os\n", 1)

# 1. Merge the rows that sat out the last step back before admission.
old = """        # For prefill-only batch, filter out finished requests since they
        # won't go through the decode step. This keeps running_batch accurate
"""
assert s.count(old) == 1
s = s.replace(old, """        parked = getattr(self, "_decode_rows_parked", None)
        if parked is not None:
            self._decode_rows_parked = None
            if not parked.is_empty():
                if running_batch.is_empty():
                    running_batch = parked
                else:
                    running_batch.merge_batch(parked)

""" + old)

# 2. Decode step.
old = """            if not running_batch.is_empty() and not running_batch.is_prefill_only:
                running_batch = self.update_running_batch(running_batch)
"""
assert s.count(old) == 1
s = s.replace(old, """            if not running_batch.is_empty() and not running_batch.is_prefill_only:
                running_batch = self._decode_rows_select(running_batch)
                running_batch = self.update_running_batch(running_batch)
""")

# 3. 4-mode step: the verify rows attached to the prefill chunk.
old = """            running_batch.filter_batch()
            if os.environ.get("SGLANG_DBG_DRAFT_COPY") == "1":
                _si = running_batch.spec_info
                _tp = getattr(_si, "topk_p", None)
                logger.error(
                    "MIXEDGATE post-filter"""
assert s.count(old) == 1
s = s.replace(old, """            running_batch.filter_batch()
            running_batch = self._decode_rows_select(running_batch)
            if os.environ.get("SGLANG_DBG_DRAFT_COPY") == "1":
                _si = running_batch.spec_info
                _tp = getattr(_si, "topk_p", None)
                logger.error(
                    "MIXEDGATE post-filter""")

old = "    def _edf_deadline_chunk(self, running_batch, chunk_size):"
assert s.count(old) == 1
s = s.replace(old, '''    def _decode_rows_select(self, running_batch):
        """Keep the K running rows whose next token is due soonest; the others sit out this step
        (self._decode_rows_parked, merged back at the next get_next_batch_to_run)."""
        spec = os.environ.get("SGLANG_DECODE_ROWS", "")
        if not spec or running_batch.is_empty():
            return running_batch
        k, slo_ms, long_tok, long_ms = (float(x) for x in spec.split(":"))
        k = int(k)
        running_batch.filter_batch()
        n = len(running_batch.reqs)
        si = running_batch.sampling_info
        if (
            n <= k
            or si.penalizer_orchestrator.is_required
            or si.has_custom_logit_processor
            or running_batch.has_grammar
            or running_batch.return_logprob
        ):
            return running_batch
        now = time.monotonic()
        due = []
        for r in running_batch.reqs:
            if getattr(r, "_dr_t0", None) is None:
                r._dr_t0, r._dr_o0 = now, len(r.output_ids)
            s_r = long_ms if len(r.origin_input_ids) >= long_tok else slo_ms
            due.append(r._dr_t0 * 1000.0 + s_r * (len(r.output_ids) - r._dr_o0 + 1))
        order = sorted(range(n), key=due.__getitem__)
        run_idx, park_idx = sorted(order[:k]), sorted(order[k:])
        parked = copy.copy(running_batch)
        parked.sampling_info = copy.copy(si)
        if running_batch.spec_info is not None:
            parked.spec_info = copy.copy(running_batch.spec_info)
        parked.filter_batch(keep_indices=park_idx)
        running_batch.filter_batch(keep_indices=run_idx)
        self._decode_rows_parked = parked
        st = getattr(self, "_decode_rows_stats", None) or [0, 0, 0]
        st[0] += 1
        st[1] += n
        st[2] += len(park_idx)
        self._decode_rows_stats = st
        if st[0] % 500 == 1:
            logger.info(
                "DECODE_ROWS split=%d rows=%.1f parked=%.1f",
                st[0], st[1] / st[0], st[2] / st[0],
            )
        return running_batch

''' + old)
open(p, "w").write(s)
print("applied", p)
