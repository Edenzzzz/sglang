"""pdmux: choose SM sharing per prefill batch from EDF deadlines (PD item 5, the per-step operating point).

Builds on apply_pdmux_backlog_share.py. SGLANG_PDMUX_SHARE_DEADLINE=1: when a split-prefill batch is formed beside
running decode, walk the waiting requests in deadline order (EDF priority = absolute TTFT deadline, monotonic clock),
accumulating uncached prompt tokens (this batch first); a request whose deadline falls before its predicted prefill
completion under the SM split (cumulative tokens / (SGLANG_PDMUX_PREFILL_TOK_S * prefill_sm / total_sm)) means the
split would miss a TTFT deadline, so the batch runs with all SMs shared (group 0). Otherwise it keeps pdmux's split.
  python apply_pdmux_deadline_share.py <tree>/python/sglang/srt
"""
import sys

srt = sys.argv[1]
p = f"{srt}/multiplex/multiplexing_mixin.py"
s = open(p).read()
old = '''            share_backlog = int(os.environ.get("SGLANG_PDMUX_SHARE_BACKLOG_TOKENS", "0"))
            if share_backlog > 0:'''
new = '''            share_backlog = int(os.environ.get("SGLANG_PDMUX_SHARE_BACKLOG_TOKENS", "0"))
            if os.environ.get("SGLANG_PDMUX_SHARE_DEADLINE", "0") == "1":
                import time as _time

                rate = float(os.environ.get("SGLANG_PDMUX_PREFILL_TOK_S", "5000"))
                prefill_sm, decode_sm = self.sm_counts[stream_idx]
                split_rate = rate * prefill_sm / max(prefill_sm + decode_sm, 1)
                now = _time.monotonic()
                tokens = int(self.split_prefill_batch.extend_num_tokens or 0)
                late = any(
                    r.priority is not None and r.priority < now + tokens / split_rate
                    for r in self.split_prefill_batch.reqs
                )
                if not late:
                    for r in sorted(
                        (r for r in self.waiting_queue if r.priority is not None),
                        key=lambda r: r.priority,
                    ):
                        tokens += max(0, len(r.origin_input_ids) - getattr(r, "_edf_cached_len", 0))
                        if r.priority < now + tokens / split_rate:
                            late = True
                            break
                self.pdmux_split_batches = getattr(self, "pdmux_split_batches", 0) + 1
                if late:
                    stream_idx = 0
                    self.pdmux_shared_batches = getattr(self, "pdmux_shared_batches", 0) + 1
                if self.pdmux_split_batches % 200 == 1:
                    logger.info(
                        "PDMUX_SHARE_DEADLINE batches=%d shared=%d tokens=%d split_rate=%.0f",
                        self.pdmux_split_batches,
                        getattr(self, "pdmux_shared_batches", 0),
                        tokens,
                        split_rate,
                    )
            elif share_backlog > 0:'''
if new in s:
    print("already")
else:
    assert s.count(old) == 1, s.count(old)
    open(p, "w").write(s.replace(old, new, 1))
    print("patched")
