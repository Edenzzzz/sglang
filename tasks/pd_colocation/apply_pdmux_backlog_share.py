"""pdmux: run a prefill batch with all SMs shared (stream group 0) when the queued prefill backlog is large.

On H20 prefill compute is the bottleneck and pdmux's SM split keeps prefill on at most 56 of 78 SMs whenever decode
runs (decode groups need >= 16 SMs; partitions come in multiples of 8), so TTFT lost 1.9-3.2x to stock colocation.
SGLANG_PDMUX_SHARE_BACKLOG_TOKENS=N: when a new split-prefill batch is formed beside running decode and the uncached
prompt tokens waiting (queue + this batch) reach N, the batch runs in group 0 -- prefill and decode on normal streams
over the whole GPU, as stock colocation does -- otherwise in the decode-isolating group pdmux picks. 0 = off.
  python apply_pdmux_backlog_share.py <tree>/python/sglang/srt
"""
import sys

srt = sys.argv[1]
p = f"{srt}/multiplex/multiplexing_mixin.py"
s = open(p).read()
old = '''                        // self.pdmux_config.decode_bs_divisor,
                    ),
                )
            set_current_stream_idx(stream_idx)'''
new = '''                        // self.pdmux_config.decode_bs_divisor,
                    ),
                )
            share_backlog = int(os.environ.get("SGLANG_PDMUX_SHARE_BACKLOG_TOKENS", "0"))
            if share_backlog > 0:
                backlog = int(self.split_prefill_batch.extend_num_tokens or 0) + sum(
                    max(0, len(r.origin_input_ids) - getattr(r, "_edf_cached_len", 0))
                    for r in self.waiting_queue
                )
                self.pdmux_split_batches = getattr(self, "pdmux_split_batches", 0) + 1
                if backlog >= share_backlog:
                    stream_idx = 0
                    self.pdmux_shared_batches = getattr(self, "pdmux_shared_batches", 0) + 1
                if self.pdmux_split_batches % 200 == 1:
                    logger.info(
                        "PDMUX_SHARE batches=%d shared=%d backlog=%d threshold=%d",
                        self.pdmux_split_batches,
                        getattr(self, "pdmux_shared_batches", 0),
                        backlog,
                        share_backlog,
                    )
            set_current_stream_idx(stream_idx)'''
if new in s:
    print("already")
else:
    assert s.count(old) == 1, s.count(old)
    s = s.replace(old, new, 1)
    if "\nimport os\n" not in s:
        s = s.replace("\nimport logging\n", "\nimport logging\nimport os\n", 1)
    open(p, "w").write(s)
    print("patched")
