"""STEPTRACE under PD multiplexing (tree up_pdmux_mtp), for the EDF step-model check on the pdmux schedule.

STEPTRACE was emitted only in run_batch's overlap branch, and pdmux runs with overlap off, so its steps were never traced.
event_loop_pdmux now wraps its decode run_batch (decode stream) and split-prefill run_batch (prefill stream) in
_steptrace_begin/_steptrace_end -- the GPU events record on the stream the work runs on -- and the line carries the
stream group (sg=, the SM split: group i = sm_counts[i]) and the layers this split forward ran (lay=).
  python apply_steptrace_pdmux.py <tree>/python/sglang/srt
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


patch("managers/scheduler.py",
'''    def _steptrace_begin(self, batch: ScheduleBatch) -> None:''',
'''    def _steptrace_begin(self, batch: ScheduleBatch, extra: str = "") -> None:''')
patch("managers/scheduler.py",
'''               n_d, d_kv, int(sum(seq_lens)), max_ext, acc, time.perf_counter())
        )''',
'''               n_d, d_kv, int(sum(seq_lens)), max_ext, acc, time.perf_counter())
        )
        if extra:
            feat = feat.replace(" acc=", f" {extra} acc=", 1)''')
patch("managers/scheduler.py",
'''        is_extend = batch.forward_mode.is_extend()
        seq_lens = batch.seq_lens_cpu.tolist()''',
'''        is_extend = batch.forward_mode.is_extend() or batch.forward_mode.is_split_prefill()
        seq_lens = batch.seq_lens_cpu.tolist()''')
patch("multiplex/multiplexing_mixin.py",
'''                    decode_result = self.run_batch(running_batch)''',
'''                    self._steptrace_begin(running_batch, f"sg={stream_idx} lay=0")
                    decode_result = self.run_batch(running_batch)
                    self._steptrace_end()''')
patch("multiplex/multiplexing_mixin.py",
'''                    prefill_result = self.run_batch(self.split_prefill_batch)''',
'''                    self._steptrace_begin(self.split_prefill_batch, f"sg={stream_idx} lay={forward_count}")
                    prefill_result = self.run_batch(self.split_prefill_batch)
                    self._steptrace_end()''')
print("done")
