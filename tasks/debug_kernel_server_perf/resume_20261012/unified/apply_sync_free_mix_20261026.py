"""SGLANG_MIX_SYNC_FREE=1: the 4-mode merge (ScheduleBatch.mix_with_verify) without its device->host copy.

mix_with_verify read the running rows' committed prefix with `running_batch.seq_lens.cpu()`. Under the overlap
scheduler that waits for the GPU queue to drain, so the next step cannot be prepared while this one runs. Measured
(ShareGPT 32 req/s four-mode, H200, SGLANG_STEP_TRACE): a merged step's wall time exceeds its GPU time by 5.5 ms
(27.5 vs 22.0 ms) while stock MTP's steps overlap; that is the 6% throughput gap.

With the switch:
  - the host copies (prefix_lens, seq_lens_cpu) of the verify rows are an UPPER BOUND: the stale host mirror + n. The
    device is ahead of the mirror by at most one step's accepted tokens (<= n). Host consumers in the stock FA3 path
    use them only for maxima and `any(prefix)`;
  - the exact committed prefix stays on the device (`mixed_verify_prefix_dev`), and ForwardBatch.init_new overwrites
    the verify rows of `extend_prefix_lens` with it before computing positions, so positions (RoPE) stay exact;
  - the merged seq_lens tensor is built on the device from the exact prefix, as before.
Idempotent. Usage: python apply_sync_free_mix_20261026.py <run root>
Claude, 2026-10-26."""
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "python/sglang/srt"
sb = root / "managers/schedule_batch.py"
s = sb.read_text()
if "SGLANG_MIX_SYNC_FREE" not in s:
    old = """        verify_base_seq_lens_cpu = verify_base_seq_lens.cpu()"""
    new = """        if __import__("os").environ.get("SGLANG_MIX_SYNC_FREE") == "1" and running_batch.seq_lens_cpu is not None:
            # No device->host copy: an upper bound on the host (the mirror lags the device by at most one step's
            # accepted tokens, <= n); the exact prefix stays on the device for positions.
            verify_base_seq_lens_cpu = running_batch.seq_lens_cpu[:running_bs].clone() + n
            self.mixed_verify_prefix_dev = verify_base_seq_lens
        else:
            verify_base_seq_lens_cpu = verify_base_seq_lens.cpu()
            self.mixed_verify_prefix_dev = None"""
    assert s.count(old) == 1, "mix_with_verify layout changed"
    s = s.replace(old, new)
    compile(s, str(sb), "exec")
    sb.write_text(s)
fb = root / "model_executor/forward_batch_info.py"
s = fb.read_text()
if "mixed_verify_prefix_dev" not in s:
    old = """                ret.extend_prefix_lens_cpu = list(extend_prefix_lens)
                ret.extend_seq_lens_cpu = list(extend_seq_lens)"""
    new = """                ret.extend_prefix_lens_cpu = list(extend_prefix_lens)
                ret.extend_seq_lens_cpu = list(extend_seq_lens)
                # 4-mode merge without a sync (SGLANG_MIX_SYNC_FREE): the host prefix of the verify rows is an upper
                # bound; positions come from the exact device prefix.
                _mvp = getattr(batch, "mixed_verify_prefix_dev", None)
                if _mvp is not None and _mvp.numel():
                    ret.extend_prefix_lens[-_mvp.numel():] = _mvp.to(ret.extend_prefix_lens.dtype)"""
    assert s.count(old) == 1, "init_new layout changed"
    s = s.replace(old, new)
    compile(s, str(fb), "exec")
    fb.write_text(s)
print("SYNC_FREE_MIX", "SGLANG_MIX_SYNC_FREE" in sb.read_text(), "mixed_verify_prefix_dev" in fb.read_text())
