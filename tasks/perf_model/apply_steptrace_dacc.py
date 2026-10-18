"""STEPTRACE: add dacc= -- accepted tokens per verify since the previous traced step (the metrics reporter's deltas),
next to acc= (the run-wide cumulative mean, nearly constant after warm-up). dacc is the per-step acceptance signal an
online EWMA predictor would track (goal point 3: predicted acceptance length as a TPOT-model input).
  python apply_steptrace_dacc.py <tree>/python/sglang/srt
"""
import sys

p = f"{sys.argv[1]}/managers/scheduler.py"
s = open(p).read()
old = '''        feat = (
            "STEPTRACE %s et=%d bs=%d np=%d ptok=%d pf=%d nd=%d dkv=%d kv=%d mx=%d acc=%.4f t=%.6f"
            % (getattr(batch.forward_mode, "name", str(batch.forward_mode)),
               int(batch.extend_num_tokens or 0), batch.batch_size(), n_p, p_tok, p_prefix,
               n_d, d_kv, int(sum(seq_lens)), max_ext, acc, time.perf_counter())
        )'''
new = '''        tot_acc, tot_fwd = reporter.spec_total_num_accept_tokens, reporter.spec_total_num_forward_ct
        prev_acc, prev_fwd = getattr(self, "_steptrace_prev_acc", (tot_acc, tot_fwd))
        self._steptrace_prev_acc = (tot_acc, tot_fwd)
        dacc = (tot_acc - prev_acc) / (tot_fwd - prev_fwd) if tot_fwd > prev_fwd else 0.0
        feat = (
            "STEPTRACE %s et=%d bs=%d np=%d ptok=%d pf=%d nd=%d dkv=%d kv=%d mx=%d acc=%.4f dacc=%.4f t=%.6f"
            % (getattr(batch.forward_mode, "name", str(batch.forward_mode)),
               int(batch.extend_num_tokens or 0), batch.batch_size(), n_p, p_tok, p_prefix,
               n_d, d_kv, int(sum(seq_lens)), max_ext, acc, dacc, time.perf_counter())
        )'''
if new in s:
    print("already")
else:
    assert s.count(old) == 1, s.count(old)
    open(p, "w").write(s.replace(old, new, 1))
    print("patched")
