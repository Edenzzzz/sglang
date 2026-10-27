"""Demote a prefill the step model predicts will miss its TTFT deadline (user, 2026-10-27).

`apply_edf_late_demote_20261024.py` already demotes a prefill whose deadline is earlier than
`now + uncached_tokens / SGLANG_EDF_LATE_DEMOTE_TOK_S`, i.e. it predicts the finish time from one constant token
rate. Every multi-node arm ran it with `TOK_S=1e12`, which makes the test fire only for prefills that have ALREADY
missed. This replaces the constant rate with the step model, so the test becomes "can this request still make its
deadline at the current chunk size under the current decode load":

    forwards        = ceil(uncached_tokens / chunk)
    predicted_ttft  = sum over those forwards of predict(chunk, prefix_so_far, decode_avg_kv, decode_bs)
    demote if         now + predicted_ttft > deadline

The prefix grows by `chunk` each forward, so the walk is the same one the router does (`router.py
budget_prefill_ms`). The predictor is whichever one `try_load_latency_predictor` returned, so this works with the
trained MLP (held-out 3.4-4.0% on merged steps) as well as the analytic and linear models: it only ever evaluates
the model forward, and never has to invert it. Inverting is needed for choosing the chunk size, and
`edf_compute_max_chunk_size` already does that by binary search over the same predictor.

This replaces preemption. Parking an in-progress chunked prefill (`SGLANG_EDF_PREEMPT_CHUNK`) cost about a quarter
of agent SLO goodput on the cluster replay and bought only the chat TTFT tail; a request predicted to miss is
demoted instead, so nothing already running is interrupted and no computed prefix is re-queued.

  SGLANG_EDF_MODEL_DEMOTE=1           use the step model instead of the constant token rate
  SGLANG_EDF_MODEL_DEMOTE_MARGIN=1.0  multiply the predicted TTFT before comparing (headroom for model error)
  python apply_edf_model_demote_20261027.py <tree>/python/sglang/srt
Claude.
"""
import sys

p = f"{sys.argv[1]}/managers/scheduler.py"
s = open(p).read()
if "_edf_model_predicted_ttft_ms" in s:
    print("already")
    sys.exit(0)
old = """                if req.priority < now + need / tok_s:"""
new = """                predicted = self._edf_model_predicted_ttft_ms(need)
                if predicted is not None:
                    if req.priority < now + predicted / 1000.0:
                        feasible = False
                elif req.priority < now + need / tok_s:"""
assert s.count(old) == 1, s.count(old)
# The original line is an `if` whose body sets the demotion; keep that body by turning our branch into the same test.
s = s.replace(old, new, 1)
helper = '''
    def _edf_model_predicted_ttft_ms(self, uncached_tokens):
        """Predicted time to finish `uncached_tokens` of prefill at the current chunk size and decode load, by
        walking the chunks through the loaded step model. None when the model-based test is off or no predictor is
        loaded, in which case the caller falls back to the constant token rate."""
        if os.environ.get("SGLANG_EDF_MODEL_DEMOTE") != "1":
            return None
        predictor = getattr(self, "edf_latency_predictor", None)
        if predictor is None or uncached_tokens <= 0:
            return None
        chunk = self.chunked_prefill_size or 2048
        running = self.running_batch.reqs if self.running_batch is not None else []
        decode_bs = len(running)
        decode_avg_kv = (sum(len(r.origin_input_ids) + len(r.output_ids) for r in running) / decode_bs
                         if decode_bs else 0.0)
        total_ms, prefix, left = 0.0, 0.0, float(uncached_tokens)
        for _ in range(1024):
            if left <= 0:
                break
            step_tokens = min(float(chunk), left)
            total_ms += float(predictor.predict(step_tokens, prefix, decode_avg_kv, decode_bs))
            prefix += step_tokens
            left -= step_tokens
        margin = float(os.environ.get("SGLANG_EDF_MODEL_DEMOTE_MARGIN", "1.0"))
        return total_ms * margin
'''
anchor = "\n    def _edf_rescore_and_park(self"
if anchor not in s:
    anchor = "\n    def get_new_batch_prefill(self"
assert s.count(anchor) >= 1
s = s.replace(anchor, helper + anchor, 1)
open(p, "w").write(s)
print("patched")
