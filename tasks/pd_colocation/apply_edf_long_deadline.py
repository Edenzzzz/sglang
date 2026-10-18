"""EDF: multiply the TTFT deadline of long uncached prefills (user, 2026-10-18).

SGLANG_EDF_LONG_DEADLINE="tokens:scale": a request whose uncached prompt (with SGLANG_EDF_UNCACHED_DEADLINE; else its
whole prompt) is at least `tokens` gets `scale` x the length-scaled TTFT SLO. With 200000:2 a > 200K-token prefill is
ordered against twice its usual slack, so it stops pre-empting chat and shorter agent rounds of nearer deadline; the
same tier is the evaluation SLO for those rounds (tiered TTFT SLOs by length, cf. TokenScale; JITServe's fixed
per-type SLOs ignore length).
  python apply_edf_long_deadline.py <tree>/python/sglang/srt
"""
import sys

p = f"{sys.argv[1]}/managers/edf_scheduler.py"
s = open(p).read()
old = '''    extra_tokens = max(0, input_len - 4096)
    extra_ms = (extra_tokens / TTFT_SCALE_TOKENS) * TTFT_SCALE_MS
    return TTFT_SLO_MS + extra_ms
'''
new = '''    extra_tokens = max(0, input_len - 4096)
    extra_ms = (extra_tokens / TTFT_SCALE_TOKENS) * TTFT_SCALE_MS
    slo_ms = TTFT_SLO_MS + extra_ms
    long_deadline = os.environ.get("SGLANG_EDF_LONG_DEADLINE", "")
    if long_deadline:
        tokens, scale = long_deadline.split(":")
        if input_len >= int(tokens):
            slo_ms *= float(scale)
    return slo_ms
'''
if new in s:
    print("already")
else:
    assert s.count(old) == 1, s.count(old)
    open(p, "w").write(s.replace(old, new, 1))
    print("patched")
