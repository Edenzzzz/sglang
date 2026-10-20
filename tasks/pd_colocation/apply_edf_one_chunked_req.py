"""At most one unfinished chunked request per step, under every schedule policy.

sglang tracks one in-progress chunked request (scheduler.chunked_req). Stock never starts a second one because the
continuing chunk takes the whole chunk budget, so rem_chunk_tokens is 0 when the waiting queue is scanned; only
shortest-prefill-first, which limits the continuing chunk, guards explicitly. EDF chunk preemption limits it too
(_edf_chunked_req_limit), leaving budget for earlier-deadline requests -- and a waiting request too long for that
budget then became a second chunked request, tripping `assert self.chunked_req is None` in _get_new_batch_prefill_raw
(H200, 2026-10-20, ours + MTP at chat 6 req/s). The guard now applies whenever a chunked request stays in progress
(`has_chunked_req`, which the scheduler passes after add_chunked_req), on both admission paths: such a request waits
and the earlier-deadline requests that fit whole still run. The first site is edf_one_chunked_req.patch (H20 pods,
2026-10-17), which the re-forked H200 tree never received; this script also covers add_one_req_ignore_eos and skips
the first site where that patch is already in.
  python apply_edf_one_chunked_req.py <tree>/python/sglang/srt
Claude, 2026-10-20.
"""
import sys

p = f"{sys.argv[1]}/managers/schedule_policy.py"
s = open(p).read()
assert "_has_chunked_in_progress" not in s, "already applied"

old = """            if (
                has_chunked_req
                and get_schedule().schedule_policy == "shortest-prefill-first"
            ):
                # Only one unfinished chunked request can be tracked.
                return AddReqResult.OTHER
"""
assert s.count(old) == 1 or "in-progress chunk below the step budget" in s
s = s.replace(old, """            if has_chunked_req:
                # Only one unfinished chunked request can be tracked; a policy that limits the continuing
                # chunk (shortest-prefill-first, EDF chunk preemption) leaves budget that must not start another.
                return AddReqResult.OTHER
""")

old = """        else:
            if self.rem_chunk_tokens <= 0:
                return AddReqResult.OTHER

            # Chunked prefill
            trunc_len = self.rem_chunk_tokens
"""
assert s.count(old) == 1
s = s.replace(old, """        else:
            if self.rem_chunk_tokens <= 0 or self._has_chunked_in_progress:
                return AddReqResult.OTHER

            # Chunked prefill
            trunc_len = self.rem_chunk_tokens
""")

old = "            return self.add_one_req_ignore_eos(req)\n"
assert s.count(old) == 1
s = s.replace(old, "            self._has_chunked_in_progress = has_chunked_req\n" + old)

old = "        self.new_chunked_req = None\n"
assert s.count(old) == 1
s = s.replace(old, old + "        self._has_chunked_in_progress = False\n")
open(p, "w").write(s)
print("applied", p)
