"""Validity of a cond_disagg arm from router_decisions.jsonl: share of requests prefilled remotely, and for those the
decode worker's measured cached fraction (cached_tokens / prompt_tokens) -- near 1 means the remote prefill's KV
reached the decode worker through the pool; near 0 means the decode worker recomputed it (an invalid baseline).
  python cond_disagg_check.py <arm log dir>
Claude, 2026-10-24."""
import json
import sys

route, done = {}, {}
for line in open(sys.argv[1] + "/router_decisions.jsonl"):
    r = json.loads(line)
    (done if "done" in r else route)[r["rid"]] = r
remote = [k for k, r in route.items() if r.get("remote_prefill")]
fr = [done[k]["cached_tokens"] / done[k]["prompt_tokens"] for k in remote
      if k in done and done[k].get("prompt_tokens")]
local_long = [k for k, r in route.items() if not r.get("remote_prefill") and r.get("n_tok", 0) > 8192]
print(f"COND_CHECK requests={len(route)} remote_prefill={len(remote)} ({100 * len(remote) / max(1, len(route)):.1f}%) "
      f"decode-side cached fraction on remote: mean={sum(fr) / max(1, len(fr)):.3f} "
      f"<0.5: {sum(1 for f in fr if f < 0.5)}/{len(fr)} | long prompts served local: {len(local_long)}")
