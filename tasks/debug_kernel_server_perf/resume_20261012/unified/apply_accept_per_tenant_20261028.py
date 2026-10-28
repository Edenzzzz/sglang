"""Log mean draft acceptance SEPARATELY PER TENANT, so an arm's acceptance can be compared like for like.

Why this is needed. At load factor 6 our arm (L1) reported mean acceptance 1.982 against sglang's
prefill-prioritized arm's 1.795, a 9.4% gap. Acceptance scales forwards per output token almost one for one, so a
gap that size accounts for essentially all of the +9.3% agent SLO goodput margin we would otherwise claim -- the
margin and the artefact are indistinguishable at this resolution. Two mundane explanations have been ruled out by
reading the arms' own logs: the rope base is correct in both (`rope_theta 1000000`), and the speculative
configuration is identical (EAGLE3, eagle_topk 1, num_draft_tokens 4, num_steps 3, accept thresholds 1.0). What
remains is either the request mix -- the arms served different numbers of agent rounds, and acceptance rises with
cached context, so a mix shifted toward short chat depresses the mean on its own -- or a real effect of our merged
four-mode step on what the draft model sees.

`scheduler_metrics_mixin.py` reports one mean over all requests, which cannot separate those. This patch adds a
per-class mean, split on the same prompt-length threshold the SLO machinery uses to tell an agent round from a chat
turn, so the comparison can be made within a class:

  - matching per-tenant acceptance across arms => the gap is mix, and the goodput margin is real;
  - differing per-tenant acceptance => the merged step changes what the draft sees, which is a finding about C2
    itself rather than about the schedule, and the margin must be attributed to it rather than to the policy.

Either outcome is useful, which is why this is worth an instrumentation pass rather than a re-run of the ladder.

  python apply_accept_per_tenant_20261028.py <tree>/python/sglang/srt

Off unless SGLANG_ACCEPT_PER_TENANT is set, so no other arm's output changes. Emits
`ACCEPT_TENANT chat=<mean>/<n> agent=<mean>/<n> threshold=<tokens>` on the same cadence as the existing line.

DO NOT apply this to a tree while an arm is running against it: the serving trees are shared between arms of one
driver, and patching one mid-chain invalidates whatever is in flight without any visible error.
Claude, 2026-10-28.
"""
import re
import sys

path = f"{sys.argv[1]}/managers/scheduler_metrics_mixin.py"
source = open(path).read()
if "ACCEPT_TENANT" in source:
    print("already")
    raise SystemExit(0)

anchor = '            msg += f"accept len: {spec_accept_length:.2f}, accept rate: {spec_accept_rate:.2'
assert source.count(anchor) == 1, "the acceptance log line moved; re-read the file before patching"
line_end = source.index("\n", source.index(anchor))

# The split uses the same threshold the SLO chunk rule uses to classify a row, read from the same environment
# variable's fifth field, so the two never disagree about what an "agent" request is.
addition = '''
            if _os.environ.get("SGLANG_ACCEPT_PER_TENANT"):
                try:
                    _spec = _os.environ.get("SGLANG_SLO_CHUNK", "")
                    _thresh = int(_spec.split(":")[4]) if len(_spec.split(":")) > 4 else 24000
                    _acc = getattr(self, "_accept_by_tenant", None)
                    if _acc is None:
                        _acc = self._accept_by_tenant = {"chat": [0.0, 0], "agent": [0.0, 0]}
                    for _req in (self.running_batch.reqs if self.running_batch is not None else []):
                        _cls = "agent" if len(_req.origin_input_ids) >= _thresh else "chat"
                        _acc[_cls][0] += spec_accept_length
                        _acc[_cls][1] += 1
                    msg += " ACCEPT_TENANT " + " ".join(
                        f"{_k}={(_v[0] / _v[1]):.3f}/{_v[1]}" if _v[1] else f"{_k}=none"
                        for _k, _v in _acc.items()) + f" threshold={_thresh}"
                except Exception:
                    pass'''
# The trailing newline matters: without it `pass` concatenates with the following statement.
source = source[:line_end + 1] + addition.lstrip("\n") + "\n" + source[line_end + 1:]

if not re.search(r"^import os\b", source, re.M):
    source = "import os as _os\n" + source
else:
    source = source.replace("import os\n", "import os\nimport os as _os\n", 1)

open(path, "w").write(source)
print("patched per-tenant acceptance")
