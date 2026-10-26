"""Diagnostic: SGLANG_PCG_REASON=1 counts which eligibility check makes the prefill CUDA-graph runner refuse a batch
(every `return False` in can_replay_locally / can_run_graph gets a tag: the source line of its condition), and logs
the tally every 200 refusals plus the replay count. Used to find why the 4-mode merged step runs eager. Idempotent.
  python apply_pcg_reason_20261026.py <run root>
Claude, 2026-10-26."""
import re
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py"
s = p.read_text()
if "_pcg_no(" not in s:
    a = s.index("    def can_replay_locally(")
    b = s.index("    def _build_capture_spec_info(")
    body = s[a:b].split("\n")
    out, cond = [], ""
    for i, line in enumerate(body):
        st = line.strip()
        if st.startswith("if ") or st.startswith("and ") or st.startswith("or ") or st.startswith("not "):
            cond = re.sub(r"[^A-Za-z0-9_]+", "_", st)[:48]
        if st == "return False":
            out.append(line.replace("return False", f'return self._pcg_no("{i}:{cond}")'))
        elif st == "return True" and "def can_run_graph" in "\n".join(body[:i]) and \
                "def can_replay_locally" not in "\n".join(body[max(0, i - 3):i]):
            out.append(line.replace("return True", "return self._pcg_yes()"))
        else:
            out.append(line)
    helper = '''    def _pcg_no(self, tag):
        if os.environ.get("SGLANG_PCG_REASON") == "1":
            mr = getattr(self, "model_runner", None)
            role = "draft" if getattr(mr, "is_draft_worker", False) else "target"
            mode = getattr(getattr(self, "capture_hidden_mode", None), "name", "?")
            tag = f"{role}/{mode}/{tag}"
            c = self.__dict__.setdefault("_pcg_counts", {})
            c[tag] = c.get(tag, 0) + 1
            n = sum(c.values())
            if n % 200 == 1:
                logger.info("PCG_REASON refused=%d replayed=%d %s", n, self.__dict__.get("_pcg_ok", 0),
                            sorted(c.items(), key=lambda kv: -kv[1])[:6])
        return False

    def _pcg_yes(self):
        if os.environ.get("SGLANG_PCG_REASON") == "1":
            self.__dict__["_pcg_ok"] = self.__dict__.get("_pcg_ok", 0) + 1
            n = self.__dict__["_pcg_ok"]
            if n % 500 == 1:
                mr = getattr(self, "model_runner", None)
                logger.info("PCG_REPLAY role=%s replayed=%d", "draft" if getattr(mr, "is_draft_worker", False) else "target", n)
        return True

'''
    s = s[:a] + helper + "\n".join(out) + s[b:]
    lines = s.split("\n")
    if "import os" not in lines:
        # after the module's __future__ imports, which must stay first
        k = max([i for i, l in enumerate(lines) if l.startswith("from __future__")], default=-1)
        lines.insert(k + 1, "import os")
        s = "\n".join(lines)
    assert "\nlogger = " in s, "runner has no module logger"
    compile(s, str(p), "exec")
    p.write_text(s)
print("PCG_REASON", "_pcg_no(" in p.read_text(), p.read_text().count("_pcg_no("))
