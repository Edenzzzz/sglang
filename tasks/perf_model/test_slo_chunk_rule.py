"""Exercise the server-side SLO chunk rule off the cluster, so a guard bug costs seconds instead of an arm.

Why this exists. The rule has now been wrong twice in ways that no metric could distinguish from "the policy does
not help": the budget is a MINIMUM over resident decode rows, so a single row that is mishandled clamps every step
in the batch, and the symptom is simply a small chunk. The 2026-10-28 `:life` arm spent 30 minutes of a 16-GPU
cluster measuring a rule whose lifetime branch never once executed. Both bugs were in guards that are pure
arithmetic over per-row state, which means they are testable with fakes.

The method under test is lifted out of `apply_slo_budget_chunk_20261026.py` -- the same text the patch injects --
and executed against a shim with the attributes it reads, so this tests the deployed code rather than a
re-implementation of it.

  python test_slo_chunk_rule.py [--patch <apply_slo_budget_chunk_20261026.py>] [--model <step model json>]

Each case prints the budget and chunk the rule produces and PASS/FAIL against what the design requires.
Claude, 2026-10-28.
"""
import argparse
import re
import sys
import textwrap
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PATCH = (HERE.parent / "debug_kernel_server_perf/resume_20261012/unified"
                 / "apply_slo_budget_chunk_20261026.py")
DEFAULT_MODEL = HERE / "step_qwen_h20_a6_gpu_20261027.json"


class Req:
    """The fields the rule reads off a running request."""

    def __init__(self, prompt_tokens, out_tokens, spent_ms, now, o0=0):
        self.origin_input_ids = [0] * prompt_tokens
        self.output_ids = [0] * out_tokens
        self._slo_t0 = now - spent_ms / 1000.0
        self._slo_o0 = o0
        self.priority = None
        self.prefix_len = 0


class Batch:
    def __init__(self, reqs):
        self.reqs = reqs


def load_method(patch_path):
    src = open(patch_path).read()
    m = re.search(r"method = '''(.*?)'''", src, re.S)
    assert m, f"could not find the injected method in {patch_path}"
    body = textwrap.dedent(m.group(1))
    ns = {"os": __import__("os"), "time": __import__("time"),
          "logger": types.SimpleNamespace(info=lambda *a, **k: None)}
    exec(compile(body, str(patch_path), "exec"), ns)
    return ns["_slo_budget_chunk"]


class Waiting:
    """A queued request, as the `:qcap` cap sees it: only its deadline matters."""

    def __init__(self, due_in_s, now):
        self.priority = now + due_in_s
        self.origin_input_ids = [0] * 1000


class Shim:
    """Stands in for the Scheduler: the rule only touches these attributes."""

    def __init__(self, fn, reqs, tps=1.8, rate_ms=13.9, life=None, mixed_chunk=True, waiting=None):
        self._fn = fn
        self.running_batch = Batch(reqs)
        self.waiting_queue = waiting or []
        self.chunked_req = None
        # The rule prices the step differently when the prefill chunk shares the forward with the decode rows,
        # which is our four-mode step, so the default here is the configuration the cluster arms run.
        self.is_mixed_chunk = mixed_chunk
        # KV page size: the rule rounds the chunk it returns down to a page boundary.
        self.page_size = 64
        self.token_to_kv_pool_allocator = None
        self.tree_cache = None
        self._slo_chunk_tps = tps
        self._slo_rate = rate_ms
        # (mean output length, number of finished samples) per tenant class, as the rule accumulates it.
        self._slo_life = life if life is not None else {"chat": (400.0, 64), "agent": (300.0, 64)}
        self._slo_seen = {}

    def budget_chunk(self, chunk_size):
        return self._fn(self, chunk_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default=str(DEFAULT_PATCH))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--tail-ms", type=float, default=600.0)
    a = ap.parse_args()
    import os
    import time

    fn = load_method(a.patch)
    now = time.monotonic()
    # chat 50 ms / agent 80 ms mean TPOT, agent class above 24000 prompt tokens, 10% headroom.
    spec = f"{a.model}:0.1:50:80:24000:demote:ttft:qcap:stepcap={a.tail_ms:.0f}:life"
    os.environ["SGLANG_SLO_CHUNK"] = spec
    MAX = 16384
    rate = 13.9

    def healthy(n, o, prompt=1000):
        """n rows that have each emitted `o` tokens at the healthy rate."""
        return [Req(prompt, o, o * rate, now) for _ in range(n)]

    cases = []
    # A row that has produced nothing yet must not bound the step: its latency so far is TTFT. Before the fix this
    # case alone pinned the budget to the ~90 ms per-step allowance.
    cases.append(("one fresh row (o=0) beside healthy rows",
                  healthy(13, 40) + [Req(1000, 0, 250.0, now)], "big"))
    # The same failure one token later, via the demotion test on the instantaneous mean.
    cases.append(("one row at o=2 whose first step waited 600 ms",
                  healthy(13, 40) + [Req(1000, 2, 628.0, now)], "big"))
    cases.append(("all rows healthy and well into their output", healthy(14, 60), "big"))
    cases.append(("no rows resident at all", [], "max"))
    # A row that genuinely cannot reach its target by completion must be EXCLUDED from the minimum, not given the
    # tightest bound in it. Clamping on its behalf made overload self-reinforcing: at alpha 6 most rows are behind,
    # so nearly every step was pinned to the bare per-step allowance whatever the tail cap said, and the measured
    # stepcap sweep was flat from 600 to 3000 ms (chat TPOT p90 55.0-56.1 ms, TTFT p90 31-42 s, attainment
    # 23.6-35.3% against a fixed-2048 control's 60.5%). A row that has already missed cannot be rescued, so
    # constraining the rows that have not costs the prefill that would stop the queue growing.
    cases.append(("one row past rescue, which must NOT clamp the others (mean 120 ms against a 50 ms target)",
                  healthy(13, 40) + [Req(1000, 390, 390 * 120.0, now)], "big"))
    # ...but a batch of ONLY past-rescue rows must still leave the tail cap in force, or nothing bounds the step.
    cases.append(("every row past rescue: the tail cap is the only bound left",
                  [Req(1000, 390, 390 * 120.0, now) for _ in range(14)], "big"))
    # Before eight requests of a tenant have finished there is no projected length, so the rule falls back to the
    # banked bound. Rows well into their output have banked real slack, so a large chunk is correct there...
    cases.append(("lifetime length not learned, rows 60 tokens in (banked slack carries it)",
                  healthy(14, 60), "big", {"chat": (0.0, 3)}))
    # ...but with young rows the banked bound has almost nothing to give, which is exactly the regime the lifetime
    # projection exists for: without it the chunk collapses even though no row is in any danger.
    cases.append(("lifetime length not learned, rows 1 token in (fallback clamps)",
                  healthy(14, 1), "small", {"chat": (0.0, 3)}))
    cases.append(("lifetime length learned, rows 1 token in (projection keeps the chunk open)",
                  healthy(14, 1), "big"))

    # The `:qcap` cap, tested separately because it takes its bound from the WAITING queue rather than from the
    # resident rows. A queued request's deadline says when it must be ADMITTED, not what one step may cost, so
    # capping every step by the nearest deadline shrinks the chunk exactly when the queue is deep -- the same
    # category of error as judging a fresh row on a mean it does not have.
    print("\n-- the :qcap cap, against a queue of requests due soon")
    for due in (5.0, 1.0, 0.3, 0.1):
        sh = Shim(fn, healthy(14, 60), waiting=[Waiting(due, now) for _ in range(8)])
        print(f"   nearest queued deadline {due:>4.1f} s -> chunk {sh.budget_chunk(MAX):>6} tokens")

    fails = 0
    for case in cases:
        name, reqs, want = case[0], case[1], case[2]
        life = case[3] if len(case) > 3 else None
        sh = Shim(fn, reqs, life=life)
        out = sh.budget_chunk(MAX)
        if want == "max":
            ok = out == MAX
            need = f"the server maximum ({MAX})"
        elif want == "big":
            ok = out >= 1024
            need = ">= 1024 tokens"
        else:
            ok = out < 1024
            need = "< 1024 tokens (clamped)"
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        chunk {out:>6} tokens, need {need}")
    print(f"\n{len(cases) - fails}/{len(cases)} cases pass")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
