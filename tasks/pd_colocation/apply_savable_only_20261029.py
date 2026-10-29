"""Exclude rows that cannot be rescued from the budget's minimum (2026-10-29).

The budget is a MIN over resident rows of each row's allowance
`T_r = s_r*L - spent - remaining*rate`. The demote test that is supposed to drop hopeless rows asks
`spent > s_r*L` -- a question about HISTORY. That misses the row that actually vetoes: a row ten tokens into its
output has spent almost nothing and so passes the test, while its PROJECTION `remaining*rate` at a slow observed
rate is hundreds of seconds, making T_r deeply negative and pinning the whole step.

Measured 2026-10-29: `budget_ms = -11731` with demotion firing on only ~10% of calls, and the rule returning a
chunk of ZERO on 41% of steps. Prefill starved, which is self-reinforcing.

Fix: judge savability by the budget's OWN formula. `T_r <= 0` means no chunk size whatever lets the row reach its
target -- not even a chunk of zero -- so it cannot be rescued and is excluded from the minimum instead of setting
it. That is SLOs-Serve section 4.1: price the step against requests that can still meet their deadlines, and serve
the rest from what is left over.

Offline, with every resident row unsavable, this takes the admitted chunk from 256 tokens (the bare floor) to
4753. `tasks/perf_model/test_slo_chunk_rule.py` pins both the floor and this case, 11/11.

  python apply_savable_only.py <tree>/sglang/srt
Idempotent; verifies with compile() and prints a marker count.
"""
import sys

path = f"{sys.argv[1]}/managers/scheduler.py"
src = open(path).read()
if "slo_chunk_unsavable" in src:
    print("already marker=1")
    raise SystemExit(0)

OLD = """                    remaining = max(0.0, L - o - tps)
                    budget = min(budget, s_r * L - spent - remaining * rate)
                    continue"""
assert src.count(OLD) == 1, f"budget site not found exactly once (count={src.count(OLD)})"
NEW = """                    remaining = max(0.0, L - o - tps)
                    _t_r = s_r * L - spent - remaining * rate
                    if cfg["demote"] and _t_r <= 0.0:
                        # Not savable by the budget's own formula: no chunk size, not even zero, lets this row
                        # reach its target. Excluded from the minimum rather than setting it, so one hopeless row
                        # cannot veto prefill for every other row. The demote test above asks about history
                        # (spent > s*L) and misses exactly this case, where spent is small but the projection
                        # remaining*rate is huge.
                        self.slo_chunk_unsavable = getattr(self, "slo_chunk_unsavable", 0) + 1
                        continue
                    budget = min(budget, _t_r)
                    continue"""
src = src.replace(OLD, NEW, 1)
compile(src, path, "exec")
open(path, "w").write(src)
print(f"patched marker={src.count('slo_chunk_unsavable')}")
