"""Add an AGENT-DOMINANT mixed-step set to the generator (2026-10-29).

The existing `longtail` set is 90% chat inputs plus 10% agent inputs of 64K-200K, which inverts the mix the
evaluation is agreed on: agent inputs dominate the workload we claim (>70%, user 2026-10-29). A sweep on the
inverted mix measures heterogeneity, but it is not the composition our headline numbers should come from, because
the share of wide-KV inputs is exactly what decides how much tile waste there is to remove.

`agentmix` keeps the same construction as `longtail` and only changes that share: 75% of inputs are agent rounds of
64K-200K tokens, the remaining 25% chat inputs from the ShareGPT distribution. Everything else -- the 512-token
prefill chunk on a 4096-token context, the Qwen2.5 head geometry, the two-mode and four-mode variants -- is
unchanged, so it is directly comparable with the sets already measured.

  python add_agent_dominant.py <ub dir>

Idempotent.
"""
import sys

path = f"{sys.argv[1]}/gen_synth_meld_20261025.py"
src = open(path).read()
if "agentmix" in src:
    print("already")
    raise SystemExit(0)

# 1. the KV-length branch: treat agentmix like longtail for the base distribution...
OLD_BRANCH = '''    else:  # longtail
        p, r = lognormal(929, 1264, n), lognormal(466, 652, n)'''
NEW_BRANCH = '''    else:  # longtail, agentmix
        p, r = lognormal(929, 1264, n), lognormal(466, 652, n)'''
assert OLD_BRANCH in src, "the kv_lengths branch moved"
src = src.replace(OLD_BRANCH, NEW_BRANCH, 1)

# 2. ...but with the agent share raised from a tenth to three quarters.
OLD_FRAC = '''    if kind == "longtail":
        agent = torch.rand(n, generator=g) < 0.1
        agent[0] = True'''
NEW_FRAC = '''    if kind in ("longtail", "agentmix"):
        # The share of wide-KV inputs is what decides how much tile waste exists to remove, so it is the one
        # parameter that must match the evaluated workload rather than a convenient default.
        agent = torch.rand(n, generator=g) < (0.75 if kind == "agentmix" else 0.1)
        agent[0] = True'''
assert OLD_FRAC in src, "the agent-share block moved"
src = src.replace(OLD_FRAC, NEW_FRAC, 1)

# 3. the batch plan: same batch sizes as longtail, which sit in our serving regime.
OLD_PLAN = '''"longtail": (16, 32, 64, 128),'''
NEW_PLAN = '''"longtail": (16, 32, 64, 128), "agentmix": (16, 32, 64, 128),'''
assert OLD_PLAN in src, "the PLAN dict moved"
src = src.replace(OLD_PLAN, NEW_PLAN, 1)

src = src.replace("  longtail      90% ShareGPT rows + 10% agent rows of 64K-200K: the imbalanced case",
                  "  longtail      90% ShareGPT rows + 10% agent rows of 64K-200K: the imbalanced case\n"
                  "  agentmix      75% agent rows of 64K-200K + 25% ShareGPT: the evaluated mix, agent-dominant", 1)
compile(src, path, "exec")
open(path, "w").write(src)
print("added agentmix (75% agent)")
