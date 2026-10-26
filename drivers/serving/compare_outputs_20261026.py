"""Greedy-output agreement between serving arms that replayed the same requests (chat.json from
run_native_serving_pair): fraction of requests with identical text, and the mean common-prefix length as a share of
the shorter output. Pass a null pair first (e.g. two repeats of one configuration) for the noise floor.
  python compare_outputs_20261026.py <armA> <armB> [<armC> <armD> ...]   (pairs)
Claude, 2026-10-26."""
import json
import sys


def load(a):
    d = json.load(open(f"{a}/requests/chat.json"))
    return {k: t for k, t, e in zip(d["request_keys"], d["generated_texts"], d["errors"]) if not e}


args = sys.argv[1:]
for a, b in zip(args[::2], args[1::2]):
    x, y = load(a), load(b)
    keys = sorted(set(x) & set(y))
    same = sum(1 for k in keys if x[k] == y[k])
    pref = []
    for k in keys:
        s, t = x[k], y[k]
        n = 0
        for c1, c2 in zip(s, t):
            if c1 != c2:
                break
            n += 1
        pref.append(n / max(1, min(len(s), len(t))))
    print(f"{a.rstrip('/').split('/')[-2]}/{a.rstrip('/').split('/')[-1]} vs {b.rstrip('/').split('/')[-2]}/{b.rstrip('/').split('/')[-1]}: "
          f"n={len(keys)} identical={same / max(1, len(keys)) * 100:.1f}% mean common prefix={sum(pref) / max(1, len(pref)) * 100:.1f}%")
