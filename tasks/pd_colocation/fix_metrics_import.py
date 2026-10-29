"""Repair metrics_reporter.py files whose injected import landed above `from __future__`.

The per-tenant acceptance patch prepended `import os as _os` to the top of the file. Python requires every
`from __future__` import to be the first statement after the docstring, so the prepend turned the module into a
SyntaxError and every scheduler process on the tree died at import. The symptom was "0/16 workers healthy", which
reads as a cluster or networking fault and sent two retries chasing pdmux parameters.

This moves the alias import below the last `from __future__` line, leaving the acceptance instrumentation in place,
and verifies the result compiles before writing it. Idempotent; prints one line per file.

Use `compile`, NOT `ast.parse`, to decide whether a file is broken. `ast.parse` builds the tree without applying
the rule that a `from __future__` import must come first, so it accepts this exact defect -- the first version of
this tool used it and reported "already-ok" for four trees whose workers were dying on that very line. A gate that
cannot see the fault it was written for is worse than no gate, because it reports success.
  python fix_metrics_import.py <file> [<file> ...]
Claude, 2026-10-29.
"""
import re
import sys

ALIAS = "import os as _os\n"
for path in sys.argv[1:]:
    src = open(path).read()
    if ALIAS not in src:
        print(f"{path} no-alias")
        continue
    try:
        compile(src, path, "exec")
        print(f"{path} already-ok")
        continue
    except SyntaxError:
        pass
    body = src.replace(ALIAS, "", 1)
    futures = list(re.finditer(r"^from __future__ import .*$", body, re.M))
    if not futures:
        print(f"{path} FAILED: no __future__ import; not the known defect")
        continue
    at = body.index("\n", futures[-1].end()) + 1
    fixed = body[:at] + ALIAS + body[at:]
    try:
        compile(fixed, path, "exec")
    except SyntaxError as e:
        print(f"{path} FAILED: still broken after the move: {e}")
        continue
    if "ACCEPT_TENANT" not in fixed:
        print(f"{path} FAILED: the acceptance instrumentation is gone; refusing to write")
        continue
    open(path, "w").write(fixed)
    print(f"{path} FIXED")
