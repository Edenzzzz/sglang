"""One EDF clock per scheduling iteration for every TP rank (serving tree patch, idempotent).

Under tensor parallelism each TP rank runs its own scheduler and must take the same decisions. The EDF code read each
rank's own time.monotonic() (arrival stamps, late demotion, deadline-sized chunks, ITL slack, decode observations), so a
deadline near "now" could be judged differently on different ranks; the ranks then built different batches and their
collectives deadlocked (GLM-4.5-Air TP4 stage 2, 2026-10-25: rank 0 ran 2 requests while ranks 1-2 ran 4, scheduler
watchdog after 600 s). With this patch rank 0 reads the clock once per iteration and ships it inside the request
broadcast it already makes; every EDF decision on every rank uses that value (edf_now()). TP = 1 is unchanged.
  python apply_edf_tp_clock_20261025.py <tree containing python/sglang>
Claude, 2026-10-25.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "python/sglang/srt/managers"
MARK = "edf_now()"


def patch(path, edits):
    text = path.read_text()
    if "EDF_TP_CLOCK" in text:
        print("already applied:", path)
        return
    for old, new, count in edits:
        n = text.count(old)
        if n != count:
            raise SystemExit(f"{path}: expected {count} x {old!r}, found {n}")
        text = text.replace(old, new)
    path.write_text(text)
    print("patched:", path)


edf = root / "edf_scheduler.py"
patch(edf, [
    ("import numpy as np\n",
     "import numpy as np\n\n"
     "# EDF_TP_CLOCK: the scheduling iteration's clock, read once on TP rank 0 and broadcast with the received requests\n"
     "# (request_receiver), so every TP rank takes the same deadline decisions. None = this rank's own clock (TP = 1).\n"
     "_EDF_CLOCK = None\n\n\n"
     "class EdfClock:\n"
     "    \"\"\"Marker carried in the request broadcast: rank 0's time.monotonic() for this iteration.\"\"\"\n\n"
     "    def __init__(self, t):\n"
     "        self.t = t\n\n\n"
     "def edf_set_clock(t):\n"
     "    global _EDF_CLOCK\n"
     "    _EDF_CLOCK = t\n\n\n"
     "def edf_now():\n"
     "    return _EDF_CLOCK if _EDF_CLOCK is not None else time.monotonic()\n", 1),
    ("    now = time.monotonic()\n", "    now = edf_now()\n", 2),
    # last_tic is stamped by each rank's own clock: with the shared clock in use, fall back to it instead.
    ("getattr(req, \"last_tic\", time.monotonic())",
     "(edf_now() if _EDF_CLOCK is not None else getattr(req, \"last_tic\", time.monotonic()))", 2),
    ("    req._edf_arrival_monotonic = time.monotonic()\n", "    req._edf_arrival_monotonic = edf_now()\n", 1),
])

sched = root / "scheduler.py"
text = sched.read_text()
if "EDF_TP_CLOCK" not in text:
    lines = text.split("\n")
    # Every time.monotonic() inside the EDF methods (between the first and last EDF helper) becomes edf_now().
    first = next(i for i, l in enumerate(lines) if "def _edf_slack_step_ms" in l)
    last = next(i for i, l in enumerate(lines) if "def _edf_observe_batch" in l)
    end = next(i for i in range(last + 1, len(lines)) if lines[i].startswith("    def "))
    n = 0
    for i in range(first, end):
        if "time.monotonic()" in lines[i]:
            lines[i] = lines[i].replace("time.monotonic()", "_edf_now()")
            n += 1
    if n < 4:
        raise SystemExit(f"scheduler.py: expected >= 4 EDF clock reads, found {n}")
    text = "\n".join(lines)
    anchor = "\nlogger = logging.getLogger(__name__)\n"
    if text.count(anchor) != 1:
        raise SystemExit("scheduler.py: logger anchor not found once")
    text = text.replace(anchor, anchor + "\n\ndef _edf_now():  # EDF_TP_CLOCK: one clock per iteration on every TP rank\n"
                        "    from sglang.srt.managers.edf_scheduler import edf_now\n\n    return edf_now()\n", 1)
    sched.write_text(text)
    print(f"patched: {sched} ({n} EDF clock reads)")
else:
    print("already applied:", sched)

recv = root / "scheduler_components/request_receiver.py"
patch(recv, [
    ("""            if recv_reqs is not None:
                recv_reqs = [*recv_reqs, *local_reqs]
            if get_parallel().tp_size != 1:
                tp_group = get_parallel().tp_group
                recv_reqs = broadcast_pyobj(
                    recv_reqs,
                    tp_group.rank,
                    tp_group.cpu_group,
                    src=tp_group.ranks[0],
                )
""", """            if recv_reqs is not None:
                recv_reqs = [*recv_reqs, *local_reqs]
            if get_parallel().tp_size != 1:
                # EDF_TP_CLOCK: rank 0's clock for this iteration rides the broadcast, so every TP rank's EDF decisions
                # read the same time.
                from sglang.srt.managers.edf_scheduler import EdfClock, edf_set_clock

                if recv_reqs is not None:
                    recv_reqs.append(EdfClock(time.monotonic()))
                tp_group = get_parallel().tp_group
                recv_reqs = broadcast_pyobj(
                    recv_reqs,
                    tp_group.rank,
                    tp_group.cpu_group,
                    src=tp_group.ranks[0],
                )
                clocks = [r for r in recv_reqs if isinstance(r, EdfClock)]
                if clocks:
                    edf_set_clock(clocks[-1].t)
                recv_reqs = [r for r in recv_reqs if not isinstance(r, EdfClock)]
""", 1),
])
rt = recv.read_text()
if "\nimport time\n" not in rt:
    rt = rt.replace("\nimport logging\n", "\nimport logging\nimport time\n", 1) if "\nimport logging\n" in rt else "import time\n" + rt
    recv.write_text(rt)
print("EDF_TP_CLOCK_APPLIED")
