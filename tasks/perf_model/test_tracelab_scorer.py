"""Check the TraceLab scorer's reported metrics against hand-computable records.

Why this exists. The 2026-10-29 metric change (SLO attainment plus mean TTFT and mean TPOT with a p10-p99 spread,
no token rate) rewrote the one line every multi-node result is read from, and the reader that judges arms against
each other parses that line by regex. A format slip is silent: the regex simply does not match, the field reads as
absent, and an arm appears to have no latency rather than a bad one. So the format is tested here rather than on
the cluster.

Three things are checked, each for a reason that cost something before:
  1. The mean is a MEAN. A median of right-skewed latencies sits well below it, and the two are not comparable; the
     fixture's TTFTs are built so the two differ by a wide margin, and the test fails if the reported value matches
     the median.
  2. An arm with NO records reports that fact and does not abort the batch. One dead pdmux arm's empty directory
     stopped a 40-arm re-scoring pass at arm six, discarding the rest.
  3. The reader recovers from the line exactly what the scorer put in it, on a round trip.

  python test_tracelab_scorer.py
Claude, 2026-10-29.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORER = HERE / "tracelab_windows.py"
READER = HERE.parent / "pd_colocation" / "superset_report.py"


def records(n, ttfts, itl_ms, out_len=21):
    """One tenant's record file. Every request emits out_len tokens at a constant gap, so its mean TPOT is exact."""
    return dict(request_start_epoch=[1000.0 + i for i in range(n)],
                errors=[""] * n, output_lens=[out_len] * n, ttfts=list(ttfts),
                itls=[[itl_ms / 1e3] * (out_len - 1) for _ in range(n)],
                input_lens=[2000] * n, cached_tokens=[500] * n)


def main():
    tmp = Path(tempfile.mkdtemp())
    arm = tmp / "coloc_cache_edf_mtp_hc_tl2_a3_fixture_r1_ours"
    arm.mkdir()
    # 90 fast requests and 10 very slow ones: mean 5.41 s against a median of 0.10 s, so a scorer reporting the
    # median cannot be mistaken for one reporting the mean.
    ttfts = [0.1] * 90 + [54.0] * 10
    mean_ttft = sum(ttfts) / len(ttfts)
    (arm / "chat.json").write_text(json.dumps(records(100, ttfts, 20.0)) + "\n")
    (arm / "agent.json").write_text(json.dumps(records(50, [1.0] * 50, 40.0)) + "\n")
    empty = tmp / "coloc_sgl_mtps_hc_tl2_a3_fixture_r1_dead"
    empty.mkdir()

    out = subprocess.run([sys.executable, str(SCORER), str(arm), str(empty)],
                         capture_output=True, text=True).stdout
    print(out)
    fails = []

    def check(cond, what):
        print(f"[{'PASS' if cond else 'FAIL'}] {what}")
        if not cond:
            fails.append(what)

    chat = out.split("chat:", 1)[1].split("|", 1)[0] if "chat:" in out else ""
    m = re.search(r"TTFT mean=([\d.]+)s \[p10 ([\d.]+) p99 ([\d.]+)\]", chat)
    check(bool(m), "the chat cell carries TTFT as a mean with a p10-p99 spread")
    if m:
        check(abs(float(m.group(1)) - mean_ttft) < 0.05,
              f"the reported TTFT {m.group(1)} is the mean ({mean_ttft:.2f}), not the median (0.10)")
        check(float(m.group(2)) < float(m.group(3)), "p10 is below p99")
    m = re.search(r"TPOT mean=([\d.]+)ms", chat)
    check(bool(m) and abs(float(m.group(1)) - 20.0) < 0.1,
          "mean TPOT is the per-token gap (20.0 ms) and not a per-request total")
    check("tok/s" not in out, "no token rate is reported")
    check("NO_RECORDS" in out, "the record-free arm reports NO_RECORDS")
    check(out.count("PAPER ") == 2, "the dead arm did not abort the batch: both arms produced a line")

    # Round trip: what the reader recovers must be what the scorer wrote.
    log = tmp / "fixture.log"
    log.write_text("".join("SCORE " + l + "\n" for l in out.splitlines() if l.startswith("PAPER "))
                   + f"ACCEPT {arm.name} accept=1.990 n=100\n")
    rd = subprocess.run([sys.executable, str(READER), str(log), "--ours", "ours_fixture"],
                        capture_output=True, text=True).stdout
    print(rd)
    check(f"{mean_ttft:.2f}" in rd, "the reader's table shows the same mean TTFT the scorer reported")
    check("*" not in rd, "the current format is not flagged as a legacy median")

    print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
