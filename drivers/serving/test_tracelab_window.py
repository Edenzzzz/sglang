#!/usr/bin/env python3
"""CPU-only checks for the TraceLab timestamp window, natural-text contents and the
start-at-timestamps replay rule.

  python3 test_tracelab_window.py [--trace PATH] [--tokenizer DIR] [--text-source JSON]
      [--reference ORIGINAL_tracelab.py] [--window-start S] [--stats-windows S1,S2,...]

Loads the benchmark modules by file path (the sglang package import needs the GPU stack).
--reference is the tracelab.py before the window/text changes; the default-path check
compares its rows with the edited loader's under the same seeds.
"""
import argparse
import asyncio
import gzip
import importlib.util
import json
import os
import random
import sys
import tempfile
import time
import types
from pathlib import Path
from types import SimpleNamespace

UP = Path("/opt/tiger/sglang-up/python/sglang/benchmark")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for package in ("sglang", "sglang.benchmark", "sglang.benchmark.datasets"):
    sys.modules.setdefault(package, types.ModuleType(package))
common = load_module("sglang.benchmark.datasets.common", UP / "datasets/common.py")
tracelab = load_module("tracelab_new", UP / "datasets/tracelab.py")
replay = load_module("session_replay_new", UP / "session_replay.py")

FAILURES = []


def check(condition, message):
    print(("PASS " if condition else "FAIL ") + message, flush=True)
    if not condition:
        FAILURES.append(message)


def row_key(row):
    return (
        tuple(row.prompt), row.prompt_len, row.output_len, row.timestamp, row.session_id,
        row.round_index, row.inter_round_delay_s, row.warmup_only,
    )


def make(module, path, **overrides):
    fields = dict(
        dataset_path=path, num_requests=60, num_groups=10, min_context_len=16384,
        max_context_len=65536, min_output_len=1,
    )
    fields.update(overrides)
    return module.TraceLabClaudeDataset(**fields)


def load_seeded(dataset, tokenizer, seed):
    random.seed(seed)
    return dataset.load(tokenizer)


def trace_index(path):
    index = {}
    with gzip.open(path, "rt") as source:
        for line in source:
            record = json.loads(line)
            index[(record["session_id"], int(record["round_index"]))] = record
    return index


def measured(rows):
    return [row for row in rows if not row.warmup_only]


def check_structure(rows, index, label):
    sessions = {}
    for row in measured(rows):
        sessions.setdefault(row.session_id, []).append(row)
    ordered = all(
        [row.round_index for row in group] == sorted(row.round_index for row in group)
        for group in sessions.values()
    )
    check(ordered, f"{label}: rounds emitted in round order within every session")
    lengths = all(
        row.prompt_len == len(row.prompt)
        == int(index[(row.session_id, row.round_index)]["input_tokens_total"])
        and row.output_len == int(index[(row.session_id, row.round_index)]["output_tokens"])
        for row in measured(rows)
    )
    check(lengths, f"{label}: input/output lengths equal the trace records")
    shared = checked = 0
    for group in sessions.values():
        for previous, row in zip(group, group[1:]):
            prefix = int(index[(row.session_id, row.round_index)]["prefix_tokens"])
            common_len = min(prefix, previous.prompt_len)
            checked += 1
            shared += row.prompt[:common_len] == previous.prompt[:common_len]
    check(checked > 0 and shared == checked,
          f"{label}: prefix ids repeat the previous round's ids ({shared}/{checked} pairs)")
    return sessions


def test_default_unchanged(reference, path, tokenizer):
    original = load_module("tracelab_reference", reference)
    configs = [
        ("uniform, global RNG", {}),
        ("rounds, selection seed", dict(session_weighting="rounds", selection_seed=20261014)),
        ("empty prefix allowed, min output 64",
         dict(allow_empty_prefix=True, min_output_len=64, num_requests=40, num_groups=8)),
    ]
    for label, overrides in configs:
        before = load_seeded(make(original, path, **overrides), tokenizer, 1234)
        after = load_seeded(make(tracelab, path, **overrides), tokenizer, 1234)
        after_state = random.getstate()
        load_seeded(make(original, path, **overrides), tokenizer, 1234)
        check(
            len(before) == len(after)
            and all(row_key(a) == row_key(b) for a, b in zip(before, after))
            and after_state == random.getstate(),
            f"default path ({label}): {len(after)} rows identical to the unmodified loader, "
            "global RNG left in the same state",
        )
        check(all(not hasattr(row, "trace_offset_s") for row in after),
              f"default path ({label}): rows carry no trace offset")


def test_window(path, tokenizer, index, window_start, text_source):
    base = dict(min_context_len=0, max_context_len=131072, min_output_len=1,
                allow_empty_prefix=True, window_start_s=window_start, window_s=3600.0)
    one = load_seeded(make(tracelab, path, time_scale=1.0, **base), tokenizer, 1)
    ten = load_seeded(make(tracelab, path, time_scale=10.0, **base), tokenizer, 1)
    m1, m10 = measured(one), measured(ten)
    check(len(m1) == len(m10) and [(r.session_id, r.round_index) for r in m1]
          == [(r.session_id, r.round_index) for r in m10],
          f"window: alpha=1 and alpha=10 select the same {len(m1)} rounds")
    check(all(abs(b.trace_offset_s - a.trace_offset_s / 10) < 1e-9 for a, b in zip(m1, m10)),
          "window: trace offsets scale by 1/alpha")
    check(all(abs(b.inter_round_delay_s - a.inter_round_delay_s / 10) < 1e-9 for a, b in zip(m1, m10)),
          "window: inter-round gaps scale by 1/alpha")
    sessions = check_structure(m1, index, "window alpha=1")
    firsts = [group[0].trace_offset_s for group in sessions.values()]
    check(all(0 <= offset < 3600 for offset in firsts),
          f"window: all {len(sessions)} sessions' first rounds start inside [0, 3600) s")
    check(all(row.trace_offset_s >= 0 for row in m1), "window: offsets nonnegative")

    text_args = dict(base, time_scale=10.0, text_source=text_source)
    natural = load_seeded(make(tracelab, path, **text_args), tokenizer, 1)
    again = load_seeded(make(tracelab, path, **text_args), tokenizer, 99)
    check([row_key(r) for r in natural] == [row_key(r) for r in again],
          "natural text: contents deterministic (independent of the global seed)")
    check_structure(natural, index, "natural text alpha=10")
    vocabulary_ids = set(common.get_available_tokens(tokenizer))
    rows = measured(natural)
    check(all(set(row.prompt) <= vocabulary_ids for row in rows[:50]),
          "natural text: ids are valid vocabulary ids")
    sample = tokenizer.decode(rows[0].prompt[:120])
    words = sample.split()
    alpha_words = sum(word.strip(".,;:!?\"'()").isalpha() for word in words)
    readable = lambda ids: (
        lambda words: len(words) > 10
        and sum(w.strip(".,;:!?\"'()").isalpha() and w.isascii() for w in words) / len(words) > 0.85
    )(tokenizer.decode(ids).split())
    natural_ok = sum(readable(row.prompt[:200]) for row in rows[:40])
    random_rows = measured(load_seeded(make(tracelab, path, time_scale=10.0, **base), tokenizer, 1))
    random_ok = sum(readable(row.prompt[:200]) for row in random_rows[:40])
    check(natural_ok >= 36 and random_ok == 0,
          f"natural text decodes to readable text: {natural_ok}/40 rows readable "
          f"(null case, random ids: {random_ok}/40)")
    print("  sample:", repr(sample[:300]))

    segment = measured(load_seeded(make(tracelab, path, time_scale=10.0, window_rounds="segment", **base),
                                   tokenizer, 1))
    check(segment and all(0 <= row.trace_offset_s < 360 for row in segment),
          f"segment mode: all {len(segment)} rounds start inside the window (offsets < 3600/alpha)")
    check_structure(segment, index, "segment alpha=10")
    warmups = [row for row in natural if row.warmup_only]
    expected_warmups = sum(
        int(index[(g[0].session_id, g[0].round_index)]["prefix_tokens"]) > 0 for g in sessions.values()
    )
    check(len(warmups) == expected_warmups and all(len(w.prompt) == w.prompt_len for w in warmups),
          f"natural text: {len(warmups)} warmup rows, one per session with a nonempty first prefix")


class Output:
    def __init__(self, success=True):
        self.success = success


def test_superposed(path, tokenizer, starts):
    """Superposed segments: each segment replays exactly as it would alone, on one shared clock."""
    base = dict(min_context_len=0, max_context_len=131072, min_output_len=1, allow_empty_prefix=True,
                window_s=3600.0, time_scale=10.0, window_rounds="segment", providers=("claude", "codex"))
    alone = [measured(load_seeded(make(tracelab, path, window_start_s=start, **base), tokenizer, 1))
             for start in starts]
    both = measured(load_seeded(make(tracelab, path, window_starts=tuple(starts), **base), tokenizer, 1))
    check(len(both) == sum(len(rows) for rows in alone),
          f"superposed: {len(both)} rounds = sum of the {len(starts)} segments alone")
    for index, rows in enumerate(alone):
        mine = [row for row in both if row.session_id.endswith(f"@{index}")]
        check([(row.session_id.rsplit("@", 1)[0], row.round_index) for row in mine]
              == [(row.session_id, row.round_index) for row in rows],
              f"superposed: segment {index} keeps its own {len(rows)} rounds in order, ids suffixed @{index}")
        check(all(abs(a.trace_offset_s - b.trace_offset_s) < 1e-9 and abs(a.inter_round_delay_s - b.inter_round_delay_s) < 1e-9
                  and a.prompt_len == b.prompt_len and a.output_len == b.output_len for a, b in zip(mine, rows)),
              f"superposed: segment {index} offsets, gaps and lengths equal its standalone replay")
        check(min(row.trace_offset_s for row in mine) >= 0 and min(row.trace_offset_s for row in mine) < 360.0,
              f"superposed: segment {index} starts on the shared clock within its scaled hour")
    try:
        make(tracelab, path, window_starts=tuple(starts), window_start_s=starts[0], **base).load(tokenizer)
        check(False, "superposed: --tracelab-window-starts with --tracelab-window-start-s is rejected")
    except ValueError:
        check(True, "superposed: --tracelab-window-starts with --tracelab-window-start-s is rejected")


def test_replay_horizon():
    """Rounds scheduled past BENCH_REPLAY_HORIZON_S are not issued; their outputs come from skipped_output."""
    plan = {"A": [0.00, 0.05, 0.40], "B": [0.30, 0.35]}  # horizon 0.2 s: A2 and all of B are past it
    requests = []
    for session, offsets in plan.items():
        for round_index, offset in enumerate(offsets):
            row = common.DatasetRow(prompt=[1], prompt_len=1, output_len=1, session_id=session,
                                    round_index=round_index, inter_round_delay_s=0.0)
            row.trace_offset_s = offset
            requests.append(row)

    async def send(row):
        await asyncio.sleep(0.01)
        return Output()

    skipped = []
    with tempfile.TemporaryDirectory() as directory:
        os.environ["BENCH_SESSION_AUDIT_FILE"] = os.path.join(directory, "a.json")
        os.environ["BENCH_REPLAY_HORIZON_S"] = "0.2"
        try:
            outputs = asyncio.run(replay.replay_sessions(
                requests, send, float("inf"), start_at_timestamps=True,
                skipped_output=lambda row: skipped.append(row) or "SKIPPED"))
        finally:
            del os.environ["BENCH_SESSION_AUDIT_FILE"], os.environ["BENCH_REPLAY_HORIZON_S"]
        report = json.loads(Path(directory, "a.json").read_text())
    issued = [(r.session_id, r.round_index) for r, o in zip(requests, outputs) if o != "SKIPPED"]
    check(issued == [("A", 0), ("A", 1)] and len(skipped) == 3 and report["rounds_not_issued"] == 3,
          "replay horizon: rounds past the horizon are not issued and come back as skipped outputs")


def test_replay_rule():
    # (offsets, gaps, service seconds) per session; times in seconds, kept short.
    plan = {
        "A": ([0.00, 0.10, 0.20], [0.0, 0.02, 0.02], 0.05),  # server keeps up: on trace offsets
        "B": ([0.05, 0.06, 0.07], [0.0, 0.03, 0.00], 0.10),  # completion-bound: finish + gap
        "C": ([0.30], [0.0], 0.02),
    }
    requests = []
    for session, (offsets, gaps, _) in plan.items():
        for round_index, (offset, gap) in reversed(list(enumerate(zip(offsets, gaps)))):
            row = common.DatasetRow(prompt=[1], prompt_len=1, output_len=1, session_id=session,
                                    round_index=round_index, inter_round_delay_s=gap)
            row.trace_offset_s = offset
            requests.append(row)

    async def send(row):
        await asyncio.sleep(plan[row.session_id][2])
        return Output()

    with tempfile.TemporaryDirectory() as directory:
        audit = os.path.join(directory, "agent_programs.json")
        os.environ["BENCH_SESSION_AUDIT_FILE"] = audit
        try:
            outputs = asyncio.run(replay.replay_sessions(
                requests, send, float("inf"), start_at_timestamps=True))
        finally:
            del os.environ["BENCH_SESSION_AUDIT_FILE"]
        report = json.loads(Path(audit).read_text())
    check(all(output is not None and output.success for output in outputs),
          "replay: every round completed, outputs in input order")
    tolerance = 0.015
    worst = 0.0
    ok = True
    for detail in report["sessions_detail"]:
        offsets, gaps, service = plan[detail["session_id"]]
        expected_issue = None
        previous_finish = None
        for position, entry in enumerate(detail["rounds"]):
            expected = offsets[position] if not position else max(previous_finish + gaps[position], offsets[position])
            worst = max(worst, abs(entry["issue_s"] - expected))
            ok &= entry["round_index"] == position and entry["issue_s"] >= entry["scheduled_s"] - 1e-6
            ok &= abs(entry["issue_s"] - expected) < tolerance
            ok &= not previous_finish or entry["issue_s"] >= previous_finish + gaps[position] - 1e-6
            previous_finish = entry["finish_s"]
        ok &= abs(detail["start_s"] - offsets[0]) < tolerance
    check(ok, f"replay: issue_r = max(finish_(r-1) + gap_r, offset_r), start = offset_0 "
              f"(worst deviation {worst * 1000:.1f} ms)")
    b = next(d for d in report["sessions_detail"] if d["session_id"] == "B")["rounds"]
    check(b[1]["issue_s"] - b[1]["trace_offset_s"] > 0.08,
          "replay: completion-bound round issued behind its trace offset (causal)")
    check(report["sessions"] == 3 and report["rounds"] == 7 and report["peak_active_programs"] >= 2
          and report["calls_per_s"] and report["session_starts_per_s"],
          f"replay audit: peak programs {report['peak_active_programs']}, mean "
          f"{report['mean_active_programs']:.2f}, calls/s {report['calls_per_s']:.1f}, "
          f"session starts/s {report['session_starts_per_s']:.1f}")

    async def failing(row):
        await asyncio.sleep(0.005)
        return Output(success=not (row.session_id == "A" and row.round_index == 1))

    issued = []

    async def recording(row):
        issued.append((row.session_id, row.round_index))
        return await failing(row)

    try:
        asyncio.run(replay.replay_sessions(requests, recording, float("inf"), start_at_timestamps=True))
        raised = False
    except RuntimeError:
        raised = True
    check(raised and ("A", 2) not in issued, "replay: a failed round stops its dependent rounds")
    try:
        asyncio.run(replay.replay_sessions(requests, send, 2.0, start_at_timestamps=True))
        rejected = False
    except ValueError:
        rejected = True
    check(rejected, "replay: a finite request rate is rejected in start-at-timestamps mode")


def print_window_stats(path, tokenizer, starts, max_context, alphas):
    for start in starts:
        for alpha in alphas:
            print(f"--- window start {start:g}s, max context {max_context}, alpha {alpha:g}")
            try:
                for mode in ("session", "segment"):
                    make(tracelab, path, min_context_len=0, max_context_len=max_context,
                         min_output_len=1, allow_empty_prefix=True, window_start_s=start,
                         time_scale=alpha, window_rounds=mode).load(tokenizer)
            except ValueError as error:
                print("  ", error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="/opt/tiger/sglang/workloads/tracelab/syfi_coding_trace.jsonl.gz")
    parser.add_argument("--tokenizer", default="/opt/tiger/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--text-source", default="/opt/tiger/sglang/workloads/ShareGPT_V3_unfiltered_cleaned_split.json")
    parser.add_argument("--reference", default="/tmp/tlwin_ref/tracelab_orig.py")
    parser.add_argument("--window-start", type=float, default=4130 * 3600.0)
    parser.add_argument("--stats-windows", default="")
    parser.add_argument("--superpose", default="17915400,14773800",
                        help="window starts for the superposed-segment check")
    parser.add_argument("--stats-max-context", type=int, default=131072)
    parser.add_argument("--stats-alphas", default="1,10")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    started = time.time()
    test_replay_rule()
    test_replay_horizon()
    if args.reference and Path(args.reference).exists():
        test_default_unchanged(args.reference, args.trace, tokenizer)
    else:
        check(False, f"reference loader {args.reference} missing; default path not compared")
    index = trace_index(args.trace)
    test_window(args.trace, tokenizer, index, args.window_start, args.text_source)
    test_superposed(args.trace, tokenizer, [float(x) for x in args.superpose.split(",")])
    if args.stats_windows:
        print_window_stats(args.trace, tokenizer, [float(s) for s in args.stats_windows.split(",")],
                           args.stats_max_context, [float(a) for a in args.stats_alphas.split(",")])
    print(f"{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILED'} in {time.time() - started:.0f}s")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
