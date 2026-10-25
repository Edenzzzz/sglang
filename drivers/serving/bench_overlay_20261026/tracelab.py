import gzip
import json
import math
import random
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

from sglang.benchmark.datasets.common import (
    BaseDataset,
    DatasetRow,
    get_available_tokens,
)


@dataclass
class TraceLabClaudeDataset(BaseDataset):
    """Synthetic token contents with recorded TraceLab lengths and session ordering.

    This is a length/session trace replay, not a replay of original agent text.
    Selected request lengths are neither clipped nor retokenized. Token contents are random
    vocabulary ids, or with text_source sequential tokenized ShareGPT text; either way a
    round's first prefix_tokens ids repeat the previous replayed round's ids.

    Two selections: num_groups sampled sessions with num_requests rounds in total (default),
    or, with window_start_s set, every session whose first eligible round starts inside the
    timestamp window, all its eligible rounds kept (window_rounds="segment": only rounds starting
    inside the window); num_groups/num_requests then become
    outputs and each row carries trace_offset_s for session_replay's start-at-timestamps mode.
    """

    dataset_path: str
    num_requests: int
    num_groups: int
    min_context_len: int
    max_context_len: int
    min_output_len: int
    # Full-release replay: which providers to keep, and whether a session's first round (empty
    # prefix) is replayable. Defaults reproduce every earlier Claude-only, prefix>0 run.
    providers: tuple = ("claude",)
    allow_empty_prefix: bool = False
    # "rounds" picks sessions with probability proportional to their eligible round count, so the
    # replayed rounds follow the release's ROUND-level context distribution (uniform session
    # picks under-sample long contexts, which live in long sessions: median 51K vs 124K).
    session_weighting: str = "uniform"
    # Seed for session and window selection only (token draws stay on the global RNG), so a
    # selection checked offline against the release's distribution replays unchanged.
    selection_seed: Optional[int] = None
    # Timestamp-window replay (Crossflow protocol): keep every session whose first eligible
    # round starts in [window_start_s, window_start_s + window_s) seconds after the trace's
    # first round, with all its eligible rounds; offsets and gaps are divided by time_scale.
    # None keeps the sampled-session mode above.
    window_start_s: Optional[float] = None
    window_s: float = 3600.0
    time_scale: float = 1.0
    # "session": sessions whose first eligible round starts in the window, all their rounds.
    # "segment": only rounds starting inside the window (a Crossflow-style one-hour segment;
    # sessions already running at the window start join mid-session).
    window_rounds: str = "session"
    # ShareGPT JSON whose tokenized text fills the token contents instead of random vocabulary.
    text_source: Optional[str] = None

    @classmethod
    def from_args(cls, args):
        if not args.tokenize_prompt:
            raise ValueError("tracelab-claude requires --tokenize-prompt to preserve lengths")
        return cls(
            args.dataset_path,
            args.num_prompts,
            args.tracelab_num_groups,
            args.tracelab_min_context_len,
            args.tracelab_max_context_len,
            args.tracelab_min_output_len,
            tuple(p for p in getattr(args, "tracelab_providers", "claude").split(",") if p),
            bool(getattr(args, "tracelab_allow_empty_prefix", False)),
            getattr(args, "tracelab_session_weighting", "uniform"),
            getattr(args, "tracelab_selection_seed", None),
            getattr(args, "tracelab_window_start_s", None),
            float(getattr(args, "tracelab_window_s", 3600.0)),
            float(getattr(args, "tracelab_time_scale", 1.0)),
            getattr(args, "tracelab_window_rounds", "session"),
            getattr(args, "tracelab_text_source", None),
        )

    def load(self, tokenizer, model_id=None):
        if not self.dataset_path:
            raise ValueError("--dataset-path must identify a TraceLab JSONL(.gz) file")
        windowed = self.window_start_s is not None
        if windowed:
            if not (math.isfinite(self.window_start_s) and self.window_start_s >= 0):
                raise ValueError("--tracelab-window-start-s must be finite and nonnegative")
            if not (math.isfinite(self.window_s) and self.window_s > 0):
                raise ValueError("--tracelab-window-s must be finite and positive")
            if not (math.isfinite(self.time_scale) and self.time_scale > 0):
                raise ValueError("--tracelab-time-scale must be finite and positive")
            if self.window_rounds not in ("session", "segment"):
                raise ValueError("--tracelab-window-rounds must be session or segment")
        elif not 0 < self.num_groups <= self.num_requests:
            raise ValueError("--tracelab-num-groups must be in [1, num_prompts]")
        if self.min_context_len < 0 or self.max_context_len < self.min_context_len:
            raise ValueError("Invalid TraceLab context-length selection interval")
        opener = gzip.open if self.dataset_path.endswith(".gz") else open
        sessions = {}
        trace_origin_ms = None
        with opener(self.dataset_path, "rt") as source:
            for line in source:
                record = json.loads(line)
                if record.get("provider") not in self.providers:
                    continue
                if windowed:
                    record_start_ms, _ = _record_span_ms(record)
                    if record_start_ms is not None and (
                        trace_origin_ms is None or record_start_ms < trace_origin_ms
                    ):
                        trace_origin_ms = record_start_ms
                context_length = int(record.get("input_tokens_total") or 0)
                output_length = int(record.get("output_tokens") or 0)
                prefix_length = int(record.get("prefix_tokens") or 0)
                append_length = int(record.get("newly_append_tokens") or 0)
                if not (
                    self.min_context_len <= context_length <= self.max_context_len
                    and output_length >= self.min_output_len
                    and (prefix_length > 0 or (self.allow_empty_prefix and prefix_length == 0))
                    and append_length > 0
                ):
                    continue
                if context_length != prefix_length + append_length:
                    raise ValueError("TraceLab input length differs from prefix plus append")
                session_id = record.get("session_id")
                if not isinstance(session_id, str) or not session_id:
                    raise ValueError("TraceLab rows require a nonempty session_id")
                sessions.setdefault(session_id, []).append(record)

        if windowed:
            selected, session_records, window_start_ms = self._select_window(
                sessions, trace_origin_ms
            )
        else:
            quotas = [
                (index + 1) * self.num_requests // self.num_groups
                - index * self.num_requests // self.num_groups
                for index in range(self.num_groups)
            ]
            eligible = [
                session_id for session_id, records in sessions.items()
                if len(records) >= max(quotas)
            ]
            if len(eligible) < self.num_groups:
                raise ValueError(
                    f"TraceLab selection has {len(eligible)} sufficiently long sessions; "
                    f"requested {self.num_groups}"
                )
            selector = random if self.selection_seed is None else random.Random(self.selection_seed)
            if self.session_weighting == "rounds":
                # Weighted sampling without replacement (Efraimidis-Spirakis keys), seeded like
                # the uniform path.
                keyed = sorted(
                    eligible, key=lambda sid: selector.random() ** (1.0 / len(sessions[sid])), reverse=True
                )
                selected = keyed[: self.num_groups]
            else:
                selected = selector.sample(eligible, self.num_groups)
        vocabulary = get_available_tokens(tokenizer)
        if not vocabulary:
            raise ValueError("Tokenizer vocabulary is empty")
        if self.text_source:
            text_stream = NaturalTextTokenStream(self.text_source, tokenizer)
            draw = text_stream.take
        else:
            text_stream = None
            draw = lambda count: random.choices(vocabulary, k=count)

        requests = []
        warmups = []
        if windowed:
            plan = [(session_id, session_records[session_id]) for session_id in selected]
        else:
            # Lazy, so each session's slice draw stays interleaved with its token draws on the
            # global RNG exactly as before.
            def sampled_plan():
                for session_id, quota in zip(selected, quotas):
                    records = sorted(sessions[session_id], key=lambda record: record["round_index"])
                    window_start = selector.randint(0, len(records) - quota)
                    yield session_id, records[window_start : window_start + quota]

            plan = sampled_plan()
        for session_id, records in plan:
            first_prefix = int(records[0]["prefix_tokens"])
            prompt_ids = draw(first_prefix)
            if first_prefix > 0:
                warmup_rng = random.Random(f"tracelab-warmup:{session_id}:{first_prefix}")
                warmups.append(DatasetRow(
                    prompt=warmup_rng.choices(vocabulary, k=first_prefix),
                    prompt_len=first_prefix,
                    output_len=1,
                    session_id=session_id,
                    warmup_only=True,
                ))
            previous_end_ms = None
            for record in records:
                prefix_length = int(record["prefix_tokens"])
                append_length = int(record["newly_append_tokens"])
                if len(prompt_ids) < prefix_length:
                    prefix_ids = prompt_ids + draw(prefix_length - len(prompt_ids))
                else:
                    prefix_ids = prompt_ids[:prefix_length]
                prompt_ids = prefix_ids + draw(append_length)
                start_ms, end_ms = _record_span_ms(record)
                gap = (
                    max(0.0, (start_ms - previous_end_ms) / 1000)
                    if start_ms is not None and previous_end_ms is not None else 0.0
                )
                previous_end_ms = end_ms
                row = DatasetRow(
                    prompt=prompt_ids,
                    prompt_len=len(prompt_ids),
                    output_len=int(record["output_tokens"]),
                    timestamp=start_ms,
                    session_id=session_id,
                    round_index=int(record["round_index"]),
                    inter_round_delay_s=gap,
                )
                if windowed:
                    row.inter_round_delay_s = gap / self.time_scale
                    # Scaled offset from the window start; session_replay's
                    # start-at-timestamps mode issues rounds against it.
                    row.trace_offset_s = (start_ms - window_start_ms) / 1000 / self.time_scale
                requests.append(row)
        replayed = sorted(row.prompt_len for row in requests)
        if replayed:
            pick = lambda q: replayed[min(len(replayed) - 1, int(q * len(replayed)))]
            print(
                "TraceLab replayed context p10/p50/p90/p99: "
                f"{pick(0.1)}/{pick(0.5)}/{pick(0.9)}/{pick(0.99)}; providers={','.join(self.providers)}"
            )
        if windowed:
            self.num_groups = len(plan)
            self.num_requests = len(requests)
            self._print_window_stats(plan, window_start_ms)
        contents = (
            f"natural text from {self.text_source} ({text_stream.describe()})"
            if text_stream is not None else "synthetic token contents"
        )
        print(
            f"TraceLab length/session replay: {len(requests)} rounds, "
            f"{self.num_groups} sessions; {contents}, "
            "recorded input/output lengths preserved"
        )
        return warmups + requests

    def _select_window(self, sessions, trace_origin_ms):
        if trace_origin_ms is None:
            raise ValueError("TraceLab window selection requires timing_events timestamps")
        window_start_ms = trace_origin_ms + self.window_start_s * 1000
        window_end_ms = window_start_ms + self.window_s * 1000
        firsts = []
        session_records = {}
        inside = lambda record: window_start_ms <= _record_span_ms(record)[0] < window_end_ms
        for session_id, records in sessions.items():
            records = sorted(records, key=lambda record: record["round_index"])
            if self.window_rounds == "segment":
                if any(_record_span_ms(record)[0] is None for record in records):
                    raise ValueError(f"TraceLab session {session_id} has an untimed round")
                records = [record for record in records if inside(record)]
                if not records:
                    continue
            first_start_ms, _ = _record_span_ms(records[0])
            if first_start_ms is None:
                raise ValueError(f"TraceLab session {session_id} first eligible round has no timestamp")
            if window_start_ms <= first_start_ms < window_end_ms:
                if any(_record_span_ms(record)[0] is None for record in records):
                    raise ValueError(f"TraceLab session {session_id} has an untimed round")
                session_records[session_id] = records
                firsts.append((first_start_ms, session_id))
        if not firsts:
            raise ValueError(
                f"TraceLab window [{self.window_start_s}, {self.window_start_s + self.window_s}) s "
                f"contains no eligible {self.window_rounds} rounds"
            )
        firsts.sort()
        return [session_id for _, session_id in firsts], session_records, window_start_ms

    def _print_window_stats(self, plan, window_start_ms):
        window_end_ms = window_start_ms + self.window_s * 1000
        rounds = [record for _, records in plan for record in records]
        inside = [record for record in rounds if _record_span_ms(record)[0] < window_end_ms]
        span_s = max(
            (_record_span_ms(record)[0] - window_start_ms) / 1000 for record in rounds
        )
        append = sum(int(record["newly_append_tokens"]) for record in inside)
        output = sum(int(record["output_tokens"]) for record in inside)
        alpha, window = self.time_scale, self.window_s
        print(
            f"TraceLab window ({self.window_rounds}): start={self.window_start_s:g}s "
            f"length={window:g}s alpha={alpha:g} "
            f"-> num_groups(sessions)={len(plan)} num_prompts(rounds)={len(rounds)} "
            f"(rounds starting inside window: {len(inside)}; last round offset {span_s:.0f}s "
            f"offered, {span_s / alpha:.0f}s scaled)"
        )
        for label, scale in (("offered", 1.0), ("scaled", alpha)):
            print(
                f"TraceLab window {label}: session starts/s={len(plan) / window * scale:.4f} "
                f"calls/s={len(inside) / window * scale:.4f} "
                f"uncached-input tok/s={append / window * scale:.1f} "
                f"output tok/s={output / window * scale:.1f} "
                "(rounds starting inside the window, over its length)"
            )


def _record_span_ms(record):
    event_times = [
        datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        .timestamp() * 1000
        for event in record.get("timing_events") or []
        if event.get("timestamp")
    ]
    if not event_times:
        return None, None
    return min(event_times), max(event_times)


class NaturalTextTokenStream:
    """Sequential token ids from a ShareGPT corpus, tokenized lazily in file order.

    Each conversation's turns are joined with blank lines, tokenized without special tokens,
    and conversations are concatenated with a blank-line separator. take(k) returns the next k
    ids, so a fixed selection always receives the same text. Wrapping past the corpus end
    restarts it and is reported, since repeated text could produce spurious prefix hits.
    """

    batch_size = 256

    def __init__(self, path, tokenizer):
        with open(path) as source:
            corpus = json.load(source)
        self.texts = [
            "\n\n".join(
                turn["value"] for turn in entry.get("conversations") or []
                if isinstance(turn.get("value"), str) and turn["value"].strip()
            )
            for entry in corpus
        ]
        self.texts = [text for text in self.texts if text]
        if not self.texts:
            raise ValueError(f"Text source {path} has no conversation text")
        self.tokenizer = tokenizer
        self.separator = tokenizer.encode("\n\n", add_special_tokens=False)
        self.buffer = []
        self.position = 0
        self.next_text = 0
        self.wraps = 0
        self.consumed = 0

    def _refill(self):
        if self.next_text >= len(self.texts):
            self.next_text = 0
            self.wraps += 1
            print(f"WARNING: text source wrapped ({self.wraps}x) after {self.consumed} tokens")
        batch = self.texts[self.next_text : self.next_text + self.batch_size]
        self.next_text += len(batch)
        encoded = self.tokenizer(batch, add_special_tokens=False)["input_ids"]
        self.buffer = self.buffer[self.position :]
        self.position = 0
        for ids in encoded:
            self.buffer.extend(ids)
            self.buffer.extend(self.separator)

    def take(self, count):
        while len(self.buffer) - self.position < count:
            self._refill()
        ids = self.buffer[self.position : self.position + count]
        self.position += count
        self.consumed += count
        return ids

    def describe(self):
        return (
            f"{self.consumed} tokens from {self.next_text}/{len(self.texts)} conversations, "
            f"{self.wraps} wraps"
        )
