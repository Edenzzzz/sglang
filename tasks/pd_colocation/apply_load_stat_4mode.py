"""Node load for the SLO-checked router: append the 4-mode step model's inputs to sglang's router-facing LoadStat.

Upstream already publishes a per-scheduler `LoadStat` on a ZMQ PUB socket (`--load-publish-endpoint`; sent when it
changes, re-sent every 1 s otherwise, throttled to every N scheduler iterations). Its fields (running / waiting request
counts, KV tokens) cannot drive a step-time prediction: decode step time depends on the decode rows and their KV, and
a prefill's wait depends on the earlier-deadline prefill work queued ahead of it and its depth. This appends, after
the existing positional fields (routers that decode the old prefix are unaffected):
  num_decode_rows     running decode / verify rows
  decode_kv_tokens    their total KV (prompt + output tokens)
  accept_len          speculative acceptance length (tokens per verify step), 1.0 without speculation
  prefill_queue       up to 64 [ms_to_deadline, uncached_tokens, prefix_tokens] entries, earliest deadline first:
                      the in-progress chunked request (its remaining tokens at its current depth) and the waiting
                      queue. Deadlines are EDF priorities (monotonic seconds) reported relative to now, since worker
                      and router clocks differ; requests without a deadline report 1e9 ms.
  python apply_load_stat_4mode.py <tree>/python/sglang/srt
Claude, 2026-10-20.
"""
import sys

root = sys.argv[1]


def patch(rel, edits):
    p = f"{root}/{rel}"
    s = open(p).read()
    if "prefill_queue" in s:
        print("already", rel)
        return
    for old, new in edits:
        assert s.count(old) == 1, (rel, old[:60], s.count(old))
        s = s.replace(old, new, 1)
    open(p, "w").write(s)
    print("patched", rel)


patch("managers/load_snapshot.py", [(
    "    decode_moments: Optional[list[float]] = None\n",
    "    decode_moments: Optional[list[float]] = None\n"
    "    # 4-mode step-model inputs for the SLO-checked router (apply_load_stat_4mode.py).\n"
    "    num_decode_rows: int = 0\n"
    "    decode_kv_tokens: int = 0\n"
    "    accept_len: float = 1.0\n"
    "    prefill_queue: Optional[list[list[float]]] = None\n",
)])

patch("managers/scheduler_components/load_inquirer.py", [
    ("        return LoadSnapshot(\n",
     '''        running_reqs = self.get_running_batch().reqs
        decode_kv_tokens = sum(len(r.origin_input_ids) + len(r.output_ids) for r in running_reqs)
        accept_len = 1.0
        if not self.spec_algorithm.is_none() and self.get_spec_total_num_forward_ct() > 0:
            accept_len = self.get_spec_total_num_accept_tokens() / self.get_spec_total_num_forward_ct()
        now = time.monotonic()
        entries = []
        cr = self.get_chunked_req()
        if cr is not None:
            entries.append([cr.priority, max(0, cr.seqlen - cr.prefix_len), cr.prefix_len])
        for r in self.get_waiting_queue():
            cached = getattr(r, "_edf_cached_len", 0)
            entries.append([r.priority, max(0, len(r.origin_input_ids) - cached), cached])
        entries.sort(key=lambda e: float("inf") if e[0] is None else e[0])
        prefill_queue = [
            [1e9 if pr is None else round((pr - now) * 1000.0, 1), int(u), int(pf)]
            for pr, u, pf in entries[:64]
        ]

        return LoadSnapshot(
'''),
    ("            decode_moments=decode_moments,\n",
     "            decode_moments=decode_moments,\n"
     "            num_decode_rows=len(running_reqs),\n"
     "            decode_kv_tokens=decode_kv_tokens,\n"
     "            accept_len=round(accept_len, 4),\n"
     "            prefill_queue=prefill_queue,\n"),
])

patch("managers/scheduler_components/load_publisher.py", [
    ("    total_prefill_busy_us: int = 0  # Cumulative prefill step time in microseconds\n",
     "    total_prefill_busy_us: int = 0  # Cumulative prefill step time in microseconds\n"
     "    num_decode_rows: int = 0  # Running decode / verify rows (apply_load_stat_4mode.py)\n"
     "    decode_kv_tokens: int = 0  # Their total KV tokens\n"
     "    accept_len: float = 1.0  # Speculative acceptance length\n"
     "    prefill_queue: Optional[list] = None  # [ms_to_deadline, uncached, prefix], earliest first\n"),
    ("                load.total_prefill_busy_us,\n            )\n",
     "                load.total_prefill_busy_us,\n"
     "                load.num_decode_rows,\n"
     "                load.decode_kv_tokens,\n"
     "                load.accept_len,\n"
     "                load.prefill_queue,\n"
     "            )\n"),
    ("                    total_prefill_busy_us=counts[8],\n",
     "                    total_prefill_busy_us=counts[8],\n"
     "                    num_decode_rows=counts[9],\n"
     "                    decode_kv_tokens=counts[10],\n"
     "                    accept_len=counts[11],\n"
     "                    prefill_queue=counts[12],\n"),
])
