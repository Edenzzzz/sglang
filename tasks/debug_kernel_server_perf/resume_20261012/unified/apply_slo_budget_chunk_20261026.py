"""SLO-budgeted prefill chunk (user, 2026-10-26): each step's chunk is the largest one whose predicted step time keeps
every running decode row within its mean-TPOT SLO, with headroom h; with no rows decoding, the chunk is the server's
--chunked-prefill-size (launch it large, e.g. 16384, with --max-prefill-tokens at least that).

  SGLANG_SLO_CHUNK="<model.json>:<h>:<chat_ms>:<agent_ms>:<agent_prompt_tokens>[:demote]"

- model.json: lin4mode coefficients from tasks/perf_model/fit_4mode_step.py --export-lin4mode (target "gap", the
  launch-to-launch time a decode row sees, else "gpu"). Step features for a chunk of c tokens on a prefix of pf tokens
  beside nd decode rows holding dkv KV tokens: ptok = c, attn = (c*pf + c^2/2)/1e6, np = 1, mixed = [nd > 0], and the
  draft / verify terms nd*S, nd*T, dkv*T/1e3 with S / T from the model file.
- Row r's budget is its remaining mean-TPOT slack, s_r * (o_r + a) - E_r: o_r tokens emitted and E_r ms spent since
  this rule first saw r decoding, a the tokens one step emits per row (EWMA of the rows' observed progress; 1 without
  speculation). s_r is agent_ms for prompts of at least agent_prompt_tokens, else chat_ms. The step budget is
  (1 - h) * min_r budget_r.
- ":demote": rows already past their SLO (mean TPOT so far above s_r) are not rescued on their mean, but still bound
  the step at their per-step allowance s_r x a (each new token costs at most its target); without it, one late row
  stops prefill on the node. (A first version dropped such rows from the minimum, which at overload, when every row is
  late, removed the bound and let chunks reach the maximum.)
- ":bank=K" (default unlimited): a row's banked slack counts for at most K steps' allowance. The SLO is a request's
  MEAN TPOT, so banking must be unlimited by default: a row that has been decoding at 30 ms against a 50 ms target
  has earned the time for an occasional 446 ms merged step, and that is what the fixed-chunk arms rely on. Capping it
  at 4 steps turned the rule into a per-step (ITL) bound and lost every metric at alpha 3, chat TPOT included
  (45.0 vs 30.0 ms p50).
- ":qcap" / ":stepcap=MS" (2026-10-28): cap what ONE step may spend -- by the remaining TTFT slack of the tightest
  waiting request (qcap), and/or a hard ceiling (stepcap). Unlimited banking alone let ~6 s of banked slack be spent
  in a single 16,384-token step.
- ":life" (2026-10-28): budget each row by its projected mean TPOT at COMPLETION, using the tenant's expected
  output length learned online from finished requests, so a just-admitted row no longer pins every step.
- ":ttft": TTFT floor. The earliest-deadline prefill (in-progress chunk, else the head of the deadline-ordered queue),
  while it can still meet its deadline, gets at least the chunk that finishes its remaining tokens by the deadline at
  the predicted step time. A TPOT-only budget starved prefill far past capacity (alpha 6: chat TTFT p50 74 s).
- a, the tokens one step emits per row, is the EWMA of the scheduler's speculative acceptance (accepted tokens per
  verify forward); the first version measured rows' progress between calls, which spans several steps.
- The chunk solves model(c) <= budget (quadratic in c through attn). Under mixed chunk the scheduler charges one token
  per running row against the chunk, so the rows are added back. Less than one page: no prefill this step.

Inserted just before the PrefillAdder construction in get_new_batch_prefill, on whichever variable carries the chunk
there (effective_chunk_size or chunked_prefill_size), so it composes with the EDF patches. Logs
`SLO_CHUNK calls= capped= zero= rows= budget_ms= chunk=a->b` every 500 calls.
  python apply_slo_budget_chunk_20261026.py <tree>/python/sglang/srt
Claude, 2026-10-26.
"""
import re
import sys

p = f"{sys.argv[1]}/managers/scheduler.py"
s = open(p).read()
if "def _slo_budget_chunk" in s:
    print("already")
    sys.exit(0)

EDF_CHAIN = "        chunked_prefill_size = self._edf_limit_chunk(running_batch, chunked_prefill_size)\n"
if EDF_CHAIN in s:
    # Our serving tree: the EDF chunk rules run in sequence and a zero chunk already means "no prefill this step".
    var = "chunked_prefill_size"
    assert s.count(EDF_CHAIN) == 1
    s = s.replace(EDF_CHAIN, EDF_CHAIN + "        chunked_prefill_size = self._slo_budget_chunk(chunked_prefill_size, running_batch)\n", 1)
else:
    call = re.search(r"\n(        # Prefill policy\n)?        adder = PrefillAdder\(\n((?:            .*\n){6})            (\w+),\n", s)
    assert call, "PrefillAdder call not found"
    var = call.group(3)
    assert var in ("effective_chunk_size", "chunked_prefill_size"), var
    anchor = call.group(0)
    hook = (f"\n        {var} = self._slo_budget_chunk({var})\n        if {var} == 0:\n            return None" + anchor)
    assert s.count(anchor) == 1
    s = s.replace(anchor, hook, 1)

method = '''
    def _slo_budget_chunk(self, chunk_size, running_batch=None):
        """Largest prefill chunk whose predicted step keeps every running decode row within its mean-TPOT SLO."""
        spec = os.environ.get("SGLANG_SLO_CHUNK", "")
        if not spec or not chunk_size or chunk_size <= 0:
            return chunk_size
        running_batch = running_batch if running_batch is not None else self.running_batch
        rows = running_batch.reqs if running_batch is not None else []
        if not rows:
            return chunk_size
        cfg = getattr(self, "_slo_chunk_cfg", None)
        if cfg is None:
            import json as _json

            fields = spec.split(":")
            model = _json.load(open(fields[0]))
            target = "gap" if "gap" in model else "gpu"
            cfg = self._slo_chunk_cfg = dict(
                coef=model[target]["coef"], S=float(model["S"]), T=float(model["T"]), h=float(fields[1]),
                chat_ms=float(fields[2]), agent_ms=float(fields[3]), agent_tokens=int(fields[4]),
                demote="demote" in fields[5:],
                bank_steps=float(next((f[5:] for f in fields[5:] if f.startswith("bank=")), "inf")),
                ttft="ttft" in fields[5:],
                qcap="qcap" in fields[5:],
                life="life" in fields[5:],
                stepcap=float(next((f[8:] for f in fields[5:] if f.startswith("stepcap=")), "inf")))
            logger.info("SLO_CHUNK model=%s target=%s S=%g T=%g h=%g chat=%gms agent=%gms (>= %d tokens) demote=%s",
                        fields[0], target, cfg["S"], cfg["T"], cfg["h"], cfg["chat_ms"], cfg["agent_ms"],
                        cfg["agent_tokens"], cfg["demote"])
        now = time.monotonic()
        # Tokens per step per row: EWMA of the speculative acceptance length (accepted tokens per verify forward,
        # from the scheduler's metrics), 1 without speculation. Progress between two calls of this method spans
        # several decode-only steps, so it overstates the per-step rate (it read 8.8 at acceptance ~2).
        reporter = getattr(self, "metrics_reporter", None)
        tok = getattr(reporter, "spec_total_num_accept_tokens", 0) or 0
        fwd = getattr(reporter, "spec_total_num_forward_ct", 0) or 0
        last = getattr(self, "_slo_chunk_acc_last", None)
        self._slo_chunk_acc_last = (tok, fwd)
        if last is not None and fwd > last[1]:
            sample = max(1.0, (tok - last[0]) / (fwd - last[1]))
            prev = getattr(self, "_slo_chunk_tps", None)
            self._slo_chunk_tps = sample if prev is None else 0.9 * prev + 0.1 * sample
        tps = getattr(self, "_slo_chunk_tps", None) or 1.0
        if cfg.get("life"):
            seen = getattr(self, "_slo_seen", {})
            live = {id(r) for r in rows}
            lives = getattr(self, "_slo_life", {})
            for rid, (cls, n, ms) in list(seen.items()):
                if rid not in live:          # the row left the batch: it finished, so its length is a sample
                    m, k = lives.get(cls, (0.0, 0))
                    lives[cls] = (m + (n - m) / (k + 1), k + 1)
                    if n >= 4:
                        prev = getattr(self, "_slo_rate", None)
                        self._slo_rate = ms / n if prev is None else 0.9 * prev + 0.1 * ms / n
                    del seen[rid]
            for r in rows:
                if getattr(r, "_slo_t0", None) is not None:
                    cls = "agent" if len(r.origin_input_ids) >= cfg["agent_tokens"] else "chat"
                    seen[id(r)] = (cls, len(r.output_ids) - r._slo_o0, (now - r._slo_t0) * 1000.0)
            self._slo_seen, self._slo_life = seen, lives
        budget = float("inf")
        for r in rows:
            if getattr(r, "_slo_t0", None) is None:
                r._slo_t0, r._slo_o0 = now, len(r.output_ids)
            s_r = cfg["agent_ms"] if len(r.origin_input_ids) >= cfg["agent_tokens"] else cfg["chat_ms"]
            o = len(r.output_ids) - r._slo_o0
            spent = (now - r._slo_t0) * 1000.0
            per_step = s_r * tps
            if o < 1:
                # The row has not emitted a token yet, so it HAS no mean TPOT: the latency it has accumulated so
                # far is its TTFT, which the deadline and demotion machinery govern, not this bound. Judging it
                # here is what broke the rule in the measured arms. `spent` for such a row is the time since it
                # was first seen, which is already tens of milliseconds after one ordinary step, so the test below
                # fired on it, pinned the budget to the bare per-step allowance, and -- because the budget is a
                # MIN over rows -- let one just-admitted row clamp every step in the batch. Measured 2026-10-28:
                # budget_ms 90-146 and capped=7001 of calls=7001, i.e. the lifetime budget never once applied.
                self.slo_chunk_fresh = getattr(self, "slo_chunk_fresh", 0) + 1
                continue
            # L and rate are needed BEFORE the demotion test, because under the lifetime reading the test is
            # "can this row still reach its target by completion", not "is its mean above target right now". A row
            # three tokens in whose first step waited behind a long prefill chunk fails the instantaneous test on
            # every one of its early steps, and since the budget is a MIN over rows, that one row then clamps the
            # step to the bare per-step allowance -- the same failure as the fresh-row case above, one token later.
            L = rate = None
            if cfg["life"]:
                cls_l = "agent" if s_r == cfg["agent_ms"] else "chat"
                life_l = getattr(self, "_slo_life", {}).get(cls_l)
                if life_l is not None and life_l[1] >= 8:
                    L = max(life_l[0], o + tps)
                    rate = spent / o if o >= 4 else getattr(self, "_slo_rate", s_r)
            if cfg["demote"] and spent > s_r * (L if L is not None else o):
                # Past rescue: even running the rest of its output at its target rate it cannot reach its mean, so
                # it lends no slack. Each new token still costs at most its target, so the step is bounded by its
                # per-step allowance (no backlog recovery, no banking).
                budget = min(budget, per_step)
                self.slo_chunk_demoted = getattr(self, "slo_chunk_demoted", 0) + 1
                continue
            if cfg["life"]:
                # Lifetime mean TPOT. The SLO is a request's mean TPOT over its WHOLE output, so a step's cost is
                # spread over the tokens it has not produced yet. Projected mean at completion:
                #     (spent + step + remaining * rate) / L  <=  s_r   =>   step <= s_r * L - spent - remaining * rate
                # with L the tenant's expected output length (running mean of requests that actually finished,
                # never below what this row has already produced) and rate this row's own measured ms per token.
                # Without this, the bound below makes a just-admitted row -- which has banked nothing -- the minimum
                # on nearly every step, and the mean-TPOT rule degrades into a per-step bound (measured 2026-10-28:
                # chunks of 147-333 tokens, agent TTFT p90 9.5 s against prefill-prioritized's 3.4 s).
                if L is None:
                    # Fewer than eight finished requests for this tenant, so there is no output length to project
                    # from yet. Counted, because a silent fall-through to the per-step bound below is
                    # indistinguishable in the metrics from the lifetime rule simply not helping.
                    self.slo_chunk_nolife = getattr(self, "slo_chunk_nolife", 0) + 1
                else:
                    self.slo_chunk_life = getattr(self, "slo_chunk_life", 0) + 1
                    remaining = max(0.0, L - o - tps)
                    budget = min(budget, s_r * L - spent - remaining * rate)
                    continue
            # Banked slack is capped at BANK_STEPS steps' worth, so one step cannot stall a row for seconds.
            budget = min(budget, s_r * (o + tps) - spent, cfg["bank_steps"] * per_step)
        # Per-step cap, and the bound that actually binds. Unlimited banking is the right reading of each resident
        # row's MEAN TPOT, but it lets slack a row banked over many fast steps be spent as ONE long stall (measured
        # 2026-10-28: budgets up to 5,867 ms, full 16,384-token chunks on ~7% of steps), which every resident row
        # and every request still waiting must sit through.
        #
        # The offline sweep over 10,185 recorded merged steps (tasks/perf_model/chunk_predictor.py --rule) shows
        # the mean-TPOT budget has almost no usable band: at the paper's 50 ms target against a 13.9 ms/token
        # observed rate the lifetime budget is 16.3 s and saturates the chunk maximum on 100% of steps, while the
        # per-step form grants 90 ms and admits less than the run used on 99% of them. Only a target within about
        # 8-15% of the achieved rate (14.5-16 ms here) puts the lifetime budget in the 1-4 K token range. So the
        # bound that binds at normal load has to be the INTER-TOKEN TAIL target, which the workload also states:
        # ":stepcap=MS" is that target in milliseconds, and the chunk solved under it still adapts with prefix
        # length and decode batch size rather than being a fixed token count -- with no rows resident the bound is
        # vacuous and the chunk is the server maximum. The sweep puts a 600 ms tail target at the same median
        # chunk as sglang's 2048 (p50 2176 tokens) while admitting more on 65% of steps and less on 35%.
        # ":qcap" bounds one step by the remaining TTFT slack of the tightest WAITING request, so no step is longer
        # than the most urgent queued request can afford. Both come from SLO deadlines, not a tuned chunk constant.
        if cfg["qcap"] or cfg["stepcap"] != float("inf"):
            cap = cfg["stepcap"]
            if cfg["qcap"]:
                head = self.chunked_req
                for q in self.waiting_queue[:64]:
                    if q is head:
                        continue
                    d = getattr(q, "priority", None)
                    if d is not None and d > now:
                        cap = min(cap, (d - now) * 1000.0)
            if cap < budget:
                budget = cap
                self.slo_chunk_qcapped = getattr(self, "slo_chunk_qcapped", 0) + 1
        self.slo_chunk_calls = getattr(self, "slo_chunk_calls", 0) + 1
        if budget == float("inf"):
            out = chunk_size
        else:
            budget *= 1.0 - cfg["h"]
            def prefix_of(req):
                # Our serving tree tracks the computed prefix on req.kv / prefix_len and the enqueue match in
                # _edf_cached_len; the upstream tree keeps prefix_indices.
                kv = getattr(req, "kv", None)
                committed = getattr(kv, "kv_committed_len", 0) if kv is not None else 0
                indices = getattr(req, "prefix_indices", None)
                return max(getattr(req, "prefix_len", 0) or 0, committed or 0, getattr(req, "_edf_cached_len", 0) or 0,
                           len(indices) if indices is not None else 0)

            if self.chunked_req is not None:
                pf = prefix_of(self.chunked_req)
            elif self.waiting_queue:
                pf = prefix_of(self.waiting_queue[0])
            else:
                pf = 0
            c0, c_tok, c_attn, c_bs, c_kv, c_np, c_mixed, c_draft, c_ver, c_vkv = cfg["coef"]
            nd = len(rows)
            dkv = sum(len(r.origin_input_ids) + len(r.output_ids) for r in rows)
            fixed = (c0 + c_bs * nd + c_kv * dkv / 1e3 + c_np + c_mixed + c_draft * nd * cfg["S"]
                     + c_ver * nd * cfg["T"] + c_vkv * dkv * cfg["T"] / 1e3)
            room = budget - fixed
            # c_attn/2e6 * c^2 + (c_tok + c_attn*pf/1e6) * c - room <= 0
            qa, qb = c_attn / 2e6, c_tok + c_attn * pf / 1e6
            if room <= 0:
                c = 0.0
            elif qa <= 0:
                c = room / qb if qb > 0 else float(chunk_size)
            else:
                c = (-qb + (qb * qb + 4 * qa * room) ** 0.5) / (2 * qa)
            mixed_rows = nd if self.is_mixed_chunk else 0
            out = int(min(chunk_size - mixed_rows, c))
            if out < max(1, self.page_size):
                out = 0
            elif self.is_mixed_chunk:
                out += mixed_rows
        if cfg["ttft"] and out < chunk_size:
            # TTFT floor: the earliest-deadline prefill that can still meet its deadline gets at least the chunk
            # that finishes its remaining tokens by then at the predicted step time; a late one gets no floor.
            head = self.chunked_req if self.chunked_req is not None else (self.waiting_queue[0] if self.waiting_queue else None)
            deadline = getattr(head, "priority", None) if head is not None else None
            if deadline is not None and deadline > now:
                ids = getattr(head, "full_untruncated_fill_ids", None) or head.origin_input_ids
                done = max(getattr(head, "prefix_len", 0) or 0, getattr(head, "_edf_cached_len", 0) or 0)
                remaining = max(0, len(ids) - done)
                window_ms = (deadline - now) * 1000.0
                c0_, c_tok_, c_attn_, c_bs_, c_kv_, c_np_, c_mixed_, c_draft_, c_ver_, c_vkv_ = cfg["coef"]
                nd_ = len(rows)
                dkv_ = sum(len(r.origin_input_ids) + len(r.output_ids) for r in rows)
                base = (c0_ + c_bs_ * nd_ + c_kv_ * dkv_ / 1e3 + c_np_ + c_mixed_ + c_draft_ * nd_ * cfg["S"]
                        + c_ver_ * nd_ * cfg["T"] + c_vkv_ * dkv_ * cfg["T"] / 1e3)
                floor, meets = max(out, self.page_size, 256), False
                while remaining > 0:
                    step = base + c_tok_ * floor + c_attn_ * (floor * done + floor * floor / 2) / 1e6
                    if remaining / floor * step <= window_ms:
                        meets = True
                        break
                    if floor >= chunk_size:
                        break
                    floor = min(chunk_size, int(floor * 1.5))
                # A prefill that cannot meet its deadline at any chunk size is treated as late: no floor.
                if meets and floor > out:
                    self.slo_chunk_floored = getattr(self, "slo_chunk_floored", 0) + 1
                    out = int(floor)
        self.slo_chunk_capped = getattr(self, "slo_chunk_capped", 0) + int(out < chunk_size)
        self.slo_chunk_zero = getattr(self, "slo_chunk_zero", 0) + int(out == 0)
        if self.slo_chunk_calls % 500 == 1:
            logger.info("SLO_CHUNK calls=%d capped=%d zero=%d floored=%d qcapped=%d fresh=%d life=%d nolife=%d "
                        "rows=%d tps=%.2f budget_ms=%.1f chunk=%d->%d",
                        self.slo_chunk_calls, self.slo_chunk_capped, self.slo_chunk_zero,
                        getattr(self, "slo_chunk_floored", 0), getattr(self, "slo_chunk_qcapped", 0),
                        getattr(self, "slo_chunk_fresh", 0), getattr(self, "slo_chunk_life", 0),
                        getattr(self, "slo_chunk_nolife", 0), len(rows), tps, budget, chunk_size, out)
            logger.info("SLO_CHUNK demoted=%d rate_ms=%.1f life_len=%s",
                        getattr(self, "slo_chunk_demoted", 0), getattr(self, "_slo_rate", float("nan")),
                        {k: (round(v[0], 1), v[1]) for k, v in getattr(self, "_slo_life", {}).items()})
        return out
'''
anchor2 = re.search(r"\n    def get_new_batch_prefill\(self[^)]*\)[^:]*:", s).group(0)
assert s.count(anchor2) == 1
s = s.replace(anchor2, method + anchor2, 1)
for module in ("import os\n", "import time\n"):
    assert module in s, module
open(p, "w").write(s)
print(f"patched (chunk variable: {var})")
