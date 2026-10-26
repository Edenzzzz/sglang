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
- ":demote": rows already past their SLO (mean TPOT so far above s_r) leave the minimum, the same rule EDF applies to
  prefills that have missed their deadline; without it, one late row stops prefill on the node.
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

call = re.search(r"\n(        # Prefill policy\n)?        adder = PrefillAdder\(\n((?:            .*\n){6})            (\w+),\n", s)
assert call, "PrefillAdder call not found"
var = call.group(3)
assert var in ("effective_chunk_size", "chunked_prefill_size"), var
anchor = call.group(0)
hook = (f"\n        {var} = self._slo_budget_chunk({var})\n        if {var} == 0:\n            return None" + anchor)
assert s.count(anchor) == 1
s = s.replace(anchor, hook, 1)

method = '''
    def _slo_budget_chunk(self, chunk_size):
        """Largest prefill chunk whose predicted step keeps every running decode row within its mean-TPOT SLO."""
        spec = os.environ.get("SGLANG_SLO_CHUNK", "")
        if not spec or not chunk_size or chunk_size <= 0:
            return chunk_size
        rows = self.running_batch.reqs if self.running_batch is not None else []
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
                demote="demote" in fields[5:])
            logger.info("SLO_CHUNK model=%s target=%s S=%g T=%g h=%g chat=%gms agent=%gms (>= %d tokens) demote=%s",
                        fields[0], target, cfg["S"], cfg["T"], cfg["h"], cfg["chat_ms"], cfg["agent_ms"],
                        cfg["agent_tokens"], cfg["demote"])
        now = time.monotonic()
        # Tokens per step per row: EWMA of the rows' observed progress between consecutive calls.
        seen = getattr(self, "_slo_chunk_seen", {})
        progressed = [len(r.output_ids) - seen[r.rid] for r in rows if r.rid in seen]
        if progressed:
            sample = max(1.0, sum(progressed) / len(progressed))
            prev = getattr(self, "_slo_chunk_tps", None)
            self._slo_chunk_tps = sample if prev is None else 0.9 * prev + 0.1 * sample
        self._slo_chunk_seen = {r.rid: len(r.output_ids) for r in rows}
        tps = getattr(self, "_slo_chunk_tps", None) or 1.0
        budget = float("inf")
        for r in rows:
            if getattr(r, "_slo_t0", None) is None:
                r._slo_t0, r._slo_o0 = now, len(r.output_ids)
            s_r = cfg["agent_ms"] if len(r.origin_input_ids) >= cfg["agent_tokens"] else cfg["chat_ms"]
            o = len(r.output_ids) - r._slo_o0
            spent = (now - r._slo_t0) * 1000.0
            if cfg["demote"] and spent > s_r * max(o, 1):
                continue
            budget = min(budget, s_r * (o + tps) - spent)
        self.slo_chunk_calls = getattr(self, "slo_chunk_calls", 0) + 1
        if budget == float("inf"):
            out = chunk_size
        else:
            budget *= 1.0 - cfg["h"]
            if self.chunked_req is not None:
                pf = len(self.chunked_req.prefix_indices)
            elif self.waiting_queue:
                nxt = self.waiting_queue[0]
                pf = len(nxt.prefix_indices) if nxt.prefix_indices is not None else 0
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
        self.slo_chunk_capped = getattr(self, "slo_chunk_capped", 0) + int(out < chunk_size)
        self.slo_chunk_zero = getattr(self, "slo_chunk_zero", 0) + int(out == 0)
        if self.slo_chunk_calls % 500 == 1:
            logger.info("SLO_CHUNK calls=%d capped=%d zero=%d rows=%d tps=%.2f budget_ms=%.1f chunk=%d->%d",
                        self.slo_chunk_calls, self.slo_chunk_capped, self.slo_chunk_zero, len(rows), tps, budget,
                        chunk_size, out)
        return out
'''
anchor2 = "\n    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:"
assert s.count(anchor2) == 1
s = s.replace(anchor2, method + anchor2, 1)
for module in ("import os\n", "import time\n"):
    assert module in s, module
open(p, "w").write(s)
print(f"patched (chunk variable: {var})")
