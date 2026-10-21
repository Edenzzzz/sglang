"""SGLANG_SPLIT_MARGIN_AUTO=1: the co-run's SM partition is chosen per step from the step's composition.

With SPLIT_OVERLAP=1 the prefill chunk runs on FA3 with sm_margin m and the decode / verify rows on the m SMs left
(apply_split_rows_fa3.py). Every serving run before this left SGLANG_FUSED_SM_MARGIN at 0, so the two launches
competed for all SMs. rows_layout_check_20261021.py swept m over 8..112 against one stock FA3 call: the best m moves
from 16 to 80 SMs with the rows' share of the step, and the co-run gains 4-25% at the best m.

Cost model, per step: chunk time = causal FLOPs / FA3 peak (4 * Hq * D * q * (prefix + (q + 1) / 2) per request);
rows time = their K/V bytes / HBM bandwidth + their FLOPs / FA3 peak. m = round8(num_SMs * rows / (chunk + rows) *
scale), clamped to [8, num_SMs - 16]. Defaults fit the H200 sweep (600 TFLOP/s, 4.0 TB/s, scale 0.75: memory-bound
rows saturate bandwidth on fewer SMs than their time share; within one 8-SM step of the best m in 7 of 9 cases, two
steps in the other two); H20 defaults 120 TFLOP/s, 3.6 TB/s. Override with SGLANG_SPLIT_FA3_TFLOPS,
SGLANG_SPLIT_KV_TBPS, SGLANG_SPLIT_ROWS_SM_SCALE. The chosen m is logged with the split counters (SPLIT_MARGIN lines).
  python apply_split_auto_margin.py <tree>/python/sglang/srt
Claude, 2026-10-21.
"""
import sys

p = f"{sys.argv[1]}/layers/attention/flashinfer_backend.py"
s = open(p).read()
assert "SPLIT_MARGIN_AUTO" not in s, "already applied"
assert "_fs_rows_fa3_plan" in s, "apply_split_rows_fa3.py first"


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:120])
    s = s.replace(old, new)


sub(
    'SPLIT_ROWS_FA3 = os.environ.get("SGLANG_SPLIT_ROWS_FA3", "0") == "1"\n',
    'SPLIT_ROWS_FA3 = os.environ.get("SGLANG_SPLIT_ROWS_FA3", "0") == "1"\n'
    "# Per-step SM partition of the co-run from the step's composition (see _fs_auto_margin).\n"
    'SPLIT_MARGIN_AUTO = os.environ.get("SGLANG_SPLIT_MARGIN_AUTO", "0") == "1"\n',
)

sub(
    """            n_fa3=len(fa3_rows),
            n_fa2=n2,
        )
        if FUSED_SPLIT_DEBUG:
            self._fused_split_debug_check("plan")
        return True

    def _fused_split_plan(self, forward_batch: ForwardBatch) -> bool:""",
    """            n_fa3=len(fa3_rows),
            n_fa2=n2,
        )
        if SPLIT_MARGIN_AUTO:
            self.fused_split_plan["sm_margin"] = self._fs_auto_margin(
                qo_lens, seq_lens_cpu, fa3_rows, fa2_rows, dev
            )
        if FUSED_SPLIT_DEBUG:
            self._fused_split_debug_check("plan")
        return True

    def _fs_auto_margin(self, qo_lens, seq_lens_cpu, fa3_rows, fa2_rows, dev):
        \"\"\"SMs to leave the rows: their predicted share of the step's attention time, scaled.\"\"\"
        if getattr(self, "_fs_cost", None) is None:
            props = torch.cuda.get_device_properties(dev)
            h20 = "H20" in props.name
            self._fs_cost = dict(
                nsm=props.multi_processor_count,
                tflops=float(os.environ.get("SGLANG_SPLIT_FA3_TFLOPS", "120" if h20 else "600")) * 1e12,
                tbps=float(os.environ.get("SGLANG_SPLIT_KV_TBPS", "3.6" if h20 else "4.0")) * 1e12,
                scale=float(os.environ.get("SGLANG_SPLIT_ROWS_SM_SCALE", "0.75")),
            )
        c = self._fs_cost
        iu = self.indices_updater_prefill
        hq, hk, d = iu.num_qo_heads, iu.num_kv_heads, iu.head_dim
        kv = seq_lens_cpu.numpy()
        q3, k3 = qo_lens[fa3_rows], kv[fa3_rows]
        chunk = float((4 * hq * d * q3 * ((k3 - q3) + (q3 + 1) / 2)).sum()) / c["tflops"]
        q2, k2 = qo_lens[fa2_rows], kv[fa2_rows]
        rows = float(k2.sum()) * hk * d * 2 * 2 / c["tbps"] + float((4 * hq * d * q2 * k2).sum()) / c["tflops"]
        m = c["nsm"] * rows / (chunk + rows) * c["scale"]
        m = int(min(max(round(m / 8) * 8, 8), c["nsm"] - 16))
        self.split_margin_sum = getattr(self, "split_margin_sum", 0) + m
        self.split_margin_n = getattr(self, "split_margin_n", 0) + 1
        if self.split_margin_n % 500 == 1:
            logger.info(
                "SPLIT_MARGIN m=%d (chunk %.3f ms, rows %.3f ms predicted) mean=%.1f over %d steps",
                m, chunk * 1e3, rows * 1e3, self.split_margin_sum / self.split_margin_n, self.split_margin_n,
            )
        return m

    def _fused_split_plan(self, forward_batch: ForwardBatch) -> bool:""",
)

# Dispatch: the chunk's and the rows' margins come from the plan when it carries one.
sub(
    """            sm_margin=self.fused_sm_margin,
        )
        if _overlap:""",
    """            sm_margin=pl.get("sm_margin", self.fused_sm_margin) if _overlap else self.fused_sm_margin,
        )
        if _overlap:""",
)
sub(
    """                sm_margin=(
                    self._fs_num_sms - self.fused_sm_margin
                    if _overlap and self.fused_sm_margin > 0
                    else 0
                ),""",
    """                sm_margin=(
                    self._fs_num_sms - pl.get("sm_margin", self.fused_sm_margin)
                    if _overlap and pl.get("sm_margin", self.fused_sm_margin) > 0
                    else 0
                ),""",
)
open(p, "w").write(s)
print("applied", p)
