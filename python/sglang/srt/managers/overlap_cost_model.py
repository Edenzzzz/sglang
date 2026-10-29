"""Analytic forward-step cost model for the fused prefill/decode overlap kernel.

Drop-in replacement for the trained MLP latency predictor
(``SGLANG_EDF_LATENCY_MODEL=analytic``). Same interface, no checkpoints, no
per-workload training: every constant is a per-GPU physical parameter fit once
from microbenchmarks (tasks/debug_kernel_server_perf/fit_overlap_cost_model.py).

The model is the reason SLO-aware chunk control works with the fused kernel:
a mixed batch's attention time obeys

    T_attn = max(T_p, T_d) + beta * min(T_p, T_d)

where T_d is the decode pool's (bandwidth-bound) time, T_p the chunk's
(compute-bound) time, and beta the measured non-overlappable fraction. On the
H800 sweep this law holds to 3.1% MAPE (3.2% leave-one-out), so a closed-form
controller can spend an ITL budget precisely: below the balance point (T_p <= T_d), prefill tokens
cost only beta * dT_p — the overlap discount. The same construction fails for
the split-kernel baseline: BatchPrefill's mixed cost is super-additive in its
parts (up to 8.1x their sum; fitting max + beta*min yields beta=2.19), because
one chunk flips the batch-wide CTA tile and consumes the KV-split work budget.

That is a statement about this functional form, not about learnability in
general: a learned model does fit BatchPrefill (MLP 9.9% MAPE over 1,316
measured compositions). What survives is the relative gap -- on identical
data, features and model class, BatchPrefill is 1.7-3x harder to predict than
this kernel (GBDT 10.1% vs 3.3% MAPE, 103% vs 63% worst case), and closing it
needs per-GPU profiling of ~1,300 compositions rather than the 35 kernel
measurements behind the constants below. See
tasks/debug_kernel_server_perf/fit_learned_cost_model.py.

The whole forward step adds a linear non-attention term:

    T_step = c0 + c1 * total_tokens + n_layers * T_attn(layer)

c0 is dominated by reading the weights once per step (params_bytes / HBM BW)
plus fixed launch/host overhead; c1 by per-token GEMM FLOPs at effective
throughput. Sanity: at decode_bs=47 over a 512K-token KV pool this predicts
~15.1 ms per step; the serving benchmark measured 15.75 ms mean ITL.
"""

import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OverlapCostParams:
    # per-layer decode attention: T_d(ms) = a_d * decode_kv_tokens + c_d
    a_d: float
    c_d: float
    # per-layer chunk attention: T_p(ms) = a_p * (n*prefix + n^2/2) + c_p
    a_p: float
    c_p: float
    # non-overlappable fraction of min(T_p, T_d)
    beta: float
    # non-attention step cost: c0 + c1 * total_tokens (ms)
    c0: float
    c1: float
    n_layers: int

    @classmethod
    def load(cls, path: str) -> "OverlapCostParams":
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


# H800 + Qwen2.5-7B (28 layers, GQA 28/4, head_dim 128), bf16.
# Fit 2026-08-06 from tasks/debug_kernel_server_perf/overlap_attribution.json;
# see fit_overlap_cost_model.py for derivation and refresh instructions.
_DEFAULT_H800_QWEN25 = OverlapCostParams(
    a_d=6.547e-7, c_d=0.0392,   # ~1.5 GB/ms effective KV bandwidth per layer
    a_p=4.007e-8, c_p=0.196,    # ~62 TFLOP/s effective bf16 per layer slice
    beta=0.5285,
    c0=5.137, c1=0.02533,       # weights read + overheads; per-token GEMMs
    n_layers=28,
)


class AnalyticOverlapPredictor:
    """Interface-compatible with LatencyPredictor.predict()."""

    def __init__(self, params: OverlapCostParams = None):
        p = params
        if p is None:
            path = os.environ.get("SGLANG_OVERLAP_COST_PARAMS", "")
            if path and os.path.exists(path):
                p = OverlapCostParams.load(path)
                logger.info(f"analytic overlap cost model loaded from {path}")
            else:
                p = _DEFAULT_H800_QWEN25
                logger.info("analytic overlap cost model: built-in H800/Qwen2.5-7B params")
        self.p = p

    def attn_layer_ms(self, n_p: float, r_p: float, decode_kv: float) -> float:
        p = self.p
        t_d = p.a_d * decode_kv + p.c_d if decode_kv > 0 else 0.0
        if n_p > 0:
            work = n_p * r_p + 0.5 * n_p * n_p
            t_p = p.a_p * work + p.c_p
        else:
            t_p = 0.0
        if t_d == 0.0 or t_p == 0.0:
            return t_d + t_p
        return max(t_p, t_d) + p.beta * min(t_p, t_d)

    def predict(
        self,
        n_p: float,
        r_p: float,
        decode_avg_kv: float,
        decode_bs: float,
        acceptance_length: float = 1.0,
    ) -> float:
        """Whole forward-step latency in ms.

        Args match the EDF MLP contract: n_p prefill chunk tokens, r_p the
        chunk's cached prefix length, decode_avg_kv the decode pool's average
        KV length, decode_bs the number of running decode requests.
        """
        p = self.p
        decode_kv = decode_avg_kv * decode_bs
        tokens = n_p + decode_bs
        return p.c0 + p.c1 * tokens + p.n_layers * self.attn_layer_ms(n_p, r_p, decode_kv)
