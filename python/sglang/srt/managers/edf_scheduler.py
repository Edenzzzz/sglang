"""EDF (Earliest Deadline First) scheduling support for SLO-aware serving.

Provides:
1. Deadline computation from SLO targets
2. SLO-aware chunk size control using latency prediction model
3. Integration hooks for the existing scheduler

Enabled by SGLANG_EDF_SCHEDULING=1 environment variable.
Configure SLOs via SGLANG_TTFT_SLO_MS (default 2000) and SGLANG_ITL_SLO_MS (default 100).
Set SGLANG_EDF_LATENCY_MODEL to the path of the latency model .pt file.

Usage:
    The scheduler calls edf_set_priority() to set request priorities based on deadlines.
    The scheduler calls edf_compute_max_chunk_size() to limit prefill tokens per SLO.
"""

import logging
import math
import os
import time
import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

# Configuration from environment
EDF_ENABLED = os.environ.get("SGLANG_EDF_SCHEDULING", "0") == "1"
TTFT_SLO_MS = float(os.environ.get("SGLANG_TTFT_SLO_MS", "2000"))
ITL_SLO_MS = float(os.environ.get("SGLANG_ITL_SLO_MS", "100"))
TTFT_SCALE_TOKENS = float(os.environ.get("SGLANG_TTFT_SCALE_TOKENS", "5000"))
TTFT_SCALE_MS = float(os.environ.get("SGLANG_TTFT_SCALE_MS", "1000"))
LATENCY_MODEL_PATH = os.environ.get("SGLANG_EDF_LATENCY_MODEL", "")
# Multiplier applied to MLP-predicted step latency before comparing to slack.
# >1.0 makes chunk control more conservative — useful when the latency model
# under-predicts at large decode_bs (observed for DMC at nebius rate=inf:
# 0.68% decode steps > 150ms vs LPT_CG's 0.34%, see task #110 analysis).
LATENCY_SAFETY_MARGIN = float(os.environ.get("SGLANG_EDF_LATENCY_SAFETY", "1.0"))
# Overhead (ms) added to model predictions for persistent-kernel planning and
# launch. A numeric value preserves the fixed controller. "auto" interpolates
# the validated 5/7.5/10ms Pareto points from live output budgets and protects
# extreme continued prefixes with the conservative 10ms guard.
PLAN_OVERHEAD_SETTING = os.environ.get("SGLANG_EDF_PLAN_OVERHEAD_MS", "0")
ADAPTIVE_PLAN_OVERHEAD = PLAN_OVERHEAD_SETTING.lower() == "auto"
PLAN_OVERHEAD_MS = (
    0.0 if ADAPTIVE_PLAN_OVERHEAD else float(PLAN_OVERHEAD_SETTING)
)
# overlap_profiler.py trains the checked-in models with r_p in [0, 4096].
# Clamp continued-prefill prefixes to that domain instead of allowing unstable
# MLP extrapolation for later chunks of 8K+ prompts.
MAX_PROFILED_PREFILL_PREFIX = float(
    os.environ.get("SGLANG_EDF_MAX_PROFILED_PREFILL_PREFIX", "4096")
)
# Clamping remains the best throughput/ITL point through the 12K workload, but
# becomes severely optimistic on the last chunks of 16K prompts.  A 14K
# boundary is the measured WL4 Pareto knee: earlier projection loses throughput
# without improving its request tail, while never projecting causes 500ms+
# decode gaps.
LONG_PREFIX_PROJECTION_THRESHOLD = float(
    os.environ.get("SGLANG_EDF_LONG_PREFIX_PROJECTION_THRESHOLD", "14336")
)
# Stop admitting new prefill work only after live KV reaches this watermark.
# Unlike static headroom, this does not shrink the usable pool during normal
# operation. 1.0 preserves the upstream admission behavior.
KV_ADMISSION_WATERMARK = float(
    os.environ.get("SGLANG_EDF_KV_ADMISSION_WATERMARK", "1.0")
)


# Tokens a decode row emits per step. bench_serving and users measure ITL per TOKEN (a chunk's
# gap divided by its token count), so with speculative decoding a step that emits `a` tokens per
# row may take up to a * ITL_SLO before any row misses. The scheduler cannot know the next step's
# acceptance, but the draft model is fixed online, so the recent past estimates it:
#   none   a = 1 (the step-time budget of a non-speculative server)
#   global one EWMA over every row's observed tokens/step
#   req    a per-request EWMA, falling back to the global one until the request has history
ACCEPT_MODE = os.environ.get("SGLANG_EDF_ACCEPT_MODE", "none")
ACCEPT_ALPHA = float(os.environ.get("SGLANG_EDF_ACCEPT_ALPHA", "0.2"))
_acc_global: Optional[float] = None
# Online scoring of each estimator against the tokens the step then actually emitted:
# sums of absolute error for a=1, the global EWMA and the per-request EWMA.
_acc_err = {"n": 0, "obs": 0.0, "one": 0.0, "global": 0.0, "req": 0.0}
_bind = {"n": 0, "a": 0.0, "g": 0.0}
# ITL attainment is usually a per-request quantile (P90 of a request's per-token gaps), so a
# request can absorb a (1 - q) share of slow steps. With SGLANG_EDF_ITL_QUANTILE=q, a request
# constrains the chunk budget only once its own over-SLO steps have used that allowance; until
# then prefill is admitted in full chunks, which also avoids splitting it into more fixed-cost
# mixed steps. Unset: every request constrains every step (the strict per-step controller).
ITL_QUANTILE = float(os.environ.get("SGLANG_EDF_ITL_QUANTILE", "0") or 0)
_ACC_LOG_EVERY = int(os.environ.get("SGLANG_EDF_ACCEPT_LOG_EVERY", "20000"))


def edf_observe_tokens(req: "Req") -> None:
    """Fold the tokens `req` emitted in the step that just finished into the estimates.

    Called once per decode step per request, after the step's tokens were appended.
    """
    global _acc_global
    n = len(req.output_ids)
    prev = getattr(req, "_edf_prev_out", None)
    req._edf_prev_out = n
    now = time.monotonic()
    prev_t = getattr(req, "_edf_prev_obs_t", None)
    req._edf_prev_obs_t = now
    if prev is None or n <= prev:
        return
    a = float(n - prev)
    if prev_t is not None:
        req._edf_steps = getattr(req, "_edf_steps", 0) + 1
        if (now - prev_t) * 1000.0 / a > ITL_SLO_MS:
            req._edf_slow = getattr(req, "_edf_slow", 0) + 1
    g = _acc_global
    r = getattr(req, "_edf_acc_hat", None)
    if g is not None:
        e = _acc_err
        e["n"] += 1
        e["obs"] += a
        e["one"] += abs(a - 1.0)
        e["global"] += abs(a - g)
        e["req"] += abs(a - (r if r is not None else g))
        if e["n"] % _ACC_LOG_EVERY == 0:
            logger.info(
                "EDF_ACCEPT n=%d mean_tokens=%.3f mae_one=%.3f mae_global=%.3f mae_req=%.3f",
                e["n"], e["obs"] / e["n"], e["one"] / e["n"], e["global"] / e["n"],
                e["req"] / e["n"],
            )
    _acc_global = a if g is None else (1 - ACCEPT_ALPHA) * g + ACCEPT_ALPHA * a
    req._edf_acc_hat = a if r is None else (1 - ACCEPT_ALPHA) * r + ACCEPT_ALPHA * a


def edf_tokens_per_step(req: "Req") -> float:
    if ACCEPT_MODE == "none":
        return 1.0
    g = _acc_global if _acc_global is not None else 1.0
    if ACCEPT_MODE == "global":
        return g
    r = getattr(req, "_edf_acc_hat", None)
    return r if r is not None else g


def project_extreme_prefix_work(
    n_p: int, prefill_prefix_len: float
) -> tuple[float, float]:
    """Map extreme-prefix causal attention work into the model's r_p domain."""
    prefix = max(float(prefill_prefix_len), 0.0)
    if n_p <= 0 or prefix <= LONG_PREFIX_PROJECTION_THRESHOLD:
        return float(n_p), min(prefix, MAX_PROFILED_PREFILL_PREFIX)

    # Preserve n*r + n^2/2 at the largest trained prefix.  This is
    # conservative because it also increases the MLP-token feature, but unlike
    # unconstrained MLP extrapolation it remains inside the r_p domain.
    work = float(n_p) * prefix + float(n_p) * float(n_p) / 2.0
    boundary = MAX_PROFILED_PREFILL_PREFIX
    equivalent_n_p = math.sqrt(boundary * boundary + 2.0 * work) - boundary
    return equivalent_n_p, boundary


def compute_ttft_slo_ms(input_len: int) -> float:
    """Compute TTFT SLO for a request based on its input length.

    Base TTFT + scaling for longer inputs (Ascendra-style).
    """
    extra_tokens = max(0, input_len - 4096)
    extra_ms = (extra_tokens / TTFT_SCALE_TOKENS) * TTFT_SCALE_MS
    return TTFT_SLO_MS + extra_ms


def compute_ttft_deadline(req: "Req") -> float:
    """Compute TTFT deadline timestamp for a request.

    Uses time.monotonic() to match last_tic clock in schedule_batch.py.
    Note: wait_queue_entry_time uses perf_counter, so we store our own
    monotonic arrival time on the request.
    """
    input_len = len(req.origin_input_ids)
    slo_ms = compute_ttft_slo_ms(input_len)
    # Use the monotonic arrival time we set in edf_set_priority
    arrival_time = getattr(req, "_edf_arrival_monotonic", req.last_tic)
    return arrival_time + slo_ms / 1000.0


def compute_itl_deadline(req: "Req") -> float:
    """Compute next ITL deadline for a decode request.

    Uses last_tic which is time.monotonic() in schedule_batch.py.
    """
    return req.last_tic + ITL_SLO_MS / 1000.0


def edf_set_priority(req: "Req") -> None:
    """Set request priority to TTFT deadline (lower = earlier = higher priority).

    Call this when a request enters the waiting queue.
    Works with schedule_low_priority_values_first=True.
    """
    if not EDF_ENABLED:
        return
    # Store monotonic arrival time for consistent clock usage
    req._edf_arrival_monotonic = time.monotonic()
    req.priority = compute_ttft_deadline(req)


def edf_set_retracted_priority(req: "Req") -> None:
    """Keep an interrupted decode urgent when it returns to the wait queue.

    A retracted request has already paid TTFT and may have a large generated
    prefix to recompute. Resetting its arrival time assigns a new, late TTFT
    deadline and can strand the stream behind the entire prefill backlog.
    Preserve its original absolute deadline and only tighten it with the
    already-missed/current ITL deadline.
    """
    if not EDF_ENABLED:
        return

    itl_deadline = compute_itl_deadline(req)
    if req.priority is None:
        req.priority = itl_deadline
    else:
        req.priority = min(req.priority, itl_deadline)


def edf_allow_prefill_at_kv_usage(token_usage: float) -> bool:
    """Return whether EDF may admit more prefill at the current live-KV usage."""
    return token_usage < KV_ADMISSION_WATERMARK


def edf_plan_overhead_ms(
    decode_reqs: List["Req"], prefill_prefix_len: float
) -> float:
    """Return the fixed or live-workload planner guard.

    The adaptive curve is based on the accepted N=400 Pareto points: 5ms for
    1K-output requests, 7.5ms for 2K, and 10ms from 3K onward. Unknown request
    metadata and extreme continued prefixes use the conservative endpoint.
    """
    if not ADAPTIVE_PLAN_OVERHEAD:
        return PLAN_OVERHEAD_MS
    if prefill_prefix_len >= LONG_PREFIX_PROJECTION_THRESHOLD:
        return 10.0

    output_budgets = [
        getattr(getattr(req, "sampling_params", None), "max_new_tokens", None)
        for req in decode_reqs
    ]
    output_budgets = [value for value in output_budgets if value is not None]
    if len(output_budgets) != len(decode_reqs) or not output_budgets:
        return 10.0

    max_output_tokens = max(float(value) for value in output_budgets)
    return min(10.0, max(5.0, 2.5 + 2.5 * max_output_tokens / 1024.0))


def predict_forward_snapshot_ms(
    model_predict_fn,
    num_prefill_tokens: int,
    prefill_prefix_tokens: int,
    prefill_request_count: int,
    decode_kv_tokens: int,
    decode_batch_size: int,
    chunked_prefill_size: Optional[int],
) -> Optional[float]:
    """Predict the next full model-forward cost for a server load snapshot.

    This deliberately reports one forward, not an unvalidated queue-drain
    estimate. Pending prefill is capped at the server's admission chunk and
    prefix/decode KV inputs use per-request averages, matching the profiler's
    feature definitions.
    """
    if model_predict_fn is None:
        return None

    n_p = max(int(num_prefill_tokens), 0)
    if chunked_prefill_size is not None:
        n_p = min(n_p, max(int(chunked_prefill_size), 0))

    prefill_count = max(int(prefill_request_count), 0)
    prefill_prefix_len = (
        max(float(prefill_prefix_tokens), 0.0) / prefill_count
        if prefill_count
        else 0.0
    )
    model_n_p, model_r_p = project_extreme_prefix_work(n_p, prefill_prefix_len)

    decode_bs = max(int(decode_batch_size), 0)
    decode_avg_kv = (
        max(float(decode_kv_tokens), 0.0) / decode_bs if decode_bs else 0.0
    )
    if n_p == 0 and decode_bs == 0:
        return 0.0
    return float(model_predict_fn(model_n_p, model_r_p, decode_avg_kv, decode_bs))


def edf_compute_max_chunk_size(
    decode_reqs: List["Req"],
    default_chunk_size: int,
    model_predict_fn=None,
    decode_avg_kv: float = 0,
    prefill_prefix_len: float = 0,
) -> int:
    """Compute maximum prefill chunk size that won't violate decode ITL SLOs.

    Args:
        decode_reqs: Currently running decode requests
        default_chunk_size: Maximum chunk size (chunked_prefill_size)
        model_predict_fn: fn(n_p, r_p, r_d, decode_bs) -> predicted_ms
        decode_avg_kv: Average KV length of decode requests
        prefill_prefix_len: Cached KV length preceding the pending prefill
            chunk. This is nonzero for the second and later chunks of a long
            prompt and materially affects attention latency.

    Returns:
        Adjusted chunk size (may be smaller than default if SLOs are tight)
    """
    if not EDF_ENABLED or model_predict_fn is None or len(decode_reqs) == 0:
        return default_chunk_size

    now = time.monotonic()
    decode_bs = len(decode_reqs)
    prefill_prefix_len = max(prefill_prefix_len, 0)
    plan_overhead_ms = edf_plan_overhead_ms(decode_reqs, prefill_prefix_len)

    # Slack budget for the next forward step. Decode loop runs continuously, so
    # the previous step's time-of-completion is essentially "now". We track an
    # optional per-req `_edf_last_step_time` (updated by scheduler) to account
    # for stalls; if absent, fall back to now (full ITL_SLO budget).
    min_slack_ms = math.inf
    bind_a = 1.0
    for req in decode_reqs:
        if ITL_QUANTILE > 0 and getattr(req, "_edf_slow", 0) + 1 <= (1.0 - ITL_QUANTILE) * (
            getattr(req, "_edf_steps", 0) + 1
        ):
            continue
        last_t = getattr(req, "_edf_last_step_time", now)
        elapsed_ms = (now - last_t) * 1000.0
        a = edf_tokens_per_step(req)
        slack = ITL_SLO_MS * a - elapsed_ms
        if slack < min_slack_ms:
            min_slack_ms, bind_a = slack, a
    # The budget is set by the request with the least slack; record that request's tokens/step
    # estimate beside the global one to show what the minimum selects.
    _bind["n"] += 1
    _bind["a"] += bind_a
    _bind["g"] += _acc_global if _acc_global is not None else 1.0
    if _bind["n"] % 2000 == 0:
        logger.info("EDF_BIND decisions=%d mean_binding_a=%.3f mean_global_a=%.3f",
                    _bind["n"], _bind["a"] / _bind["n"], _bind["g"] / _bind["n"])

    if min_slack_ms == math.inf:
        return default_chunk_size  # no request is out of its slow-step allowance

    # If pure decode alone already exceeds slack, adding prefill can only deepen
    # the violation. A pure-decode step lets overdue requests catch up and
    # refreshes their slack before the next admission decision.
    decode_only_ms = (
        model_predict_fn(0, 0, decode_avg_kv, decode_bs) * LATENCY_SAFETY_MARGIN
        + plan_overhead_ms
    )
    if min_slack_ms <= 0 or decode_only_ms > min_slack_ms:
        return 0

    # Binary search for max chunk size that fits within slack
    lo, hi = 0, default_chunk_size
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        model_n_p, model_r_p = project_extreme_prefix_work(
            mid, prefill_prefix_len
        )
        predicted_ms = (
            model_predict_fn(model_n_p, model_r_p, decode_avg_kv, decode_bs)
            * LATENCY_SAFETY_MARGIN
            + plan_overhead_ms
        )
        if predicted_ms <= min_slack_ms:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best


class LatencyPredictor:
    """Wrapper for the trained MLP server latency prediction model.

    Separate models must be trained for LPT and DMC server configurations.
    Set SGLANG_EDF_LATENCY_MODEL to the appropriate model .pt file.
    """

    def __init__(self, model_path: str):
        """Load trained model from the specified .pt file."""
        import torch
        import torch.nn as nn

        logger.info(f"Loading EDF latency model from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.X_mean = checkpoint["X_mean"]
        self.X_std = checkpoint["X_std"]

        # Reconstruct MLP: input_dim=5 (n_p, r_p, r_d, decode_bs, overlap)
        hidden_dims = [128, 64, 32]
        layers = []
        prev = 5
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))

        class _Net(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.net = nn.Sequential(*layers)
            def forward(self, x):
                return self.net(x).squeeze(-1)

        self.model = _Net(layers)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        # This predictor is invoked once for pure decode and O(log(chunk_size))
        # times by every EDF admission decision. Calling a tiny CPU MLP through
        # PyTorch has substantially more dispatcher overhead than arithmetic
        # (about 55 us per call on the serving host). Keep the loaded module for
        # checkpoint compatibility, but extract the four Linear layers once and
        # evaluate the hot path directly with NumPy.
        self.X_mean = np.asarray(self.X_mean, dtype=np.float32)
        self.X_std = np.asarray(self.X_std, dtype=np.float32)
        self._linear_params = [
            (
                layer.weight.detach().cpu().numpy().astype(np.float32, copy=True),
                layer.bias.detach().cpu().numpy().astype(np.float32, copy=True),
            )
            for layer in self.model.net
            if isinstance(layer, nn.Linear)
        ]
        logger.info(f"EDF latency model loaded: features={checkpoint.get('feature_names', 'unknown')}")

    def predict(self, n_p: int, r_p: float, r_d: float, decode_bs: int) -> float:
        """Predict server step latency in milliseconds.

        Uses overlap=False (non-overlapped execution) for conservative estimate.
        """
        x = np.asarray([n_p, r_p, r_d, decode_bs, 0.0], dtype=np.float32)
        x_norm = (x - self.X_mean) / self.X_std
        for i, (weight, bias) in enumerate(self._linear_params):
            x_norm = weight @ x_norm + bias
            if i + 1 < len(self._linear_params):
                np.maximum(x_norm, 0, out=x_norm)
        return float(np.expm1(x_norm[0]))


class LinearStepPredictor:
    """Whole-step latency, linear in the step's work terms, fit from STEPTRACE logs
    (tasks/perf_model/fit_mtp_step.py). Used for the speculative server, whose step (draft
    steps + one target forward + draft extends) none of the non-speculative models describe:

        ms = c0 + c_tok*n_p + c_attn*(n_p*r_p + n_p^2/2)/1e6 + c_bs*bs + c_kv*bs*avg_kv/1e3

    Interface-compatible with LatencyPredictor.predict().
    """

    def __init__(self, path: str):
        import json

        with open(path) as f:
            self.c = json.load(f)["coef"]
        logger.info("EDF linear step model loaded from %s: %s", path, self.c)

    def predict(self, n_p: float, r_p: float, decode_avg_kv: float, decode_bs: float) -> float:
        c = self.c
        return (
            c["c0"]
            + c["c_tok"] * n_p
            + c["c_attn"] * (n_p * r_p + 0.5 * n_p * n_p) / 1e6
            + c["c_bs"] * decode_bs
            + c["c_kv"] * decode_bs * decode_avg_kv / 1e3
        )


def try_load_latency_predictor():
    """Attempt to load the latency predictor from SGLANG_EDF_LATENCY_MODEL.

    ``SGLANG_EDF_LATENCY_MODEL=analytic`` selects the closed-form overlap cost
    model (managers/overlap_cost_model.py): no checkpoint, no per-workload
    training, constants fit once per GPU from microbenchmarks. It exists
    because the fused overlap kernel's mixed-step latency follows
    max + beta*min of its parts (3.1% MAPE), which a scheduler can invert;
    the split-kernel baseline's mixed cost is super-additive in its parts and
    admits no such model.

    An explicitly configured predictor is also loaded when EDF scheduling is
    disabled so /get_load can publish forward-cost telemetry independently.
    """
    if LATENCY_MODEL_PATH.startswith("linear:"):
        return LinearStepPredictor(LATENCY_MODEL_PATH[len("linear:"):])
    if LATENCY_MODEL_PATH == "analytic":
        from sglang.srt.managers.overlap_cost_model import AnalyticOverlapPredictor

        return AnalyticOverlapPredictor()
    if not LATENCY_MODEL_PATH:
        if EDF_ENABLED:
            logger.warning(
                "EDF scheduling enabled but SGLANG_EDF_LATENCY_MODEL not set. "
                "SLO-aware chunk size control will be disabled."
            )
        return None
    if not os.path.exists(LATENCY_MODEL_PATH):
        logger.warning(
            f"EDF latency model not found at {LATENCY_MODEL_PATH}. "
            "SLO-aware chunk size control will be disabled."
        )
        return None
    try:
        predictor = LatencyPredictor(LATENCY_MODEL_PATH)
        if EDF_ENABLED:
            logger.info(
                "EDF controller config: TTFT=%.1fms, ITL=%.1fms, "
                "latency_safety=%.3f, plan_overhead=%s, "
                "prefix_projection=%.0f, KV_watermark=%.3f",
                TTFT_SLO_MS,
                ITL_SLO_MS,
                LATENCY_SAFETY_MARGIN,
                PLAN_OVERHEAD_SETTING,
                LONG_PREFIX_PROJECTION_THRESHOLD,
                KV_ADMISSION_WATERMARK,
            )
        return predictor
    except Exception as e:
        logger.error(f"Failed to load EDF latency model: {e}")
        return None
