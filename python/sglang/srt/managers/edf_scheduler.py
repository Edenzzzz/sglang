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
# Constant overhead (ms) added to model predictions for persistent kernel.
# The barrier scaffolding in BatchAttention plan() adds ~3ms CPU + ~4ms
# cooperative launch overhead per mixed batch step. Without this, chunk
# control allows chunks that produce max ITL ~7ms above the SLO threshold.
PLAN_OVERHEAD_MS = float(os.environ.get("SGLANG_EDF_PLAN_OVERHEAD_MS", "0"))
# overlap_profiler.py trains the checked-in models with r_p in [0, 4096].
# Clamp continued-prefill prefixes to that domain instead of allowing unstable
# MLP extrapolation for later chunks of 8K+ prompts.
MAX_PROFILED_PREFILL_PREFIX = float(
    os.environ.get("SGLANG_EDF_MAX_PROFILED_PREFILL_PREFIX", "4096")
)
# Clamping remains the best throughput/ITL point through the 12K workload, but
# becomes severely optimistic on later chunks of 16K prompts.  Only project
# attention work once the actual prefix crosses this empirically validated
# boundary.
LONG_PREFIX_PROJECTION_THRESHOLD = float(
    os.environ.get("SGLANG_EDF_LONG_PREFIX_PROJECTION_THRESHOLD", "12288")
)


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

    # Slack budget for the next forward step. Decode loop runs continuously, so
    # the previous step's time-of-completion is essentially "now". We track an
    # optional per-req `_edf_last_step_time` (updated by scheduler) to account
    # for stalls; if absent, fall back to now (full ITL_SLO budget).
    min_slack_ms = ITL_SLO_MS
    for req in decode_reqs:
        last_t = getattr(req, "_edf_last_step_time", now)
        elapsed_ms = (now - last_t) * 1000.0
        slack = ITL_SLO_MS - elapsed_ms
        if slack < min_slack_ms:
            min_slack_ms = slack

    # If pure decode alone already exceeds slack, adding prefill can only deepen
    # the violation. A pure-decode step lets overdue requests catch up and
    # refreshes their slack before the next admission decision.
    decode_only_ms = (
        model_predict_fn(0, 0, decode_avg_kv, decode_bs) * LATENCY_SAFETY_MARGIN
        + PLAN_OVERHEAD_MS
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
            + PLAN_OVERHEAD_MS
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


def try_load_latency_predictor() -> Optional[LatencyPredictor]:
    """Attempt to load the latency predictor from SGLANG_EDF_LATENCY_MODEL.

    Returns None if EDF is disabled or no model path is configured.
    """
    if not EDF_ENABLED:
        return None
    if not LATENCY_MODEL_PATH:
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
        return LatencyPredictor(LATENCY_MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to load EDF latency model: {e}")
        return None
