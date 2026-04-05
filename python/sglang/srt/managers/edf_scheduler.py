"""EDF (Earliest Deadline First) scheduling support for SLO-aware serving.

Provides:
1. Deadline computation from SLO targets
2. SLO-aware chunk size control using latency prediction model
3. Integration hooks for the existing scheduler

Enabled by SGLANG_EDF_SCHEDULING=1 environment variable.
Configure SLOs via SGLANG_TTFT_SLO_MS (default 2000) and SGLANG_ITL_SLO_MS (default 100).

Usage:
    The scheduler calls edf_compute_priority() to set request priorities based on deadlines.
    The scheduler calls edf_compute_max_chunk_size() to limit prefill tokens per SLO.
"""

import os
import time
import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

# Configuration from environment
EDF_ENABLED = os.environ.get("SGLANG_EDF_SCHEDULING", "0") == "1"
TTFT_SLO_MS = float(os.environ.get("SGLANG_TTFT_SLO_MS", "2000"))
ITL_SLO_MS = float(os.environ.get("SGLANG_ITL_SLO_MS", "100"))
# Input-length scaling: +1s TTFT per 5K extra tokens
TTFT_SCALE_TOKENS = float(os.environ.get("SGLANG_TTFT_SCALE_TOKENS", "5000"))
TTFT_SCALE_MS = float(os.environ.get("SGLANG_TTFT_SCALE_MS", "1000"))


def compute_ttft_slo_ms(input_len: int) -> float:
    """Compute TTFT SLO for a request based on its input length.

    Base TTFT + scaling for longer inputs (Ascendra-style).
    """
    extra_tokens = max(0, input_len - 4096)  # baseline assumes 4K input
    extra_ms = (extra_tokens / TTFT_SCALE_TOKENS) * TTFT_SCALE_MS
    return TTFT_SLO_MS + extra_ms


def compute_ttft_deadline(req: "Req") -> float:
    """Compute TTFT deadline timestamp for a request.

    Returns: absolute time (perf_counter) by which first token must be generated.
    """
    input_len = len(req.origin_input_ids)
    slo_ms = compute_ttft_slo_ms(input_len)
    arrival_time = req.time_stats.wait_queue_entry_time
    return arrival_time + slo_ms / 1000.0


def compute_itl_deadline(req: "Req") -> float:
    """Compute next ITL deadline for a decode request.

    Returns: absolute time by which the next token must be generated.
    """
    # last_tic is set after each decode step in schedule_batch.py
    return req.last_tic + ITL_SLO_MS / 1000.0


def edf_set_priority(req: "Req") -> None:
    """Set request priority to TTFT deadline (lower = earlier = higher priority).

    Call this when a request enters the waiting queue.
    Works with schedule_low_priority_values_first=True.
    """
    if not EDF_ENABLED:
        return
    req.priority = compute_ttft_deadline(req)


def edf_compute_max_chunk_size(
    decode_reqs: List["Req"],
    default_chunk_size: int,
    model_predict_fn=None,
    decode_avg_kv: float = 0,
) -> int:
    """Compute maximum prefill chunk size that won't violate decode ITL SLOs.

    Args:
        decode_reqs: Currently running decode requests
        default_chunk_size: Maximum chunk size (chunked_prefill_size)
        model_predict_fn: fn(n_p, r_p, r_d, decode_bs) -> predicted_ms
        decode_avg_kv: Average KV length of decode requests

    Returns:
        Adjusted chunk size (may be smaller than default if SLOs are tight)
    """
    if not EDF_ENABLED or model_predict_fn is None or len(decode_reqs) == 0:
        return default_chunk_size

    now = time.perf_counter()
    decode_bs = len(decode_reqs)

    # Find tightest ITL deadline among decode requests
    min_slack_ms = float("inf")
    for req in decode_reqs:
        deadline = compute_itl_deadline(req)
        slack_ms = (deadline - now) * 1000.0
        min_slack_ms = min(min_slack_ms, slack_ms)

    if min_slack_ms <= 0:
        # Already past deadline — run pure decode (no prefill)
        return 0

    # Binary search for max chunk size that fits within slack
    lo, hi = 0, default_chunk_size
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        predicted_ms = model_predict_fn(mid, 0, decode_avg_kv, decode_bs)
        if predicted_ms <= min_slack_ms:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best


class LatencyPredictor:
    """Wrapper for the trained MLP latency prediction model."""

    def __init__(self, model_path: str, kernel_type: str = "lpt"):
        """Load trained model.

        Args:
            model_path: Path to dual_model.pt
            kernel_type: "lpt" or "dmc"
        """
        import torch

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.X_mean = checkpoint["X_mean"]
        self.X_std = checkpoint["X_std"]
        self.input_dim = checkpoint["input_dim"]
        self.hidden_dims = checkpoint["hidden_dims"]

        from tasks.perf_model.train_perf_model_mlp import PerfMLP

        self.model = PerfMLP(self.input_dim, self.hidden_dims)
        state_key = f"{kernel_type}_state_dict"
        self.model.load_state_dict(checkpoint[state_key])
        self.model.eval()

    def predict(self, n_p: int, r_p: float, r_d: float, decode_bs: int) -> float:
        """Predict server step latency in milliseconds."""
        import torch

        x = np.array([[n_p, r_p, r_d, decode_bs]], dtype=np.float32)
        x_norm = (x - self.X_mean) / self.X_std
        with torch.no_grad():
            pred_log = self.model(torch.tensor(x_norm))
            return float(torch.expm1(pred_log).item())
