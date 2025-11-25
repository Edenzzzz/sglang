from __future__ import annotations

import dataclasses
import logging
import os
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleDraftInput


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GenerationBatchResult:
    logits_output: Optional[LogitsProcessorOutput] = None
    pp_hidden_states_proxy_tensors: Optional[PPProxyTensors] = None
    next_token_ids: Optional[torch.Tensor] = None
    num_accepted_tokens: Optional[int] = None
    can_run_cuda_graph: bool = False

    # For output processing
    extend_input_len_per_req: Optional[List[int]] = None
    extend_logprob_start_len_per_req: Optional[List[int]] = None

    # For overlap scheduling
    copy_done: Optional[torch.cuda.Event] = None
    delay_sample_func: Optional[callable] = None
    future_indices: Optional[FutureIndices] = None

    # FIXME(lsyin): maybe move to a better place?
    # sync path: forward stream -> output processor
    accept_lens: Optional[torch.Tensor] = None
    allocate_lens: Optional[torch.Tensor] = None

    # relay path: forward stream -> next step forward
    next_draft_input: Optional[EagleDraftInput] = None

    def copy_to_cpu(self, return_logprob: bool = False):
        """Copy tensors to CPU in overlap scheduling.
        Only the tensors which are needed for processing results are copied,
        e.g., next_token_ids, logits outputs
        """
        if return_logprob:
            if self.logits_output.next_token_logits is not None:
                self.logits_output.next_token_logits = (
                    self.logits_output.next_token_logits.to("cpu", non_blocking=True)
                )
            if self.logits_output.input_token_logprobs is not None:
                self.logits_output.input_token_logprobs = (
                    self.logits_output.input_token_logprobs.to("cpu", non_blocking=True)
                )
        if self.logits_output.hidden_states is not None:
            self.logits_output.hidden_states = self.logits_output.hidden_states.to(
                "cpu", non_blocking=True
            )
        self.next_token_ids = self.next_token_ids.to("cpu", non_blocking=True)

        if self.accept_lens is not None:
            self.accept_lens = self.accept_lens.to("cpu", non_blocking=True)

        if self.allocate_lens is not None:
            self.allocate_lens = self.allocate_lens.to("cpu", non_blocking=True)

        self.copy_done.record()

    @classmethod
    def from_pp_proxy(
        cls, logits_output, next_pp_outputs: PPProxyTensors, can_run_cuda_graph
    ):
        # TODO(lsyin): refactor PP and avoid using dict
        proxy_dict = next_pp_outputs.tensors
        return cls(
            logits_output=logits_output,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=next_pp_outputs["next_token_ids"],
            extend_input_len_per_req=proxy_dict.get("extend_input_len_per_req", None),
            extend_logprob_start_len_per_req=proxy_dict.get(
                "extend_logprob_start_len_per_req", None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
        )


def validate_input_length(
    req: Req, max_req_input_len: int, allow_auto_truncate: bool
) -> Optional[str]:
    """Validate and potentially truncate input length.

    Args:
        req: The request containing input_ids to validate
        max_req_input_len: Maximum allowed input length
        allow_auto_truncate: Whether to truncate long inputs

    Returns:
        Error message if validation fails, None if successful
    """
    if len(req.origin_input_ids) >= max_req_input_len:
        if allow_auto_truncate:
            logger.warning(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated. "
                f"{len(req.origin_input_ids)=}, {max_req_input_len=}."
            )
            req.origin_input_ids = req.origin_input_ids[:max_req_input_len]
            return None
        else:
            error_msg = (
                f"Input length ({len(req.origin_input_ids)} tokens) exceeds "
                f"the maximum allowed length ({max_req_input_len} tokens). "
                f"Use a shorter input or enable --allow-auto-truncate."
            )
            return error_msg

    return None


def get_logprob_dict_from_result(result: GenerationBatchResult) -> dict:

    logits_output = result.logits_output
    assert logits_output is not None

    return {
        "extend_input_len_per_req": result.extend_input_len_per_req,
        "extend_logprob_start_len_per_req": result.extend_logprob_start_len_per_req,
        "next_token_logprobs": result.logits_output.next_token_logprobs,
        "next_token_top_logprobs_val": result.logits_output.next_token_top_logprobs_val,
        "next_token_top_logprobs_idx": result.logits_output.next_token_top_logprobs_idx,
        "next_token_token_ids_logprobs_val": result.logits_output.next_token_token_ids_logprobs_val,
        "next_token_token_ids_logprobs_idx": result.logits_output.next_token_token_ids_logprobs_idx,
        "input_token_logprobs": result.logits_output.input_token_logprobs,
        "input_top_logprobs_val": result.logits_output.input_top_logprobs_val,
        "input_top_logprobs_idx": result.logits_output.input_top_logprobs_idx,
        "input_token_ids_logprobs_val": result.logits_output.input_token_ids_logprobs_val,
        "input_token_ids_logprobs_idx": result.logits_output.input_token_ids_logprobs_idx,
    }


def get_logprob_from_pp_outputs(
    next_pp_outputs: PPProxyTensors,
) -> tuple[LogitsProcessorOutput, list[int], list[int]]:
    logits_output = LogitsProcessorOutput(
        # Do not send logits and hidden states because they are large
        next_token_logits=None,
        hidden_states=None,
        next_token_logprobs=next_pp_outputs["next_token_logprobs"],
        next_token_top_logprobs_val=next_pp_outputs["next_token_top_logprobs_val"],
        next_token_top_logprobs_idx=next_pp_outputs["next_token_top_logprobs_idx"],
        next_token_token_ids_logprobs_val=next_pp_outputs[
            "next_token_token_ids_logprobs_val"
        ],
        next_token_token_ids_logprobs_idx=next_pp_outputs[
            "next_token_token_ids_logprobs_idx"
        ],
        input_token_logprobs=next_pp_outputs["input_token_logprobs"],
        input_top_logprobs_val=next_pp_outputs["input_top_logprobs_val"],
        input_top_logprobs_idx=next_pp_outputs["input_top_logprobs_idx"],
        input_token_ids_logprobs_val=next_pp_outputs["input_token_ids_logprobs_val"],
        input_token_ids_logprobs_idx=next_pp_outputs["input_token_ids_logprobs_idx"],
    )
    extend_input_len_per_req = next_pp_outputs["extend_input_len_per_req"]
    extend_logprob_start_len_per_req = next_pp_outputs[
        "extend_logprob_start_len_per_req"
    ]

    return logits_output, extend_input_len_per_req, extend_logprob_start_len_per_req


def plot_chunk_size_trends(
    csv_path: str, output_dir: str = ".", use_logger: bool = True
) -> bool:
    """
    Generate chunk size trend plots from CSV data.

    Args:
        csv_path: Path to the chunk_size_trend.csv file
        output_dir: Directory to save the plots (default: current directory)
        use_logger: If True, use logger.info/warning, else use print

    Returns:
        True if plots were generated successfully, False otherwise
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        if use_logger:
            logger.warning("matplotlib or pandas not available, skipping plotting")
        else:
            print("Error: matplotlib or pandas not available")
        return False

    log_func = logger.info if use_logger else print
    warn_func = logger.warning if use_logger else print

    # Load data from CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        if use_logger:
            logger.warning(f"CSV file not found: {csv_path}")
        else:
            print(f"Error: File not found: {csv_path}")
        return False
    except Exception as e:
        if use_logger:
            logger.warning(f"Error reading CSV file: {e}")
        else:
            print(f"Error: CSV file is empty or invalid: {csv_path}")
        return False

    if df.empty:
        if use_logger:
            logger.warning("CSV file contains no data")
        else:
            print("Error: CSV file contains no data")
        return False

    # Separate prefill and decode data
    prefill_data = df[df["prefill_tokens"].notna()].copy()
    decode_data = df[df["decode_bs"].notna()].copy()

    # Get retracted requests data (from decode data, as retractions happen during decode)
    retracted_data = df[df["num_retracted_reqs"].notna()].copy()

    # Determine overall time range (min and max across all data)
    all_times = []
    if not prefill_data.empty:
        all_times.extend(prefill_data["time"].tolist())
    if not decode_data.empty:
        all_times.extend(decode_data["time"].tolist())
    if not retracted_data.empty:
        all_times.extend(retracted_data["time"].tolist())

    if not all_times:
        if use_logger:
            logger.warning("No valid data found in CSV file")
        else:
            print("Warning: No valid data found in CSV file")
        return False

    time_min = min(all_times)
    time_max = max(all_times)

    # Use fixed 9 ticks (8 intervals)
    tick_interval = (time_max - time_min) / 8
    x_ticks = []
    current = time_min
    for _ in range(9):
        x_ticks.append(current)
        current += tick_interval
    # Ensure the last tick is exactly time_max
    x_ticks[-1] = time_max

    # Plot 1: Prefill tokens and new token ratio (dual y-axis)
    if not prefill_data.empty:
        # Calculate mean prefill length
        mean_prefill_len = prefill_data["prefill_tokens"].mean()
        prefill_times = prefill_data["time"].tolist()
        prefill_tokens = prefill_data["prefill_tokens"].tolist()
        new_token_ratios = prefill_data["new_token_ratio"].tolist()

        # Extend prefill data with zeros for times beyond the last prefill batch
        if not decode_data.empty:
            last_prefill_time = max(prefill_times) if prefill_times else time_min
            decode_times = decode_data["time"].tolist()
            # Find decode times that are beyond the last prefill time
            extended_times = [t for t in decode_times if t > last_prefill_time]
            if extended_times:
                # Sort and add to prefill data
                extended_times = sorted(set(extended_times))
                prefill_times.extend(extended_times)
                prefill_tokens.extend([0] * len(extended_times))
                new_token_ratios.extend([0] * len(extended_times))
                # Sort all data by time
                combined = sorted(zip(prefill_times, prefill_tokens, new_token_ratios))
                prefill_times, prefill_tokens, new_token_ratios = zip(*combined)
                prefill_times = list(prefill_times)
                prefill_tokens = list(prefill_tokens)
                new_token_ratios = list(new_token_ratios)

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        fig1.patch.set_facecolor("white")

        # Right y-axis for new token ratio (plot first so prefill tokens stay on top)
        ax1_twin = ax1.twinx()
        ax1_twin.set_zorder(1)  # Lower zorder so it's drawn first (behind)
        ax1.set_zorder(2)  # Higher zorder so it's drawn last (on top)
        ax1.patch.set_visible(False)  # Make ax1 transparent so we can see through it

        color2 = "tab:red"
        ax1_twin.set_ylabel("New(decode) Token Ratio", fontsize=12, color=color2)
        line2 = ax1_twin.plot(
            prefill_times,
            new_token_ratios,
            color=color2,
            linewidth=1.5,
            label="New Token Ratio",
        )
        ax1_twin.tick_params(axis="y", labelcolor=color2)

        # Left y-axis for prefill tokens (plot last so it stays on top)
        color1 = "tab:blue"
        ax1.set_xlabel("Time (seconds)", fontsize=12)
        ax1.set_ylabel("Prefill Tokens", fontsize=12, color=color1)
        line1 = ax1.plot(
            prefill_times,
            prefill_tokens,
            color=color1,
            linewidth=1.5,
            label="Prefill Tokens",
        )
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_xlim(time_min, time_max)
        ax1.set_xticks(x_ticks)
        ax1.grid(True, alpha=0.3)

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left")
        ax1.set_title("Prefill Tokens and New Token Ratio Over Time", fontsize=14)

        plt.tight_layout()
        filename = os.path.join(
            output_dir, f"chunk_size_trend_prefill_mean_{mean_prefill_len:.1f}.png"
        )
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig1)
        log_func(f"Prefill chunk size trend plot saved to {filename}")

    # Plot 2: Decode batch size and average KV length (dual y-axis)
    if not decode_data.empty:
        # Calculate mean decode KV length
        mean_decode_kv_len = decode_data["avg_kv_length"].mean()
        decode_times = decode_data["time"].tolist()
        decode_bs = decode_data["decode_bs"].tolist()
        avg_kv_lengths = decode_data["avg_kv_length"].tolist()

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        fig2.patch.set_facecolor("white")

        # Left y-axis for decode batch size
        color1 = "tab:blue"
        ax2.set_xlabel("Time (seconds)", fontsize=12)
        ax2.set_ylabel("Decode Batch Size", fontsize=12, color=color1)
        line1 = ax2.plot(
            decode_times,
            decode_bs,
            color=color1,
            linewidth=1.5,
            label="Decode Batch Size",
        )
        ax2.tick_params(axis="y", labelcolor=color1)
        ax2.set_xlim(time_min, time_max)
        ax2.set_xticks(x_ticks)
        ax2.grid(True, alpha=0.3)

        # Right y-axis for average KV length
        ax2_twin = ax2.twinx()
        color2 = "tab:red"
        ax2_twin.set_ylabel("Average KV Length", fontsize=12, color=color2)
        line2 = ax2_twin.plot(
            decode_times,
            avg_kv_lengths,
            color=color2,
            linewidth=1.5,
            label="Average KV Length",
        )
        ax2_twin.tick_params(axis="y", labelcolor=color2)

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc="upper left")
        ax2.set_title("Decode Batch Size and Average KV Length Over Time", fontsize=14)

        plt.tight_layout()
        filename = os.path.join(
            output_dir,
            f"chunk_size_trend_decode_mean_{mean_decode_kv_len:.1f}.png",
        )
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        log_func(f"Decode chunk size trend plot saved to {filename}")

    # Plot 3: Number of retracted requests
    if not retracted_data.empty:
        retracted_times = retracted_data["time"].tolist()
        num_retracted = retracted_data["num_retracted_reqs"].tolist()

        fig3, ax3 = plt.subplots(figsize=(12, 6))
        fig3.patch.set_facecolor("white")

        color = "yellow"
        ax3.set_xlabel("Time (seconds)", fontsize=12)
        ax3.set_ylabel("Number of Retracted Requests", fontsize=12, color=color)
        ax3.plot(
            retracted_times,
            num_retracted,
            color=color,
            linewidth=1.5,
            label="Retracted Requests",
        )
        ax3.tick_params(axis="y", labelcolor=color)
        ax3.set_xlim(time_min, time_max)
        ax3.set_xticks(x_ticks)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper left")
        ax3.set_title("Number of Retracted Requests Over Time", fontsize=14)

        plt.tight_layout()
        filename = os.path.join(
            output_dir,
            "chunk_size_trend_retracted_requests.png",
        )
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        log_func(f"Retracted requests trend plot saved to {filename}")

    return True
