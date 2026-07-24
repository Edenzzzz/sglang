#!/usr/bin/env python3
"""
POD overlap performance profiler.

This module profiles mixed prefill+decode batches with different sequence
configurations to collect data for training the POD overlap performance model.

It is intended to be called from the scheduler / executor when handling a
`profile_seq_lens`-style request, not as a standalone CLI.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class ProfilingRunner:
    """Profile model execution time for different mixed prefill+decode configs.

    This uses the existing `ModelRunner` and attention backend. Callers are
    expected to:
    - Construct `ModelRunner` normally (with chunked prefill / mixed chunk enabled).
    - Toggle POD / persistent attention via environment variables (
    - Run profiling from the scheduler when a
      special `profile_seq_lens` request is received.
    """

    def __init__(self, model_runner: ModelRunner, max_num_tokens: int):
        self.model_runner = model_runner
        self.device = model_runner.device
        self.max_num_tokens = max_num_tokens
        self.server_args = model_runner.server_args
        assert (
            self.server_args.enable_piecewise_cuda_graph
        ), "Piecewise CUDA graph must be enabled for overlap profiling"

        # Pre-allocate buffers (reused across all configs)
        with torch.device(self.device):
            self.input_ids = torch.zeros((max_num_tokens,), dtype=torch.int64)
            self.out_cache_loc = torch.zeros((max_num_tokens,), dtype=torch.int32)
            self.positions = torch.zeros((max_num_tokens,), dtype=torch.int64)
            self.req_pool_indices = torch.zeros((max_num_tokens,), dtype=torch.int64)

    @torch.inference_mode()
    def profile_one_seq_config(
        self,
        n_p: int,  # new prefill tokens
        r_p: int,  # prefill prefix length
        r_d: int,  # decode KV length
        decode_bs: int,  # decode batch size
        repeat: int = 5,
    ) -> float:
        """Profile one configuration and return avg execution time in milliseconds.

        Args:
            n_p: New prefill tokens (extend_input_len for prefill)
            r_p: Prefill prefix length
            r_d: Decode KV length (seq_len for decode requests)
            decode_bs: Decode batch size
            repeat: Number of timed forward passes (after warmup)
        """
        assert decode_bs >= 0

        # Total batch size: 1 prefill + decode_bs decode
        total_bs = 1 + decode_bs

        # Decode allocation advances each sequence by one before the model
        # worker batch is built.  Therefore a request with r_d cached KV tokens
        # has seq_len=r_d+1 and extend_len=1 in a mixed forward.
        seq_lens = [n_p + r_p] + [r_d + 1] * decode_bs
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64, device=self.device)
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)

        # extend_prefix_lens: [r_p, r_d, r_d, ...]
        extend_prefix_lens = [r_p] + [r_d] * decode_bs
        prefix_lens_sum = r_p + r_d * decode_bs
        extend_prefix_lens_tensor = torch.tensor(
            extend_prefix_lens, dtype=torch.int32, device=self.device
        )
        extend_prefix_lens_cpu = extend_prefix_lens

        # extend_lens: [n_p, 1, 1, ...]
        extend_lens = [n_p] + [1] * decode_bs
        extend_lens_tensor = torch.tensor(
            extend_lens, dtype=torch.int32, device=self.device
        )

        # Total numbers
        seq_lens_sum = n_p + r_p + (r_d + 1) * decode_bs
        extend_num_tokens = n_p + decode_bs
        assert (
            extend_num_tokens <= self.max_num_tokens
        ), f"extend_num_tokens {extend_num_tokens} > max_num_tokens {self.max_num_tokens}"

        required_kv_tokens = seq_lens_sum
        if required_kv_tokens > self.model_runner.max_total_num_tokens:
            raise ValueError(
                f"Synthetic batch needs {required_kv_tokens} KV tokens, but the "
                f"server has {self.model_runner.max_total_num_tokens}"
            )
        if total_bs > self.model_runner.req_to_token_pool.size:
            raise ValueError(
                f"Synthetic batch needs {total_bs} request slots, but the server "
                f"has {self.model_runner.req_to_token_pool.size}"
            )

        # Use pre-allocated input buffers and realistic, disjoint request/KV
        # mappings. Pointing every synthetic request at request/page zero gives
        # attention an artificial perfect-locality workload and also routes all
        # new KV writes to the allocator's padding slot.
        input_ids = self.input_ids[:extend_num_tokens]
        out_cache_loc = self.out_cache_loc[:extend_num_tokens]
        positions = self.positions[:extend_num_tokens]
        req_pool_indices = self.req_pool_indices[:total_bs]
        req_pool_indices.copy_(
            torch.arange(total_bs, dtype=torch.int64, device=self.device)
        )

        physical_start = 1  # KV slot zero is reserved for graph padding.
        extend_start = 0
        for req_index, (seq_len, prefix_len, extend_len) in enumerate(
            zip(seq_lens, extend_prefix_lens, extend_lens)
        ):
            physical_end = physical_start + seq_len
            self.model_runner.req_to_token_pool.req_to_token[
                req_index, :seq_len
            ] = torch.arange(
                physical_start,
                physical_end,
                dtype=torch.int32,
                device=self.device,
            )
            out_cache_loc[extend_start : extend_start + extend_len] = torch.arange(
                physical_start + prefix_len,
                physical_end,
                dtype=torch.int32,
                device=self.device,
            )
            positions[extend_start : extend_start + extend_len] = torch.arange(
                prefix_len,
                seq_len,
                dtype=torch.int64,
                device=self.device,
            )
            physical_start = physical_end
            extend_start += extend_len

        # Compute extend_start_loc: cumulative sum of extend_seq_lens
        # extend_start_loc[i] = sum(extend_seq_lens[0:i])
        # Technically, only required for triton backend and hybrid attention.
        extend_start_loc = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=self.device),
                torch.cumsum(extend_lens_tensor[:-1], dim=0),
            ]
        )

        # Determine forward mode
        if n_p > 0:
            if r_d > 0:
                forward_mode = ForwardMode.MIXED
            else:
                forward_mode = ForwardMode.EXTEND
        else:
            forward_mode = ForwardMode.DECODE

        forward_batch = ForwardBatch(
            forward_mode=forward_mode,
            batch_size=total_bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens_tensor,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens_sum,
            orig_seq_lens=seq_lens_tensor,
            seq_lens_cpu=seq_lens_cpu,
            return_logprob=False,
            next_token_logits_buffer=None,
            positions=positions,
            extend_num_tokens=extend_num_tokens,
            extend_prefix_lens=extend_prefix_lens_tensor,
            prefix_lens_sum=prefix_lens_sum,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens=torch.tensor(
                extend_lens, dtype=torch.int32, device=self.device
            ),
            extend_seq_lens_cpu=extend_lens,
            extend_start_loc=extend_start_loc,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_forward_mode=forward_mode,
            decoding_bs=decode_bs,
        )

        # Measure the deployed model-runner path, including attention metadata
        # planning, CUDA-graph selection/replay, the full model forward, and
        # host-side launch overhead.  The previous profiler initialized metadata
        # once, disabled graph replay, and timed CUDA kernels only; models trained
        # on that target systematically missed costs paid by the live server.
        times: List[float] = []
        for _ in range(repeat):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            # Let ModelRunner prepare the backend itself.  In particular, mixed
            # CUDA-graph replay_prepare selects and pads the capture by token
            # count; pre-initializing and passing skip_attn_backend_init=True
            # leaves stale replay dimensions from the preceding server batch.
            self.model_runner.forward(forward_batch, skip_attn_backend_init=False)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start_time) * 1000)

        return float(np.median(times))


def generate_default_configs() -> List[Tuple[int, int, int, int]]:
    """Generate the default (n_p, r_p, r_d, decode_bs) configs.

    Sampling scheme (as requested):
      1) new_prefill_tokens: {(256, 4096, 256), 6000, 8192}
      2) prefill_prefix_len: {(256, 4096, 256)}
      3) decode_kv_len: {(256, 10000, 512)}

    For each r_d, we cap decode_bs by: max_decode_bs <= 4096 * 100 / r_d,
    then sample a few representative decode_bs uniformly in (0, max_decode_bs].
    """
    configs: List[Tuple[int, int, int, int]] = []

    n_p_values = list(range(256, 4097, 256)) + [6000, 8192]
    r_p_values = list(range(0, 4097, 256))
    r_d_values = list(range(256, 10001, 512))

    for n_p in n_p_values:
        for r_p in r_p_values:
            for r_d in r_d_values:
                max_decode_bs = (4096 * 100) // r_d
                if max_decode_bs <= 0:
                    continue

                candidates = [
                    max_decode_bs // 4,
                    max_decode_bs // 2,
                    (3 * max_decode_bs) // 4,
                    max_decode_bs,
                ]
                decode_bs_values = sorted(
                    {bs for bs in candidates if bs > 0 and bs <= 350}
                )

                for decode_bs in decode_bs_values:
                    configs.append((n_p, r_p, r_d, decode_bs))
    random_configs = generate_validation_configs(1800, seed=9999)
    long_prefill_configs = generate_long_prefill_configs(1000, seed=9999)
    return configs + random_configs + long_prefill_configs


def generate_long_prefill_configs(
    num_configs: int,
    seed: int = 42,
) -> List[Tuple[int, int, int, int]]:
    """Generate long prefill configs for document summary/QA dataset such as LooGLE"""
    rng = np.random.RandomState(seed)
    configs: List[Tuple[int, int, int, int]] = []
    for _ in range(num_configs):
        n_p = int(rng.uniform(10000, 21000))
        r_p = int(rng.uniform(0, 4097))
        r_d = int(rng.uniform(5, 300))
        max_decode_bs = min(150, (4096 * 100) // r_d)
        if max_decode_bs <= 0:
            continue
        decode_bs = int(rng.uniform(1, max_decode_bs + 1))
        configs.append((n_p, r_p, r_d, decode_bs))
    return configs


def generate_validation_configs(
    num_configs: int, seed: int = 42
) -> List[Tuple[int, int, int, int]]:
    """Generate random validation configs within the same ranges but without step sizes.

    Randomly samples from:
      - n_p: [256, 8192] (continuous, not just step sizes)
      - r_p: [256, 4096] (continuous, not just step sizes)
      - r_d: [256, 10000] (continuous, not just step sizes)
      - decode_bs: [1, min(350, 4096 * 100 / r_d)] (continuous)

    Args:
        num_configs: Number of validation configs to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of (n_p, r_p, r_d, decode_bs) tuples.
    """
    rng = np.random.RandomState(seed)
    configs: List[Tuple[int, int, int, int]] = []

    for _ in range(num_configs):
        # Sample n_p: [256, 8192] (allow 6000 and 8192 as special values, otherwise uniform)
        rand = rng.random()
        if rand < 0.1:  # 10% chance for special values
            n_p = int(rng.uniform(6000, 8193))
        if rand > 0.9:  # 10% chance for long prefill
            n_p = int(rng.uniform(10000, 21000))
        else:
            n_p = int(rng.uniform(256, 4097))  # Up to 4096, but allow 4096

        # Sample r_p: [256, 4096]
        r_p = int(rng.uniform(256, 4097))

        # Sample r_d: [256, 10000]
        r_d = int(rng.uniform(256, 10001))

        # Sample decode_bs: [1, min(350, 4096 * 100 / r_d)]
        max_decode_bs = min(350, (4096 * 100) // r_d)
        if max_decode_bs <= 0:
            continue
        decode_bs = int(rng.uniform(1, max_decode_bs + 1))

        configs.append((n_p, r_p, r_d, decode_bs))

    return configs


@torch.inference_mode()
def profile_configs(
    model_runner: ModelRunner,
    configs: Sequence[Tuple[int, int, int, int]],
    max_num_tokens: int,
    overlap: bool,
    output_path: str,
    repeat: int = 5,
) -> pd.DataFrame:
    """Profile a set of (n_p, r_p, r_d, decode_bs) configs.

    This toggles POD / persistent attention via environment variables:
      - overlap=False: USE_POD=False, USE_PERSISTENT=False
      - overlap=True:  USE_POD=True,  USE_PERSISTENT=False

    Args:
        model_runner: The model runner to profile.
        configs: Sequence of (n_p, r_p, r_d, decode_bs) tuples to profile.
        max_num_tokens: Maximum number of tokens to profile.
        overlap: Whether to use POD overlap mode.
        output_path: Path to CSV file. If provided, results are appended
            progressively. If None, results are returned as a DataFrame.

    Returns:
        A pandas DataFrame with columns: [n_p, r_p, r_d, decode_bs, time_ms]
        Feature computation (S_p, S_d, R_pd, term1-4) should be done in the training script.
    """
    valid_configs = [
        config
        for config in configs
        if config[0] + config[1] + (config[2] + 1) * config[3]
        <= model_runner.max_total_num_tokens
        and config[3] + 1 <= model_runner.req_to_token_pool.size
        and config[0] + config[1] <= model_runner.req_to_token_pool.max_context_len
        and config[2] + 1 <= model_runner.req_to_token_pool.max_context_len
    ]
    if len(valid_configs) != len(configs):
        logger.info(
            "Dropped %d synthetic configs that exceed this server's KV/request capacity",
            len(configs) - len(valid_configs),
        )
    configs = valid_configs
    if not configs:
        raise ValueError("No profiling configs fit this server")

    # Configure attention backend mode
    if overlap:
        envs.SGLANG_FLASHINFER_USE_POD.set(True)
        envs.SGLANG_FLASHINFER_USE_PERSISTENT.set(False)
    else:
        envs.SGLANG_FLASHINFER_USE_POD.set(False)
        # envs.SGLANG_FLASHINFER_USE_PERSISTENT.set(False)

    runner = ProfilingRunner(model_runner, max_num_tokens)
    # warmup
    runner.profile_one_seq_config(*configs[0], repeat=1)

    # Check if output file exists to determine if we need to write headers
    write_header = output_path is None or not os.path.exists(output_path)

    print(f"Profiling {len(configs)} configs")
    for n_p, r_p, r_d, decode_bs in tqdm(
        configs, desc="Profiling configs", leave=False
    ):
        time_ms = runner.profile_one_seq_config(n_p, r_p, r_d, decode_bs, repeat=repeat)
        result = {
            "n_p": n_p,
            "r_p": r_p,
            "r_d": r_d,
            "decode_bs": decode_bs,
            "time_ms": time_ms,
            "overlap": overlap,
        }

        # Append to CSV if output_path is provided
        if output_path is not None:
            df_row = pd.DataFrame([result])
            df_row.to_csv(
                output_path,
                mode="a",
                header=write_header,
                index=False,
            )
            write_header = False  # Only write header once

    return len(configs)


@torch.inference_mode()
def profile_default_grid(
    model_runner: ModelRunner,
    overlap: bool,
    output_path: str,
    max_configs: Optional[int] = None,
    resume_idx: Optional[int] = None,
    repeat: int = 5,
) -> pd.DataFrame:
    """Convenience helper that profiles the default grid of configs.

    Args:
        model_runner: The model runner to profile.
        overlap: Whether to use POD overlap mode.
        output_path: Path to CSV file for progressive output.
        max_configs: Maximum number of configs to profile.
        resume_idx: Resume profiling from this config index (0-based). If provided,
            configs before this index are skipped. Useful for resuming interrupted runs.
    """
    configs = generate_default_configs()
    max_num_tokens = max(n_p + r_p + r_d for n_p, r_p, r_d, _ in configs)

    # Apply resume_idx: skip configs before the resume index
    if resume_idx is not None and resume_idx > 0:
        configs = configs[resume_idx:]

    # Apply max_configs limit
    if max_configs is not None:
        configs = configs[:max_configs]
    if len(configs) == 0:
        logger.warning(
            "No configs to profile with resume_idx %d and max_configs %d",
            resume_idx,
            max_configs,
        )
        return pd.DataFrame()
    return profile_configs(
        model_runner,
        configs,
        max_num_tokens=max_num_tokens,
        overlap=overlap,
        output_path=output_path,
        repeat=repeat,
    )


@torch.inference_mode()
def profile_validation_configs(
    model_runner: ModelRunner,
    num_configs: int,
    seed: int,
    overlap: bool,
    output_path: str,
    resume_idx: Optional[int] = None,
    repeat: int = 5,
) -> int:
    """Profile random validation configs for model evaluation.

    Args:
        model_runner: The model runner to profile.
        num_configs: Number of validation configs to generate and profile.
        seed: Random seed for generating configs.
        overlap: Whether to use POD overlap mode.
        output_path: Path to CSV file for progressive output.
        resume_idx: Resume profiling from this config index (0-based). If provided,
            configs before this index are skipped. Useful for resuming interrupted runs.
    Returns:
        Number of configs profiled.
    """
    configs = generate_validation_configs(num_configs, seed=seed)
    # Calculate max_num_tokens from all configs (before slicing) to ensure sufficient buffer allocation
    max_num_tokens = max(n_p + r_p + r_d for n_p, r_p, r_d, _ in configs)

    if resume_idx is not None:
        if resume_idx < 0:
            resume_idx = 0
        if resume_idx >= len(configs):
            logger.warning(
                "resume_idx %d >= num_configs %d, nothing to profile",
                resume_idx,
                len(configs),
            )
            return 0
        logger.info(
            "Resuming from index %d: skipping first %d configs, profiling %d configs",
            resume_idx,
            resume_idx,
            len(configs) - resume_idx,
        )
        configs = configs[resume_idx:]
    if len(configs) == 0:
        logger.warning("No configs to profile with resume_idx %d", resume_idx)
        return 0
    return profile_configs(
        model_runner,
        configs,
        max_num_tokens=max_num_tokens,
        overlap=overlap,
        output_path=output_path,
        repeat=repeat,
    )
