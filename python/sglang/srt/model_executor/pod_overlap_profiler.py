#!/usr/bin/env python3
"""
POD overlap performance profiler.

This module profiles mixed prefill+decode batches with different sequence
configurations to collect data for training the POD overlap performance model.

It is intended to be called from the scheduler / executor when handling a
`profile_seq_lens`-style request, not as a standalone CLI.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner


class PODProfilingRunner:
    """Profile model execution time for different mixed prefill+decode configs.

    This uses the existing `ModelRunner` and attention backend. Callers are
    expected to:
    - Construct `ModelRunner` normally (with chunked prefill / mixed chunk enabled).
    - Toggle POD / persistent attention via environment variables (see `profile_configs`).
    - Call `profile_one_seq_config` or `profile_configs` from the scheduler when a
      special `profile_seq_lens` request is received.
    """

    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        self.device = model_runner.device

    @torch.inference_mode()
    def profile_one_seq_config(
        self,
        n_p: int,  # new prefill tokens
        r_p: int,  # prefill prefix length
        r_d: int,  # decode KV length
        decode_bs: int,  # decode batch size
        num_runs: int = 10,
    ) -> float:
        """Profile one configuration and return avg execution time in milliseconds.

        Args:
            n_p: New prefill tokens (extend_input_len for prefill)
            r_p: Prefill prefix length
            r_d: Decode KV length (seq_len for decode requests)
            decode_bs: Decode batch size
            num_runs: Number of timed forward passes (after warmup)
        """
        assert decode_bs >= 0

        # Total batch size: 1 prefill + decode_bs decode
        total_bs = 1 + decode_bs

        # Create seq_lens: [n_p + r_p, r_d, r_d, ...]
        seq_lens = [n_p + r_p] + [r_d] * decode_bs
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64, device=self.device)
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)

        # extend_prefix_lens: [r_p, r_d, r_d, ...]
        extend_prefix_lens = [r_p] + [r_d] * decode_bs
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
        seq_lens_sum = int(seq_lens_tensor.sum().item())
        extend_num_tokens = int(sum(extend_lens))

        # Dummy token ids (only shapes matter)
        input_ids = torch.randint(
            0, 1000, (extend_num_tokens,), dtype=torch.int64, device=self.device
        )

        # Dummy req_pool_indices (0..total_bs-1)
        req_pool_indices = torch.arange(total_bs, dtype=torch.int64, device=self.device)

        # Dummy out_cache_loc: sequential layout for `extend_num_tokens` tokens.
        # In practice this should match the real KV allocator, but for performance
        # modeling purposes the layout detail does not affect timing much.
        out_cache_loc = torch.arange(
            extend_num_tokens, dtype=torch.int32, device=self.device
        )

        # Positions and logits buffer
        positions = torch.arange(
            extend_num_tokens, dtype=torch.int64, device=self.device
        )
        vocab_size = self.model_runner.model_config.vocab_size
        next_token_logits_buffer = torch.zeros(
            (extend_num_tokens, vocab_size),
            dtype=torch.float32,
            device=self.device,
        )
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
            next_token_logits_buffer=next_token_logits_buffer,
            positions=positions,
            extend_num_tokens=extend_num_tokens,
            extend_prefix_lens=extend_prefix_lens_tensor,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens=torch.tensor(
                extend_lens, dtype=torch.int32, device=self.device
            ),
            extend_seq_lens_cpu=extend_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=None,
            global_forward_mode=forward_mode,
            decoding_bs=decode_bs,
        )

        # Initialize attention backend metadata
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)

        # Warmup
        for _ in range(3):
            self.model_runner.forward(forward_batch, skip_attn_backend_init=True)
        torch.cuda.synchronize()

        # CUDA timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        times: List[float] = []
        for _ in tqdm(range(num_runs), desc="Profiling seq lengths", leave=False):
            start_event.record()
            self.model_runner.forward(forward_batch, skip_attn_backend_init=True)
            end_event.record()
            torch.cuda.synchronize()
            times.append(float(start_event.elapsed_time(end_event)))

        return float(np.mean(times))


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
    r_p_values = list(range(256, 4097, 256))
    r_d_values = list(range(256, 10001, 512))

    for n_p in n_p_values:
        for r_p in r_p_values:
            for r_d in r_d_values:
                max_decode_bs = (4096 * 100) // r_d
                if max_decode_bs <= 0:
                    continue

                candidates = [
                    1,
                    max_decode_bs // 4,
                    max_decode_bs // 2,
                    (3 * max_decode_bs) // 4,
                    max_decode_bs,
                ]
                decode_bs_values = sorted({bs for bs in candidates if bs > 0})

                for decode_bs in decode_bs_values:
                    configs.append((n_p, r_p, r_d, decode_bs))

    return configs


@torch.inference_mode()
def profile_configs(
    model_runner: ModelRunner,
    configs: Sequence[Tuple[int, int, int, int]],
    overlap: bool,
) -> pd.DataFrame:
    """Profile a set of (n_p, r_p, r_d, decode_bs) configs.

    This toggles POD / persistent attention via environment variables:
      - overlap=False: USE_POD=False, USE_PERSISTENT=False
      - overlap=True:  USE_POD=True,  USE_PERSISTENT=False

    Returns:
        A pandas DataFrame with columns: [n_p, r_p, r_d, decode_bs, time_ms]
        Feature computation (S_p, S_d, R_pd, term1-4) should be done in the training script.
    """
    # Configure attention backend mode
    if overlap:
        envs.SGLANG_FLASHINFER_USE_POD.set(True)
        envs.SGLANG_FLASHINFER_USE_PERSISTENT.set(False)
    else:
        envs.SGLANG_FLASHINFER_USE_POD.set(False)
        envs.SGLANG_FLASHINFER_USE_PERSISTENT.set(False)

    runner = PODProfilingRunner(model_runner)
    results = []

    for n_p, r_p, r_d, decode_bs in configs:
        time_ms = runner.profile_one_seq_config(n_p, r_p, r_d, decode_bs)
        results.append(
            {
                "n_p": n_p,
                "r_p": r_p,
                "r_d": r_d,
                "decode_bs": decode_bs,
                "time_ms": time_ms,
            }
        )

    return pd.DataFrame(results)


@torch.inference_mode()
def profile_default_grid(
    model_runner: ModelRunner,
    overlap: bool,
    max_configs: Optional[int] = None,
) -> pd.DataFrame:
    """Convenience helper that profiles the default grid of configs."""
    configs = generate_default_configs()
    if max_configs is not None:
        configs = configs[:max_configs]
    return profile_configs(model_runner, configs, overlap=overlap)
