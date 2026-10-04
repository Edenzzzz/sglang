from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import (
    get_int_env_var,
    is_flashinfer_available,
    is_sm100_supported,
    next_power_of_2,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

if envs.SGLANG_ENABLE_TORCH_COMPILE.get():
    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True


if is_flashinfer_available():
    import flashinfer
    from flashinfer import (
        BatchAttention,
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
    from flashinfer.cascade import merge_state

    fast_decode_plan = getattr(flashinfer, "fast_decode_plan", None)

    # BatchPOD is not part of every FlashInfer build. Keep it optional so a
    # persistent-only build does not fail at module import time.
    _batch_pod_wrapper = getattr(
        flashinfer, "BatchPODWithPagedKVCacheWrapper", None
    )
    _batch_pod_available = _batch_pod_wrapper is not None
    if _batch_pod_available:
        BatchPODWithPagedKVCacheWrapper = _batch_pod_wrapper
    else:

        class BatchPODWithPagedKVCacheWrapper:
            pass


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


def select_persistent_flipped_schedule(
    total_kv_tokens: int,
    batch_size: int,
    force_nonflipped: bool = False,
) -> Optional[bool]:
    """Select the persistent-kernel work ordering from server-level KV load."""
    if force_nonflipped or total_kv_tokens >= batch_size * 32768:
        return False
    # None preserves FlashInfer's optimized flipped default.
    return None


class MixedKernelGate:
    """Per-batch kernel + schedule choice for mixed-batch prefix attention.

    A k-NN lookup over a measured (BatchPrefill, persistent-flipped,
    persistent-nonflipped) kernel-time sweep decides, from CPU-side batch
    composition only, whether the persistent BatchAttention kernel should
    replace BatchPrefill for the cached-prefix part of a mixed batch, and
    which persistent work ordering to use. The persistent kernel is chosen
    only where all k nearest measured compositions agree it wins by the
    table's label margin, and only inside the table's measured support
    (max_nn_dist_sq guard); everything else keeps the BatchPrefill baseline,
    so serving can never regress beyond noise while decode-heavy batches
    keep 1.2-20x wins. k, the label margin, and the guard radius are
    properties of the table file, not this class.

    Tables are built by tasks/debug_kernel_server_perf/
    build_mixed_kernel_gate_table.py from gate_sweep3 measurements.
    """

    _cache: dict = {}

    def __init__(self, table_path: str):
        with open(table_path) as f:
            table = json.load(f)
        self.mu = np.asarray(table["mu"], dtype=np.float64)
        self.sd = np.asarray(table["sd"], dtype=np.float64)
        self.points = np.asarray(table["points"], dtype=np.float64)
        self.use_pa = np.asarray(table["use_pa"], dtype=bool)
        self.flip = np.asarray(table["flip"], dtype=bool)
        self.max_nn_dist_sq = float(table.get("max_nn_dist_sq", float("inf")))
        self.k = int(table.get("knn_k", 3))

    @classmethod
    def for_gqa(cls, gqa_group_size: int) -> Optional["MixedKernelGate"]:
        if gqa_group_size in cls._cache:
            return cls._cache[gqa_group_size]
        table_path = os.path.join(
            os.path.dirname(__file__), f"mixed_kernel_gate_h800_gqa{gqa_group_size}.json"
        )
        gate = None
        if os.path.exists(table_path):
            gate = cls(table_path)
        else:
            logger.warning(
                "No mixed-kernel gate table for gqa=%d (%s); "
                "using conservative analytic fallback",
                gqa_group_size,
                table_path,
            )
        cls._cache[gqa_group_size] = gate
        return gate

    def decide(
        self,
        decode_kv: int,
        prefill_work: int,
        prefill_tokens: int,
        decode_bs: int,
        kv_concentration: float,
    ):
        x = np.array(
            [
                np.log1p(decode_kv),
                np.log1p(prefill_work),
                np.log1p(prefill_tokens),
                np.log1p(decode_bs),
                3.0 * decode_kv / (decode_kv + prefill_work + 1),
                np.log1p(kv_concentration),
            ]
        )
        d = self.points - (x - self.mu) / self.sd
        dist = np.einsum("ij,ij->i", d, d)
        nn = np.argpartition(dist, self.k)[: self.k]
        if dist[nn].min() > self.max_nn_dist_sq:
            # Query sits outside the measured region; a k-NN answer here is an
            # extrapolation, not a measurement. Keep the BatchPrefill baseline
            # so an unswept composition can never cause a regression.
            return False, False
        use_pa = bool(self.use_pa[nn].all())
        flipped = bool(self.flip[nn].sum() * 2 >= self.k)
        return use_pa, flipped



from sglang.srt.layers.attention.mixed_dispatch import (
    plan_dispatch as md_plan_dispatch,
    plan_dispatch_prefix as md_plan_prefix,
)

class _MixedCompositionLog:
    """Append-only capture of every mixed batch's true composition.

    Enabled with SGLANG_LOG_MIXED_COMPOSITION=<path>. Unlike the scheduler's
    `Prefill batch` log lines this records the *attended* prefix per chunk
    (not the radix hit at admission, which is 0 for continued chunks) and the
    per-row decode KV vector, so a sweep can be resampled from the realized
    distribution instead of a Cartesian grid.
    """

    _inst = None
    _checked = False

    @classmethod
    def get(cls):
        if not cls._checked:
            cls._checked = True
            path = os.environ.get("SGLANG_LOG_MIXED_COMPOSITION", "")
            if path:
                try:
                    cls._inst = cls(path)
                except Exception as e:  # never break serving for telemetry
                    logger.warning(f"mixed-composition log disabled: {e}")
        return cls._inst

    def __init__(self, path):
        self.f = open(path, "a", buffering=1024 * 1024)
        self.n = 0

    def record_mode(self, mode, seq_lens_cpu, extend_lens):
        """Record any forward mode's shape, so the composition a *fused*
        draft+verify+prefill kernel would see can be reconstructed. Speculative
        decoding forces enable_mixed_chunk=False, so these arrive as separate
        launches and never appear in a MIXED batch today."""
        try:
            self.f.write(
                '{"t":%.3f,"mode":"%s","kv":%s,"q":%s}\n'
                % (time.time(), mode, list(seq_lens_cpu), list(extend_lens))
            )
            self.n += 1
            if self.n % 64 == 0:
                self.f.flush()
        except Exception:
            pass

    def record(self, dec_kvs, chunks, prefixes, use_pa, flipped):
        try:
            self.f.write(
                '{"t":%.3f,"d":%s,"q":%s,"p":%s,"pa":%d,"fl":%d}\n'
                % (time.time(), dec_kvs, chunks, prefixes,
                   int(bool(use_pa)), int(bool(flipped)))
            )
            self.n += 1
            if self.n % 64 == 0:
                self.f.flush()
        except Exception:
            pass


# Four work classes a mixed batch can contain once speculative decoding shares
# the batch with chunked prefill. `packed_qo_len = qo_len * gqa_group_size`
# decides the tile a row wants, so the classes are separated by qo_len:
#
#   LARGE_PREFILL  chunk with a long cached prefix   -> tile 128
#   SPEC_VERIFY    (k+1) candidate tokens per row     -> tile 64 when 16<(k+1)*r<=64
#   MIXED_DECODE   small/fresh-prefix chunk           -> tile 128, our LPT regime
#   SPEC_DRAFT     one drafted token per row          -> tile 16, smallest work
#
# Which kernel runs which class is decided per batch by mixed_dispatch, not by
# a fixed table: measured on the captured sets, the best strategy flips with the
# number of q heads per rank and with how the prefill and decode populations
# balance. See plan_4mode_dispatch below.
CLASS_LARGE_PREFILL = "large_prefill"
CLASS_SPEC_VERIFY = "spec_verify"
CLASS_MIXED_DECODE = "mixed_decode"
CLASS_SPEC_DRAFT = "spec_draft"

LARGE_PREFILL_MIN_PREFIX = 8192


def classify_mixed_batch(extend_seq_lens_cpu, extend_prefix_lens_cpu,
                         gqa_group_size, draft_token_num=1):
    """Split a batch's rows into the four work classes.

    `draft_token_num` is the speculative candidate count (k+1); rows whose
    qo_len equals it are verify rows, qo_len==1 rows are decode/draft steps.
    Returns {class: [(qo_len, kv_len), ...]} plus the tile each class wants.
    """
    e = np.asarray(extend_seq_lens_cpu, dtype=np.int64)
    c = np.asarray(extend_prefix_lens_cpu, dtype=np.int64)
    out = {CLASS_LARGE_PREFILL: [], CLASS_SPEC_VERIFY: [],
           CLASS_MIXED_DECODE: [], CLASS_SPEC_DRAFT: []}
    for qo, pfx in zip(e.tolist(), c.tolist()):
        if qo <= 1:
            out[CLASS_SPEC_DRAFT].append((qo, pfx))
        elif draft_token_num > 1 and qo == draft_token_num:
            out[CLASS_SPEC_VERIFY].append((qo, pfx))
        elif pfx >= LARGE_PREFILL_MIN_PREFIX:
            out[CLASS_LARGE_PREFILL].append((qo, pfx))
        else:
            out[CLASS_MIXED_DECODE].append((qo, pfx))
    tiles = {}
    for k, rows in out.items():
        if not rows:
            continue
        p = max(qo * gqa_group_size for qo, _ in rows)
        tiles[k] = 128 if p > 64 else (64 if p > 16 else 16)
    return out, tiles


def plan_4mode_dispatch(extend_seq_lens_cpu, extend_prefix_lens_cpu,
                        num_qo_heads, num_kv_heads, draft_token_num=1,
                        num_sm=None):
    """Choose how to execute a fused batch.

    Returns the dict from mixed_dispatch.plan_dispatch:
      mode      'fused' (persistent only) or 'split' (disjoint SM sets)
      sm_fa3    SMs given to FA3, 0 when fused
      sm_pers   SMs for the persistent kernel
      kv_split  FA3's KV-split factor
      fa3_mask  per-row mask selecting the FA3 class, None when fused

    Routing is on packed_qo_len and kv_tiles -- tile occupancy, not KV length
    alone, which is what an earlier version used and which measured 14x slower
    by sending short verify rows to FA3.
    """
    gqa = max(num_qo_heads // max(num_kv_heads, 1), 1)
    if num_sm is None:
        num_sm = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count
    return md_plan_dispatch(
        np.asarray(extend_seq_lens_cpu, dtype=np.int64),
        np.asarray(extend_prefix_lens_cpu, dtype=np.int64),
        gqa, num_kv_heads, num_sm=num_sm,
    )


def select_mixed_attention_kernel(
    extend_seq_lens_cpu,
    extend_prefix_lens_cpu,
    gqa_group_size: int,
):
    """Choose the kernel for a mixed batch's cached-prefix attention.

    Returns (use_persistent, flipped_schedule). Uses only CPU-side metadata.
    The H800 sweep (tasks/debug_kernel_server_perf.md, 2026-07-30) shows the
    persistent kernel matches or beats BatchPrefill except when the decode
    population is too small to overlap with a compute-bound prefill chunk, or
    the whole batch is tiny enough that fixed kernel overhead dominates.
    """
    mode = envs.SGLANG_MIXED_ATTENTION_KERNEL.get()
    if mode == "persistent":
        return True, None
    if mode == "batch_prefill":
        return False, None

    e = np.asarray(extend_seq_lens_cpu, dtype=np.int64)
    c = np.asarray(extend_prefix_lens_cpu, dtype=np.int64)
    is_decode = e <= 1
    dec_kv = c[is_decode]
    decode_bs = int(is_decode.sum())
    decode_kv = int(dec_kv.sum())
    prefill_work = int((e[~is_decode] * c[~is_decode]).sum())
    prefill_tokens = int(e[~is_decode].sum())
    # KV-weighted mean row length: separates a few long-context rows (which
    # leave most SMs idle under BatchPrefill) from many short ones.
    kv_conc = float((dec_kv * dec_kv).sum() / decode_kv) if decode_kv > 0 else 0.0

    gate = MixedKernelGate.for_gqa(gqa_group_size)
    if gate is None:
        # Conservative analytic fallback: persistent only where a small number
        # of long-context rows would starve BatchPrefill of parallelism.
        use_pa = kv_conc >= 16384 and decode_kv * 20 >= prefill_work
        flipped = False
    else:
        use_pa, flipped = gate.decide(
            decode_kv, prefill_work, prefill_tokens, decode_bs, kv_conc
        )
    cap = _MixedCompositionLog.get()
    if cap is not None:
        cap.record(
            dec_kv.tolist(),
            e[~is_decode].tolist(),
            c[~is_decode].tolist(),
            use_pa,
            flipped,
        )
    return use_pa, flipped


@dataclass
class MultiItemScoringParams:
    """Parameters for multi-item scoring in attention computation.

    Used when processing sequences with multiple items separated by delimiters,
    where each item needs specific attention patterns that respect item boundaries.

    Attributes:
        prefix_len_ptr: A uint32 1D tensor indicating the prefix length of each prompt.
                       The tensor size is equal to the batch size.
        token_pos_in_items_ptr: A uint16 1D tensor indicating the token position of each item
                               starting from 0 (delimiter) for each item. For batch size > 1,
                               sequences are concatenated with zero padding to ensure same length.
        token_pos_in_items_len: Zero padding length for token_pos_in_items_ptr to handle
                               batch_size > 1 case. Defines the padded length for each sequence.
        max_item_len_ptr: A uint16 tensor containing the max token length of all items
                         for each prompt in the batch.

    """

    prefix_len_ptr: Optional[torch.Tensor] = None
    token_pos_in_items_ptr: Optional[torch.Tensor] = None
    token_pos_in_items_len: int = 0
    max_item_len_ptr: Optional[torch.Tensor] = None

    def is_enabled(self) -> bool:
        """Check if multi-item scoring is enabled."""
        return self.prefix_len_ptr is not None


@dataclass
class DecodeMetadata:
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]


@dataclass
class PrefillMetadata:
    prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper]
    use_ragged: bool
    extend_no_prefix: bool
    multi_item_params: Optional[MultiItemScoringParams] = None


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None

# Use as a fast path to override the indptr in flashinfer's plan function
# This is used to remove some host-to-device copy overhead.
global_override_indptr_cpu = None


class FlashInferAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
        init_new_workspace: bool = False,
    ):
        super().__init__()

        # Store multi-item scoring delimiter for efficient access
        self.multi_item_scoring_delimiter = (
            model_runner.server_args.multi_item_scoring_delimiter
        )

        # Parse constants
        self.decode_use_tensor_cores = should_use_tensor_core(
            kv_cache_dtype=model_runner.kv_cache_dtype,
            num_attention_heads=model_runner.model_config.num_attention_heads
            // get_attention_tp_size(),
            num_kv_heads=model_runner.model_config.get_num_kv_heads(
                get_attention_tp_size()
            ),
        )
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self._mixed_gate_flipped = None
        # Kept for the fused FA3+FA2 planner, which needs the geometry and
        # dtypes that this backend otherwise only reads off a layer.
        self.head_dim = model_runner.model_config.head_dim
        self.q_data_type = model_runner.dtype
        self.kv_data_type = model_runner.kv_cache_dtype

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        if model_runner.sliding_window_size is not None:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW
        elif model_runner.model_config.is_encoder_decoder:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.CROSS_ATTENTION
        else:
            self.num_wrappers = 1
            self.dispatch_reason = None

        # Qwen2/Qwen3 models require higher flashinfer workspace size
        if (
            "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
        ):
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(512 * 1024 * 1024)

        # When deterministic inference is enabled, tensor cores should be used for decode
        # Also set split tile sizes for prefill and decode from environment variables, and disable kv split for cuda graph
        # More information can be found here: https://github.com/flashinfer-ai/flashinfer/pull/1675
        self.enable_deterministic = (
            model_runner.server_args.enable_deterministic_inference
        )
        self.prefill_split_tile_size = None
        self.decode_split_tile_size = None
        self.disable_cuda_graph_kv_split = False
        if self.enable_deterministic:
            self.decode_use_tensor_cores = True
            self.prefill_split_tile_size = get_int_env_var(
                "SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096
            )
            self.decode_split_tile_size = get_int_env_var(
                "SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE", 2048
            )
            self.disable_cuda_graph_kv_split = True
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(2048 * 1024 * 1024)

        # Set mixed batch backend
        self.use_persistent = envs.SGLANG_FLASHINFER_USE_PERSISTENT.get()
        self.use_pod = envs.SGLANG_FLASHINFER_USE_POD.get()
        self.no_flipped_schedule = envs.SGLANG_FLASHINFER_NO_FLIPPED_SCHEDULE.get()
        if self.use_pod and not _batch_pod_available:
            raise RuntimeError(
                "SGLANG_FLASHINFER_USE_POD requires a FlashInfer build with "
                "BatchPODWithPagedKVCacheWrapper"
            )
        if self.use_pod and self.use_persistent:
            raise ValueError(
                "SGLANG_FLASHINFER_USE_POD and SGLANG_FLASHINFER_USE_PERSISTENT cannot both be True"
            )

        # Allocate buffers
        global global_workspace_buffer
        if global_workspace_buffer is None:
            # different from flashinfer zero_init_global_workspace_buffer
            global_workspace_size = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()
            global_workspace_buffer = torch.empty(
                global_workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        if init_new_workspace:
            self.workspace_buffer = torch.empty(
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                dtype=torch.uint8,
                device=model_runner.device,
            )
        else:
            self.workspace_buffer = global_workspace_buffer
        max_bs = model_runner.req_to_token_pool.size
        if kv_indptr_buf is None:
            self.kv_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(self.num_wrappers)
            ]
        else:
            assert self.num_wrappers == 1
            self.kv_indptr = [kv_indptr_buf]

        if kv_last_page_len_buf is None:
            self.kv_last_page_len = torch.ones(
                (max_bs,), dtype=torch.int32, device=model_runner.device
            )
        else:
            assert self.num_wrappers == 1
            self.kv_last_page_len = kv_last_page_len_buf

        if not self.skip_prefill:
            self.qo_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(
                    self.num_wrappers if not self.use_pod else self.num_wrappers * 2
                )
            ]

        fmha_backend = "auto"
        if is_sm100_supported():
            # Disable CUTLASS backend when piecewise cuda graph is enabled
            # due to TMA descriptor initialization issues on B200
            if model_runner.server_args.enable_piecewise_cuda_graph:
                logger.warning(
                    "CUTLASS backend is disabled when piecewise cuda graph is enabled "
                    "due to TMA descriptor initialization issues on B200. "
                    "Using auto backend instead for stability."
                )
            else:
                fmha_backend = "cutlass"
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=fmha_backend
        )

        # Two wrappers: one for sliding window attention and one for full attention.
        # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        self.prefill_wrappers_paged = []
        self.prefill_wrappers_verify = []
        self.prefill_wrappers_mixed = []
        self.decode_wrappers = []
        self.pod_wrappers = []
        if self.use_persistent and not skip_prefill:
            # Both work-ordering variants of the persistent module must be
            # compiled before serving: the gate picks the schedule per batch,
            # and a cold-cache JIT build mid-replay stalls every running
            # request for ~20 seconds.
            from flashinfer.attention import prewarm_holistic_attention_modules

            prewarm_holistic_attention_modules(
                model_runner.dtype,
                model_runner.kv_cache_dtype,
                torch.int32,
                model_runner.model_config.head_dim,
                model_runner.model_config.head_dim,
            )
        for _ in range(self.num_wrappers):
            if not skip_prefill:
                self.prefill_wrappers_mixed.append(
                    BatchAttention(
                        self.workspace_buffer,
                        "NHD",
                    )
                )
                self.prefill_wrappers_paged.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend="fa2",
                    )
                )
                self.prefill_wrappers_verify.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                    )
                )
                if self.use_pod:
                    self.pod_wrappers.append(
                        BatchPODWithPagedKVCacheWrapper(
                            self.workspace_buffer,
                            "NHD",
                        )
                    )
            self.decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.decode_use_tensor_cores,
                )
            )

        # Create indices updater
        if not skip_prefill:
            self.indices_updater_prefill = FlashInferIndicesUpdaterPrefill(
                model_runner, self
            )  # for verify
        self.indices_updater_decode = FlashInferIndicesUpdaterDecode(model_runner, self)

        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None

        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}  # For verify
        self.draft_extend_cuda_graph_metadata = {}  # For draft extend
        self.mixed_cuda_graph_metadata = {}  # For mixed batch persistent kernel

    def _process_multi_item_scoring(
        self, forward_batch: ForwardBatch
    ) -> MultiItemScoringParams:
        """Process multi-item scoring tensors for FlashInfer attention.

        This method handles sequences containing multiple "items" separated by delimiter tokens,
        where each item needs specific attention patterns that respect item boundaries.

        The method produces four key tensors for FlashInfer:
        - prefix_len_ptr: uint32 tensor with prefix length for each prompt in batch
        - token_pos_in_items_ptr: uint16 tensor with token positions starting from 0 at delimiters
        - token_pos_in_items_len: padding length for batch processing
        - max_item_len_ptr: uint16 tensor with max item length for each prompt

        Args:
            forward_batch: The forward batch containing input sequences and delimiter info

        Returns:
            MultiItemScoringParams: The processed multi-item scoring parameters

        Examples:
            Following FlashInfer definition: for 3 items of length 3, 2, 4 respectively:
            token_pos_in_items_ptr = [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0]

            Case 1: Single sequence
            Text: "What is the capital of France? <delim> London <delim> Paris <delim> Berlin <delim>"
            Tokens: [What, is, the, capital, of, France, ?, <delim>, London, <delim>, Paris, <delim>, Berlin, <delim>]
            Indices: [ 0,   1,  2,   3,      4,  5,     6,   7,     8,      9,     10,    11,    12,     13]
            - prefix_len_ptr: [7] (query length before first delimiter)
            - token_pos_in_items_ptr: [0, 1, 0, 1, 0, 1, 0] (delim=0, London=1, delim=0, Paris=1, delim=0, Berlin=1, delim=0)
            - token_pos_in_items_len: 7 (actual length)
            - max_item_len_ptr: [1] (max item length is 1 token - all options are single tokens)

            Case 2: Batch processing (batch_size=2)
            Sequence 1: 2 items of length 2, 1 → [0, 1, 2, 0, 1, 0] (6 elements)
            Sequence 2: 3 items of length 1, 3, 2 → [0, 1, 0, 1, 2, 3, 0, 1, 2, 0] (10 elements)
            After padding both to length 10:
            - token_pos_in_items_ptr: [0, 1, 2, 0, 1, 0, 0, 0, 0, 0,    0, 1, 0, 1, 2, 3, 0, 1, 2, 0]
            - token_pos_in_items_len: 10 (padded length for batch processing)
            - max_item_len_ptr: [2, 3] (max lengths per sequence)
        """

        delimiter = self.multi_item_scoring_delimiter
        if delimiter is None or forward_batch.forward_mode == ForwardMode.DECODE:
            return MultiItemScoringParams()

        delimiter_mask = forward_batch.input_ids == delimiter
        prefix_cache_lens = getattr(forward_batch, "extend_prefix_lens", None)
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        prefix_len_ptr, token_pos_in_items_ptr = [], []
        token_pos_in_items_len = 0

        # If no extend_seq_lens, treat whole batch as one sequence
        if extend_seq_lens is None or len(extend_seq_lens) <= 1:
            extend_seq_lens = [forward_batch.input_ids.size(0)]

        seq_start = 0
        for i, seq_len in enumerate(extend_seq_lens):
            seq_end = seq_start + seq_len
            mask = delimiter_mask[seq_start:seq_end]
            pos = forward_batch.positions[seq_start:seq_end]
            delimiter_indices = torch.nonzero(mask, as_tuple=True)[0]

            if len(delimiter_indices) > 0:
                first_delim = delimiter_indices[0]
                # Prefix length: store as scalar
                prefix_len = first_delim + (
                    prefix_cache_lens[i] if prefix_cache_lens is not None else 0
                )
                prefix_len_ptr.append(
                    prefix_len.item() if torch.is_tensor(prefix_len) else prefix_len
                )

                # Compute relative positions within items after delimiters
                diff = pos[first_delim:] - torch.cummax(mask[first_delim:], 0)[1]
                token_pos = (diff - pos[first_delim]).to(torch.uint16)
                token_pos_in_items_ptr.append(token_pos)

                # Update forward_batch positions in-place
                pos[first_delim:] = diff - 1
                forward_batch.positions[seq_start:seq_end] = pos

            seq_start = seq_end

        # Pad token_pos_in_items_ptr for batch processing
        if token_pos_in_items_ptr:
            token_pos_in_items_len = max(t.numel() for t in token_pos_in_items_ptr)
            device = forward_batch.input_ids.device
            token_pos_in_items_ptr = [
                torch.cat(
                    [
                        t,
                        torch.zeros(
                            token_pos_in_items_len - t.numel(),
                            dtype=torch.uint16,
                            device=device,
                        ),
                    ]
                )
                for t in token_pos_in_items_ptr
            ]

        if not prefix_len_ptr or not token_pos_in_items_ptr:
            return MultiItemScoringParams()

        # Build final params
        device = forward_batch.input_ids.device
        return MultiItemScoringParams(
            prefix_len_ptr=torch.tensor(
                prefix_len_ptr, dtype=torch.uint32, device=device
            ),
            token_pos_in_items_ptr=torch.cat(token_pos_in_items_ptr, dim=0),
            token_pos_in_items_len=token_pos_in_items_len & 0xFFFFFFFF,
            max_item_len_ptr=torch.stack(
                [
                    t.to(torch.int32).max().to(torch.uint16)
                    for t in token_pos_in_items_ptr
                ],
                dim=0,
            ),
        )

    def _count_forward(self, kind: str):
        """Tally forward passes by attention path so the mixed-batch share of
        server time can be read off a run instead of inferred from
        interval-gated scheduler logs."""
        stats = getattr(self, "_forward_stats", None)
        if stats is None:
            stats = self._forward_stats = {}
        stats[kind] = stats.get(kind, 0) + 1
        total = sum(stats.values())
        every = int(os.environ.get("SGLANG_FORWARD_MIX_EVERY", "2000"))
        if every > 0 and total % every == 0:
            logger.info(
                "forward mix (n=%d): %s",
                total,
                " ".join(f"{k}={v}({v/total*100:.1f}%)" for k, v in sorted(stats.items())),
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # A fused plan is valid for exactly one step. Clear it up front so a
        # stale plan can never be applied to a different batch.
        self._fused_plan = None
        # Update attn env var as it could be changed by profiler
        self.use_persistent = envs.SGLANG_FLASHINFER_USE_PERSISTENT.get()
        self.use_pod = envs.SGLANG_FLASHINFER_USE_POD.get()
        if self.use_pod and self.use_persistent:
            raise ValueError(
                "SGLANG_FLASHINFER_USE_POD and SGLANG_FLASHINFER_USE_PERSISTENT cannot both be True"
            )

        cap = _MixedCompositionLog.get()
        if cap is not None:
            try:
                sl = forward_batch.seq_lens_cpu
                sl = sl.tolist() if sl is not None else []
                el = getattr(forward_batch, "extend_seq_lens_cpu", None) or []
                cap.record_mode(forward_batch.forward_mode.name,
                                sl, list(el))
            except Exception:
                pass
        if forward_batch.forward_mode.is_decode_or_idle():
            self._count_forward("decode")
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                decode_wrappers=self.decode_wrappers,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
                fixed_split_size=self.decode_split_tile_size,
                disable_split_kv=False,
            )
            self.forward_metadata = DecodeMetadata(self.decode_wrappers)
        elif forward_batch.forward_mode.is_draft_extend():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_wrappers_paged,
                use_ragged=False,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_paged, False, False
            )
        elif forward_batch.forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_wrappers_verify,
                use_ragged=False,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_verify, False, False
            )
        else:
            # prefill/mixed batch
            prefix_lens = forward_batch.extend_prefix_lens
            # wrappers = self.prefill_wrappers_paged
            wrappers = []

            if self.use_pod and forward_batch.forward_mode.is_mixed():
                wrappers = self.pod_wrappers
            elif self.use_persistent and forward_batch.forward_mode.is_mixed():
                # Per-batch kernel choice: the persistent kernel only pays off
                # when the decode population supplies enough overlap work, so
                # fall back to BatchPrefill outside its winning region. Mixed
                # CUDA graphs capture BatchAttention, so the gate must not
                # switch kernels when they are enabled.
                if getattr(self, "mixed_cuda_graph_max_bs", 0) > 0:
                    use_pa, self._mixed_gate_flipped = True, None
                else:
                    use_pa, self._mixed_gate_flipped = select_mixed_attention_kernel(
                        forward_batch.extend_seq_lens_cpu,
                        forward_batch.extend_prefix_lens_cpu,
                        self.num_qo_heads // self.num_kv_heads,
                    )
                self._count_forward("mixed_persistent" if use_pa else "mixed_batch_prefill")
                if use_pa:
                    wrappers = self.prefill_wrappers_mixed
                else:
                    wrappers = self.prefill_wrappers_paged
                # Plan the fused FA3+FA2 split for this step, if enabled. Both
                # roles are planned once here and reused across layers, like
                # every other wrapper; only the launch is per layer. Returns
                # None whenever the batch does not satisfy the kernel's layout
                # contract, in which case persistent_attn_forward runs the
                # ordinary path.
                self._fused_want = bool(use_pa)
            else:
                self._count_forward(
                    "mixed_other"
                    if forward_batch.forward_mode.is_mixed()
                    else "pure_prefill"
                )
                wrappers = self.prefill_wrappers_paged
            use_ragged = True

            # Disable ragged wrapper and ensure prefix handling for multimodal and multi-item scoring
            if self.is_multimodal or self.multi_item_scoring_delimiter is not None:
                if isinstance(wrappers[0], BatchAttention) or isinstance(
                    wrappers[0], BatchPODWithPagedKVCacheWrapper
                ):
                    raise NotImplementedError(
                        "Multi-item scoring is not supported for persistent attention yet"
                    )
                # use_ragged = False: Multi-item scoring requires the paged wrapper because:
                # 1. Ragged wrapper doesn't support the specialized multi-item parameters
                #    (prefix_len_ptr, token_pos_in_items_ptr, etc.)
                # 2. Paged wrapper provides better control over attention masking needed
                #    for respecting item boundaries in multi-item sequences
                # 3. Custom masking logic conflicts with ragged wrapper's assumptions
                use_ragged = False
                extend_no_prefix = False
            elif (
                # forward_batch.forward_mode.is_mixed()
                self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW
                or self.enable_deterministic
            ):  # TODO(Wenxuan): support sliding window and deterministic in flashinfer
                if isinstance(wrappers[0], BatchAttention) or isinstance(
                    wrappers[0], BatchPODWithPagedKVCacheWrapper
                ):
                    raise NotImplementedError(
                        "Sliding window and deterministic are not supported for persistent attention yet"
                    )
                wrappers = self.prefill_wrappers_mixed
                use_ragged = True
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            elif self.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
                if isinstance(wrappers[0], BatchAttention):
                    raise NotImplementedError(
                        "Cross attention is not supported for persistent attention yet"
                    )
            else:
                use_ragged = not self.enable_deterministic
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            if forward_batch.seq_lens_sum - forward_batch.prefix_lens_sum <= 1:
                # NOTE(Wenxuan): on some H100 machines this throws fail to initialize TMA descriptor error. See https://github.com/flashinfer-ai/flashinfer/issues/2150
                use_ragged = False
                extend_no_prefix = False

            # Process multi-item scoring in attention backend instead of ForwardBatch
            multi_item_params = MultiItemScoringParams()
            if self.multi_item_scoring_delimiter is not None:
                # Use new backend-specific implementation
                multi_item_params = self._process_multi_item_scoring(forward_batch)

            # Use POD for mixed batches if enabled
            if self.use_pod and forward_batch.forward_mode.is_mixed():
                assert (
                    self.dispatch_reason == None
                ), "POD only supports regular attention"
                if multi_item_params.is_enabled():
                    raise NotImplementedError(
                        "Multi-item scoring is not supported for POD attention yet"
                    )
                num_decode = forward_batch.decoding_bs
                num_prefill = forward_batch.batch_size - num_decode

                self.indices_updater_prefill.update_pod(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_cpu,
                    forward_batch.seq_lens_sum,
                    prefix_lens,
                    self.prefill_wrapper_ragged,
                    wrappers[0],
                    use_ragged=use_ragged,
                    encoder_lens=forward_batch.encoder_lens,
                    page_size=forward_batch.token_to_kv_pool.page_size,
                    num_prefill=num_prefill,
                    num_decode=num_decode,
                    prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
                )
            else:
                # Both persistent and batch prefill included
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_cpu,
                    forward_batch.seq_lens_sum,
                    prefix_lens,
                    prefill_wrappers=wrappers,
                    use_ragged=use_ragged,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=None,
                    fixed_split_size=self.prefill_split_tile_size,
                    multi_item_params=multi_item_params,
                    prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
                )
            # Planned here, not before the update above: the updater has just
            # built the batch's page index array on the GPU, and reusing it is
            # far cheaper than rebuilding it with a per-request Python loop.
            if getattr(self, "_fused_want", False):
                self._fused_plan = self._plan_fused_split(forward_batch, wrappers)
                self._fused_want = False
            try:
                self.forward_metadata = PrefillMetadata(
                    wrappers,
                    use_ragged,
                    extend_no_prefix,
                    multi_item_params,
                )
            except:
                torch.distributed.breakpoint()

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        if kv_indices_buf is None:
            cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len,),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = kv_indices_buf

        self.cuda_graph_kv_indices = [cuda_graph_kv_indices] + [
            cuda_graph_kv_indices.clone() for _ in range(self.num_wrappers - 1)
        ]

        # Ensure tensors are properly allocated
        for i in range(self.num_wrappers):
            # Force allocation by performing a small operation
            if len(self.cuda_graph_kv_indices[i]) > 0:
                self.cuda_graph_kv_indices[i][0] = 0

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device="cuda",
            )
            self.cuda_graph_qk_indptr = [x.clone() for x in self.kv_indptr]
            self.cuda_graph_qo_indptr = [x.clone() for x in self.kv_indptr]

    def init_mixed_cuda_graph_state(
        self, max_mixed_bs: int, max_total_kv_tokens: int
    ):
        """Initialize state for mixed batch (persistent kernel) CUDA graphs."""
        self.mixed_cuda_graph_max_bs = max_mixed_bs
        kv_indices_size = max(max_total_kv_tokens, max_mixed_bs * 256)
        self.mixed_cuda_graph_kv_indices = [
            torch.zeros(kv_indices_size, dtype=torch.int32, device="cuda")
            for _ in range(self.num_wrappers)
        ]
        self.mixed_cuda_graph_kv_len_arr = [
            torch.zeros(max_mixed_bs, dtype=torch.int32, device="cuda")
            for _ in range(self.num_wrappers)
        ]
        # Allocate separate kv_indptr/qo_indptr for mixed CG (don't resize shared ones).
        self.mixed_cuda_graph_kv_indptr = [
            torch.zeros(max_mixed_bs + 1, dtype=torch.int32, device="cuda")
            for _ in range(self.num_wrappers)
        ]
        self.mixed_cuda_graph_qo_indptr = [
            torch.zeros(max_mixed_bs + 1, dtype=torch.int32, device="cuda")
            for _ in range(max(len(self.qo_indptr), 1))
        ]

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        if forward_mode.is_decode_or_idle():
            decode_wrappers = []
            for i in range(self.num_wrappers):
                decode_wrappers.append(
                    # Always use BatchDecodeWithPagedKVCacheWrapper for decode CUDA graphs,
                    # even when use_persistent=True. Persistent kernel only activates on
                    # mixed batches, not decode. Using BatchAttention here was unnecessarily
                    # heavy and prevented fast_decode_plan optimization.
                    BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=self.decode_use_tensor_cores,
                        paged_kv_indptr_buffer=self.kv_indptr[i][: num_tokens + 1],
                        paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                        paged_kv_last_page_len_buffer=self.kv_last_page_len[
                            :num_tokens
                        ],
                    )
                )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_decode.update(
                req_pool_indices,
                seq_lens,
                seq_lens.cpu(),  # may add a little overhead in capture stage
                seq_lens_sum,
                decode_wrappers=decode_wrappers,
                encoder_lens=encoder_lens,
                spec_info=spec_info,
                fixed_split_size=None,
                disable_split_kv=self.disable_cuda_graph_kv_split,
            )
            self.decode_cuda_graph_metadata[bs] = decode_wrappers
            self.forward_metadata = DecodeMetadata(decode_wrappers)
            # Enable fast_decode_plan for all configs (including persistent,
            # since decode steps now always use BatchDecodeWithPagedKVCacheWrapper)
            if fast_decode_plan is not None:
                for i in range(self.num_wrappers):
                    decode_wrappers[i].plan = partial(
                        fast_decode_plan, decode_wrappers[i]
                    )

        elif forward_mode.is_target_verify():
            prefill_wrappers = []
            for i in range(self.num_wrappers):
                prefill_wrappers.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                        paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                        paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                        paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                        custom_mask_buf=self.cuda_graph_custom_mask,
                        mask_indptr_buf=self.cuda_graph_qk_indptr[i][: bs + 1],
                    )
                )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens.cpu(),  # may add a little overhead in capture stage
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=prefill_wrappers,
                use_ragged=False,
                encoder_lens=encoder_lens,
                spec_info=spec_info,
            )
            self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
        elif forward_mode.is_draft_extend():
            prefill_wrappers = []
            for i in range(self.num_wrappers):
                prefill_wrappers.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend="fa2",
                        use_cuda_graph=True,
                        qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                        paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                        paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                        paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                    )
                )

            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens.cpu(),  # may add a little overhead in capture stage
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=prefill_wrappers,
                use_ragged=False,
                encoder_lens=encoder_lens,
                spec_info=spec_info,
            )
            self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
        elif forward_mode.is_mixed():
            mixed_wrappers = []
            for i in range(self.num_wrappers):
                mixed_wrappers.append(
                    BatchAttention(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        paged_kv_indptr_buffer=self.mixed_cuda_graph_kv_indptr[i][: bs + 1],
                        paged_kv_indices_buffer=self.mixed_cuda_graph_kv_indices[i],
                        kv_len_arr_buffer=self.mixed_cuda_graph_kv_len_arr[i][:bs],
                    )
                )
            seq_lens_sum = seq_lens.sum().item()
            prefix_lens = torch.zeros(bs, dtype=torch.int32, device=req_pool_indices.device)
            # Temporarily swap indptr buffers to larger mixed CG buffers
            saved_kv_indptr = self.indices_updater_prefill.kv_indptr
            saved_qo_indptr = self.indices_updater_prefill.qo_indptr
            self.indices_updater_prefill.kv_indptr = self.mixed_cuda_graph_kv_indptr
            self.indices_updater_prefill.qo_indptr = self.mixed_cuda_graph_qo_indptr
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens.cpu(),
                seq_lens_sum,
                prefix_lens=prefix_lens,
                prefill_wrappers=mixed_wrappers,
                use_ragged=False,
                encoder_lens=encoder_lens,
                spec_info=None,
            )
            self.indices_updater_prefill.kv_indptr = saved_kv_indptr
            self.indices_updater_prefill.qo_indptr = saved_qo_indptr
            self.mixed_cuda_graph_metadata[bs] = mixed_wrappers
            self.forward_metadata = PrefillMetadata(
                mixed_wrappers, use_ragged=False, extend_no_prefix=False
            )
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        prefix_lens: Optional[torch.Tensor] = None,
        num_real_seq: int = 0,
    ):
        if forward_mode.is_decode_or_idle():
            # Graph-replayed decodes never reach init_forward_metadata, so they
            # must be tallied here or the mixed-batch share of a run looks far
            # larger than it is.
            self._count_forward("decode_cuda_graph")
            self.indices_updater_decode.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                decode_wrappers=self.decode_cuda_graph_metadata[bs],
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
                fixed_split_size=None,
                disable_split_kv=self.disable_cuda_graph_kv_split,
            )
        elif forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        elif forward_mode.is_draft_extend():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        elif forward_mode.is_mixed():
            # Swap to larger mixed CG indptr buffers for the update
            saved_kv = self.indices_updater_prefill.kv_indptr
            saved_qo = self.indices_updater_prefill.qo_indptr
            self.indices_updater_prefill.kv_indptr = self.mixed_cuda_graph_kv_indptr
            self.indices_updater_prefill.qo_indptr = self.mixed_cuda_graph_qo_indptr
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=prefix_lens[:bs] if prefix_lens is not None else None,
                prefill_wrappers=self.mixed_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=None,
            )
            self.indices_updater_prefill.kv_indptr = saved_kv
            self.indices_updater_prefill.qo_indptr = saved_qo
        else:
            raise ValueError("Invalid forward mode")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def persistent_attn_forward(
        self,
        batch_attn_wrapper: BatchAttention,
        forward_batch: ForwardBatch,
        q: torch.Tensor,
        layer: RadixAttention,
        logits_soft_cap: float,
    ):
        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        num_tokens = q.shape[0]
        out_buf = getattr(self, "_persistent_out_buf", None)
        if (
            out_buf is None
            or out_buf.shape[0] < num_tokens
            or out_buf.shape[1:] != q.shape[1:]
            or out_buf.dtype != q.dtype
        ):
            self._persistent_out_buf = torch.empty(
                num_tokens, q.shape[1], q.shape[2], dtype=q.dtype, device=q.device
            )
            self._persistent_lse_buf = torch.empty(
                num_tokens, q.shape[1], dtype=torch.float32, device=q.device
            )
            out_buf = self._persistent_out_buf
        fused = getattr(self, "_fused_plan", None)
        if fused is not None:
            try:
                return self._fused_attn_forward(
                    fused, batch_attn_wrapper, forward_batch, q, layer,
                    out_buf[:num_tokens], num_tokens,
                )
            except Exception as e:
                # A failure here must not take the server down, but it must be
                # loud: silently serving the non-fused path would make an A/B
                # measure the wrong kernel, which is exactly the error that
                # wasted a day of server benchmarking.
                logger.error("fused FA3+FA2 launch failed, using persistent: %s", e)
                self._fused_plan = None
        o, lse = batch_attn_wrapper.run(
            q,
            forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            out=out_buf[:num_tokens],
            lse=self._persistent_lse_buf[:num_tokens],
            logits_soft_cap=logits_soft_cap,
            k_scale=layer.k_scale,
            v_scale=layer.v_scale,
        )
        return o, lse

    def _fused_attn_forward(
        self, fused, batch_attn_wrapper, forward_batch, q, layer, out, num_tokens
    ):
        """Launch the fused kernel for this layer.

        The FA3 and persistent plans are built once per step by
        _plan_fused_split and reused across layers, like every other wrapper
        here; only the launch is per layer.
        """
        import os

        import flashinfer

        kv_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        # get_kv_buffer returns either a fused [N,2,...] tensor or a (k, v)
        # tuple depending on the pool layout. Assuming the tensor form made every
        # launch raise "tuple indices must be integers or slices" and fall back,
        # so the kernel never actually ran on the serving path even though the
        # plan succeeded.
        if isinstance(kv_cache, (tuple, list)):
            k_cache, v_cache = kv_cache[0], kv_cache[1]
        else:
            k_cache, v_cache = kv_cache[:, 0], kv_cache[:, 1]
        # Counted here, at the launch, not at plan time. Counting the plan made
        # the forward mix unable to distinguish "the contract accepted this
        # batch" from "the kernel ran", and a later failure in building the plan
        # dict then showed up as 39 fused forwards that never launched.
        self._count_forward("mixed_fused_fa3")
        fa3 = fused["fa3_wrapper"]
        pers = fused["pers_wrapper"]
        if os.environ.get("SGLANG_FUSED_DUMP") and not getattr(
            FlashInferAttnBackend, "_fused_dumped", False
        ):
            FlashInferAttnBackend._fused_dumped = True
            import numpy as _np

            _d = dict(
                num_qo_heads=self.num_qo_heads, num_kv_heads=self.num_kv_heads,
                scaling=layer.scaling,
                sm_fa3=fused["sm_fa3"], n_tok=fused["n_tok"],
                n_rows=fused.get("n_rows", -1), n_pages=fused.get("n_pages", -1),
                q_shape=_np.array(tuple(q.shape)), q_stride=_np.array(tuple(q.stride())),
                o_shape=_np.array(tuple(out.shape)), o_stride=_np.array(tuple(out.stride())),
                k_shape=_np.array(tuple(k_cache.shape)), k_stride=_np.array(tuple(k_cache.stride())),
                v_shape=_np.array(tuple(v_cache.shape)), v_stride=_np.array(tuple(v_cache.stride())),
                fa3_plan=_np.array(list(fa3._plan_info), dtype=_np.int64),
                pers_plan=_np.array(list(pers._plan_info), dtype=_np.int64),
                kv_indices=fused["kv_indices"].detach().cpu().numpy(),
                fa3_sparse=fused["fa3_sparse"].detach().cpu().numpy(),
                qo=_np.asarray(forward_batch.extend_seq_lens_cpu, dtype=_np.int64),
                prefix=_np.asarray(forward_batch.extend_prefix_lens_cpu, dtype=_np.int64),
                q_ptr_mod=q.data_ptr() % 16, o_ptr_mod=out.data_ptr() % 16,
                kvi_ptr_mod=fused["kv_indices"].data_ptr() % 16,
                sparse_ptr_mod=fused["fa3_sparse"].data_ptr() % 16,
                fa3_ws=fa3._int_workspace_buffer.numel(),
                pers_iws=pers.int_workspace_buffer.numel(),
                pers_fws=pers.float_workspace_buffer.numel(),
            )
            _np.savez(os.environ["SGLANG_FUSED_DUMP"], **_d)
            logger.error("fused launch dump written to %s", os.environ["SGLANG_FUSED_DUMP"])
        torch.ops.fused_fa3.fused_run(
            q,
            k_cache,
            v_cache,
            fused["kv_indices"],
            out,
            fa3._int_workspace_buffer,
            list(fa3._plan_info),
            pers.int_workspace_buffer,
            pers.float_workspace_buffer,
            list(pers._plan_info),
            self.num_qo_heads,
            self.num_kv_heads,
            1,
            layer.scaling,
            fused["sm_fa3"],
            fused["n_tok"],
            # fa3_num_ctas, and it must match what the plan was built for: num_sm
            # for the symmetric kernel, sm_split for the asymmetric one.
            self._fused_num_sm() if self._fused_symmetric() else fused["sm_fa3"],
            0,  # pers_kv_page_offset: the plan's indptr is already global
            fused["fa3_sparse"],
            out,
        )
        return out, self._persistent_lse_buf[:num_tokens]


    # ---- fused FA3 + FA2 SM pre-assignment -------------------------------
    #
    # Four constraints, every one of which was a bug in the standalone harness
    # that produced plausible-looking wrong output rather than a crash:
    #
    #  1. FA3's plan must be built for exactly sm_fa3 CTAs. Its scheduler reads
    #     work_indptr[blockIdx.x] over whatever count it was planned for, and
    #     the kernel only gives it blockIdx.x < sm_split, so a mismatch silently
    #     drops the rest of its work (~88% at sm_split=16 against a 132-CTA
    #     plan). The launch refuses a mismatch, so this is enforced twice.
    #  2. The persistent side's KV indices must be rebased past the FA3 rows'
    #     pages: the two plans are independent, so its kv_indptr restarts at 0
    #     while its pages begin after FA3's (1.52 relative error without).
    #  3. FA3 takes vector-sparse offsets, not page ids, because the kernel
    #     multiplies the index by k_stride_n (0.60 relative error without).
    #  4. FA3 must own a contiguous row prefix; see plan_dispatch_prefix.

    def _load_fused_ext(self):
        """Build the fused extension once per process."""
        ext = getattr(FlashInferAttnBackend, "_fused_ext_loaded", None)
        if ext is not None:
            return ext
        import glob
        import os

        from torch.utils.cpp_extension import load

        src = os.environ.get("SGLANG_FUSED_FA3_SRC")
        if not src or not os.path.exists(src):
            logger.warning(
                "SGLANG_FUSED_FA3_SRC unset or missing; fused FA3+FA2 path disabled"
            )
            FlashInferAttnBackend._fused_ext_loaded = False
            return False
        root = os.path.dirname(os.path.dirname(os.path.abspath(src)))
        gen = glob.glob(
            os.path.expanduser("~/.cache/flashinfer/*/generated/batch_attention_with*")
        )
        if not gen:
            logger.warning("no generated flashinfer config; fused path disabled")
            FlashInferAttnBackend._fused_ext_loaded = False
            return False
        load(
            name="fused_fa3",
            sources=[src],
            extra_include_paths=[
                gen[0],
                f"{root}/include",
                f"{root}/csrc",
                f"{root}/3rdparty/cutlass/include",
                f"{root}/3rdparty/cutlass/tools/util/include",
                f"{root}/3rdparty/spdlog/include",
            ],
            extra_cuda_cflags=[
                "-std=c++17", "--expt-relaxed-constexpr", "-arch=sm_90a", "-O3",
                "--threads", "32",
                "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            ],
            is_python_module=False,
            verbose=False,
        )
        logger.info("fused FA3+FA2 extension loaded from %s", src)
        FlashInferAttnBackend._fused_ext_loaded = True
        return True

    def _fa3_sparse_indices(self, w, kv_cache, page_size=1):
        """The index array flashinfer's own fa3 path feeds the kernel."""
        from flashinfer.page import block_sparse_indices_to_vector_sparse_offsets

        # get_kv_buffer returns either a fused [N,2,...] tensor or a (k, v)
        # tuple depending on the pool layout; only the strides are needed here.
        if isinstance(kv_cache, (tuple, list)):
            k = kv_cache[0]
        elif kv_cache.dim() >= 4:
            k = kv_cache[:, 0]
        else:
            k = kv_cache
        # FA3 addresses a page as ratio * k_stride_n, and the extension derives
        # k_stride_n by rank, so the ratio has to match. sglang's pool is rank 3
        # [tokens, H, D] with one token per page, where stride(0) // stride(1)
        # would be num_kv_heads, not the page size.
        ratio = k.stride(0) // k.stride(1) if k.dim() >= 4 else 1
        return block_sparse_indices_to_vector_sparse_offsets(
            w._paged_kv_indices_buf,
            w._paged_kv_indptr_buf,
            w._vector_sparse_indices_buffer,
            w._vector_sparse_indptr_buffer,
            w._kv_lens_buffer,
            ratio,
            1,
            page_size,
        )

    def _plan_fused_split(self, forward_batch, wrappers=None):
        """Plan both roles for this step, or return None to use the plain path."""
        import os

        if os.environ.get("SGLANG_FUSED_FA3_SM_SPLIT", "0") != "1":
            return None
        if not forward_batch.forward_mode.is_mixed():
            return None
        if not self._load_fused_ext():
            return None

        try:
            return self._plan_fused_split_inner(forward_batch, wrappers)
        except Exception as e:
            # Degrade to the ordinary persistent path rather than killing the
            # server. Logged at WARNING once per distinct message so a
            # systematic planning failure is visible instead of silent.
            key = str(e)[:120]
            seen = getattr(self, "_fused_plan_warned", None)
            if seen is None:
                seen = self._fused_plan_warned = set()
            if key not in seen:
                seen.add(key)
                logger.warning("fused plan unavailable, using persistent: %s", key)
            return None

    def _plan_fused_split_inner(self, forward_batch, wrappers=None):
        import os

        import numpy as np

        qo = np.asarray(forward_batch.extend_seq_lens_cpu, dtype=np.int64)
        prefix = np.asarray(forward_batch.extend_prefix_lens_cpu, dtype=np.int64)
        kv = prefix + qo
        gqa = self.num_qo_heads // self.num_kv_heads
        mode, sm_fa3, sm_pers, _kv_split, n_rows, n_tok, n_pages = md_plan_prefix(
            qo, kv, gqa, self.num_kv_heads, self._fused_num_sm()
        )
        if mode != "split" or n_rows == 0:
            return None

        import flashinfer
        from flashinfer.attention import BatchAttention

        B = int(qo.size)
        if n_rows >= B:
            return None  # nothing left for the persistent role

        # Per-side indptrs. Both restart at 0 because the two plans are
        # independent; the launch is told the row and page offsets so it can
        # advance the persistent side's base pointers (constraints 2 and 4).
        qo_c = np.concatenate([[0], np.cumsum(qo)]).astype(np.int32)
        kv_c = np.concatenate([[0], np.cumsum(np.maximum(kv, 1))]).astype(np.int32)
        qi_f = torch.from_numpy(qo_c[: n_rows + 1].copy())
        ki_f = torch.from_numpy(kv_c[: n_rows + 1].copy())
        qi_r = torch.from_numpy((qo_c[n_rows:] - qo_c[n_rows]).copy())
        # NOT rebased to 0, and the KV index array is NOT sliced. Giving the plan
        # kv_indices[n_pages:] put its base at base + 4*n_pages bytes, which is
        # 16-byte aligned only when n_pages % 4 == 0; the mainloop gathers those
        # indices with vectorised loads, so an odd page count produced
        # "CUDA error: misaligned address" and aborted the server. The standalone
        # harness never hit it because its page count happened to be a multiple
        # of 4. Keeping the indptr global means the plan and the launch agree on
        # one addressing scheme instead of offsetting twice.
        ki_r = torch.from_numpy(kv_c[n_rows:].copy())
        kl_r = torch.from_numpy(np.maximum(kv[n_rows:], 1).astype(np.int32))

        # The page index array for this batch, in request order.
        #
        # This used to reuse wrappers[0]._kv_indices on the assumption that the
        # mixed wrapper, planned over the whole batch moments earlier, already
        # held exactly this array. It does not: on a measured faulting step that
        # buffer held 3846 entries where the batch needed 4102, and its tail 256
        # entries were uninitialised (page ids up to 1.1e9 against a 1.8e6 pool).
        # The kernel gathered those as page ids and died in-kernel with
        # "CUDA error: misaligned address", taking the server with it.
        #
        # So build it, but vectorised on the GPU -- the reuse existed to avoid a
        # per-request Python loop, and that cost does not have to come back for
        # the sake of correctness.
        kv_indices = self._fused_kv_indices(forward_batch, kv_c)
        if kv_indices is None:
            return None
        # A wrong-length array is a declined plan, not a GPU fault. The absence
        # of this check is what let the bug above reach the kernel at all.
        need = int(kv_c[-1])
        if kv_indices.numel() < need:
            logger.warning(
                "fused: kv_indices has %d entries but the batch needs %d; declining",
                kv_indices.numel(), need,
            )
            return None

        st = getattr(self, "_fused_state", None)
        if st is None:
            dev = q_device = kv_indices.device
            # Sized deliberately, not copied from the standalone test. These are
            # allocated lazily on the first accepted batch -- after the KV pool
            # was already sized from --mem-fraction-static -- so every byte here
            # competes with activations rather than coming out of a budget. The
            # original 256+512 MB cost 8.5% of server throughput on its own,
            # with the kernel contributing nothing measurable.
            # The persistent side's float workspace is
            # max_num_kv_splits * (2*head_dim*num_kv_heads + 4) bytes, and
            # max_num_kv_splits = 4 * num_clusters * (CTA_TILE_Q[0]+CTA_TILE_Q[1]).
            # At the default 264 clusters that is ~201 MB, which is why the
            # original 512 MB was not arbitrary -- cutting it to 64 MB simply
            # made every plan fail with an allocator buffer overflow, and the
            # guard then disabled the path silently.
            #
            # The right lever is the cluster count, not the buffer: this path
            # only ever gives the persistent side num_sm - sm_fa3 CTAs, so
            # planning it for 264 clusters is overprovisioning by ~4x. Capping
            # clusters shrinks the buffer proportionally and is correct rather
            # than a workaround.
            fa3_mb = int(os.environ.get("SGLANG_FUSED_FA3_WS_MB", "64"))
            # 320, matching the lifted cluster cap above: the float workspace is
            # max_num_kv_splits * (2*head_dim*num_kv_heads + 4) and scales with
            # the cluster count, so 136 clusters does not fit in 160 MB. Leaving
            # it at 160 made every plan fail the allocator check and the guard
            # then declined the path silently -- the failure mode that once made
            # a control arm look like a result.
            pers_mb = int(os.environ.get("SGLANG_FUSED_PERS_WS_MB", "320"))
            st = self._fused_state = dict(
                fa3=flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    torch.empty(fa3_mb << 20, dtype=torch.uint8, device=dev),
                    kv_layout="NHD", backend="fa3"),
                pers=BatchAttention(
                    torch.empty(pers_mb << 20, dtype=torch.uint8, device=dev), "NHD"),
            )

        # Cap the persistent side's cluster count to what can actually run it.
        # This both matches reality and keeps its float workspace within the
        # budget allocated above.
        # Default 160, not 96. The old 96 was chosen to keep the float workspace
        # inside 160 MB, not because 96 was right: at the measured-optimal
        # sm_fa3 = 64 the persistent role gets 68 CTAs, so 2*(132-64) = 136
        # clusters is what matches its grid, and 96 truncated that by 30%.
        # Lifting the cap to 136 is worth +1.5% end-to-end and improves every
        # latency percentile (two samples: 24446.95 / 24395.78 tok/s, mean ITL
        # 18.85 / 18.90, p99 ITL 33.18 / 35.07, against 24055.80 / 19.18 / 34.74
        # at cap 96). It costs ~160 MB more workspace, which is why the default
        # workspace moves with it.
        cluster_cap = int(os.environ.get("SGLANG_FUSED_PERS_CLUSTER_CAP", "160"))
        pers_clusters = max(16, min(2 * (self._fused_num_sm() - int(sm_fa3)), cluster_cap))
        os.environ["FLASHINFER_PERSISTENT_MAX_CLUSTERS"] = str(pers_clusters)

        # Re-planning is the dominant cost of this path: two full plans, each
        # with its own H2D copy. Skip it when the layout that determines both
        # plans is unchanged from the previous step.
        key = (n_rows, int(sm_fa3), int(qo[:n_rows].sum()), int(kv[:n_rows].sum()),
               int(qo[n_rows:].sum()), int(kv[n_rows:].sum()), int(qo.size))
        if st.get("key") == key and st.get("plan") is not None:
            cached = dict(st["plan"])
            cached["kv_indices"] = kv_indices
            return cached

        # Constraint 1: FA3's plan must be built for exactly sm_fa3 CTAs.
        #
        # Scoped to this one call, not left in os.environ. PrefillSM90Plan reads
        # the variable at plan time, so leaving it set capped CTAs for every
        # later flashinfer fa3 plan in the process -- a global side effect from
        # a per-step decision.
        hd = self.head_dim
        _prev = os.environ.get("FLASHINFER_FA3_MAX_CTAS")
        # Whole grid in symmetric mode: every CTA owns an FA3 slice there, while
        # sm_split only decides who starts on FA3 rather than who may reach it.
        os.environ["FLASHINFER_FA3_MAX_CTAS"] = str(
            self._fused_num_sm() if self._fused_symmetric() else int(sm_fa3)
        )
        try:
            st["fa3"].plan(
                qi_f, ki_f, kv_indices, torch.ones(n_rows, dtype=torch.int32),
                self.num_qo_heads, self.num_kv_heads, hd, 1, causal=False,
                q_data_type=self.q_data_type, kv_data_type=self.kv_data_type,
            )
        finally:
            if _prev is None:
                os.environ.pop("FLASHINFER_FA3_MAX_CTAS", None)
            else:
                os.environ["FLASHINFER_FA3_MAX_CTAS"] = _prev
        # Constraint 2: the persistent side's pages start after FA3's -- expressed
        # in its indptr (ki_r is global), not by slicing the index array.
        st["pers"].plan(
            qi_r, ki_r, kv_indices, kl_r,
            self.num_qo_heads, self.num_kv_heads, hd, hd, 1, causal=False,
            q_data_type=self.q_data_type, kv_data_type=self.kv_data_type,
        )
        plan = dict(
            sm_fa3=int(sm_fa3), n_rows=int(n_rows), n_tok=int(n_tok),
            n_pages=int(n_pages), fa3_wrapper=st["fa3"], pers_wrapper=st["pers"],
            kv_indices=kv_indices,
            # Constraint 3: FA3 takes vector-sparse offsets, not page ids.
            fa3_sparse=self._fa3_sparse_indices(
                st["fa3"], forward_batch.token_to_kv_pool.get_kv_buffer(0)
            ),
        )
        st["key"], st["plan"] = key, plan
        if os.environ.get("SGLANG_FUSED_FA3_DRYRUN", "0") == "1":
            # Everything above still happened -- both workspaces allocated, both
            # sides planned, FLASHINFER_FA3_MAX_CTAS set -- but returning None
            # means the kernel never launches. Comparing this against the full
            # path isolates the kernel's own cost from the cost of merely
            # enabling the path, which the planning optimization showed is where
            # the -8.7% actually lives.
            return None
        return plan

    def _fused_symmetric(self):
        """Whether the symmetric (FA2-role CTAs also run FA3) kernel is selected.

        Read from the environment on every call rather than cached: the flag is
        also read per-launch inside the kernel dispatch, and an A/B that flips it
        between runs must not measure one variant twice.
        """
        import os

        return os.environ.get("FLASHINFER_FUSED_SYMMETRIC", "0") == "1"

    def _fused_num_sm(self):
        n = getattr(FlashInferAttnBackend, "_sm_count", None)
        if n is None:
            n = FlashInferAttnBackend._sm_count = torch.cuda.get_device_properties(
                0
            ).multi_processor_count
        return n

    def _fused_kv_indices(self, forward_batch, kv_cumsum):
        """Page ids for every row of this batch, concatenated in request order.

        Vectorised on the GPU: one repeat_interleave to get each output slot's
        row, one subtraction to get its offset within that row, one gather from
        req_to_token. No Python loop and no per-request host sync, which is what
        made the earlier buffer-reuse shortcut tempting in the first place.
        """
        try:
            req_pool = forward_batch.req_pool_indices
            seq_lens = forward_batch.seq_lens
            r2t = forward_batch.req_to_token_pool.req_to_token
            dev = r2t.device
            B = int(req_pool.shape[0])
            lens = seq_lens[:B].to(dev, torch.int64)
            total = int(kv_cumsum[-1])
            # kv_cumsum comes from the host-side plan; it must agree with the
            # device-side seq_lens or the two sides disagree about addressing.
            if int(lens.sum()) != total:
                logger.warning(
                    "fused: seq_lens sum %d != plan page total %d; declining",
                    int(lens.sum()), total,
                )
                return None
            row = torch.repeat_interleave(
                torch.arange(B, device=dev, dtype=torch.int64), lens
            )
            starts = torch.cumsum(lens, 0) - lens
            off = torch.arange(total, device=dev, dtype=torch.int64) - starts[row]
            return r2t[req_pool.to(dev, torch.int64)[row], off].to(torch.int32)
        except Exception as e:
            logger.warning("fused: could not build kv_indices: %s", e)
            return None

    def pod_forward(
        self,
        pod_wrapper: BatchPODWithPagedKVCacheWrapper,
        forward_batch: ForwardBatch,
        q: torch.Tensor,
        layer: RadixAttention,
        logits_soft_cap: float,
    ):
        if logits_soft_cap != 0:
            raise ValueError("logits_soft_cap is not supported for POD attention yet")
        q_reshaped = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        kv_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        num_p_tokens = pod_wrapper._qo_indptr_buf_p[-1]
        q_p = q_reshaped[:num_p_tokens]
        q_d = q_reshaped[num_p_tokens:]
        try:
            o, lse = pod_wrapper.run(
                q_p,
                kv_buffer,
                q_d,
                kv_buffer,
                causal_p=not layer.is_cross_attention,
                return_lse=True,
                join_outputs=True,
            )
        except Exception as e:
            torch.distributed.breakpoint()
            raise e
        return o, lse

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        logits_soft_cap = layer.logit_cap

        q = q.contiguous()
        if not self.forward_metadata.use_ragged:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
            # Use POD Attention for mixed batches
            if self.use_pod and forward_batch.forward_mode.is_mixed():
                wrapper_idx = self._get_wrapper_idx(layer)
                pod_wrapper = prefill_wrapper_paged
                o, _ = self.pod_forward(
                    pod_wrapper,
                    forward_batch,
                    q,
                    layer,
                    logits_soft_cap,
                )
            elif isinstance(prefill_wrapper_paged, BatchAttention):
                o, _ = self.persistent_attn_forward(
                    prefill_wrapper_paged,
                    forward_batch,
                    q,
                    layer,
                    logits_soft_cap,
                )
            else:
                o = prefill_wrapper_paged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                    causal=not layer.is_cross_attention,
                    sm_scale=layer.scaling,
                    # Disable sliding window attention for multi-item scoring:
                    # - Sliding window could cut across item boundaries, breaking semantic coherence
                    # - Multi-item sequences need full attention to properly handle delimiter tokens
                    # - Specialized multi-item parameters (prefix_len_ptr, token_pos_in_items_ptr)
                    #   provide more precise attention control than simple sliding windows
                    # - Item-aware masking takes precedence over window-based masking
                    window_left=(
                        layer.sliding_window_size
                        if not (
                            self.forward_metadata.multi_item_params
                            and self.forward_metadata.multi_item_params.is_enabled()
                        )
                        else -1
                    ),
                    logits_soft_cap=logits_soft_cap,
                    # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
                    k_scale=layer.k_scale_float,
                    v_scale=layer.v_scale_float,
                )
        else:
            causal = True
            if (
                layer.is_cross_attention
                or layer.attn_type == AttentionType.ENCODER_ONLY
            ):
                causal = False
            if save_kv_cache and layer.attn_type == AttentionType.ENCODER_ONLY:
                save_kv_cache = False

            if self.forward_metadata.extend_no_prefix:
                # NOTE: FlashInfer currently has limitations with head_dim = 32 or other dimensions
                # The FlashInfer head_dim limitation itself is tracked here:
                # https://github.com/flashinfer-ai/flashinfer/issues/1048
                o = self.prefill_wrapper_ragged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=causal,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )
            else:
                q = q.view(-1, layer.tp_q_head_num, layer.head_dim).contiguous()
                k = k.view(-1, layer.tp_k_head_num, layer.head_dim).contiguous()
                v = v.view(-1, layer.tp_v_head_num, layer.head_dim).contiguous()
                o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                    q,
                    k,
                    v,
                    causal=True,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )
                if self.use_pod and forward_batch.forward_mode.is_mixed():
                    o2, s2 = self.pod_forward(
                        prefill_wrapper_paged,
                        forward_batch,
                        q,
                        layer,
                        logits_soft_cap,
                    )
                elif isinstance(prefill_wrapper_paged, BatchAttention):
                    o2, s2 = self.persistent_attn_forward(
                        prefill_wrapper_paged,
                        forward_batch,
                        q,
                        layer,
                        logits_soft_cap,
                    )
                else:
                    o2, s2 = prefill_wrapper_paged.forward_return_lse(
                        q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                        causal=False,
                        sm_scale=layer.scaling,
                        logits_soft_cap=logits_soft_cap,
                    )
                o, _ = merge_state(o1, s1, o2, s2)

            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        decode_wrapper = self.forward_metadata.decode_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Always use standard decode forward — persistent kernel only activates
        # on mixed batches (forward_extend with is_mixed()), not decode.
        o = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _get_wrapper_idx(self, layer: RadixAttention):
        if self.num_wrappers == 1:
            return 0

        if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            return layer.sliding_window_size == -1
        if self.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            return layer.is_cross_attention

        raise ValueError(f"Unknown dispatch reason: {self.dispatch_reason}")


class FlashInferIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.qo_indptr = attn_backend.qo_indptr
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator

        # Dispatch the update function
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        decode_wrappes = decode_wrappers or self.decode_wrappers
        self.call_begin_forward(
            decode_wrappers[0],
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.kv_indptr[0],
            None,
            spec_info,
            seq_lens_cpu,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        assert self.sliding_window_size is not None
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Sliding window attention
                paged_kernel_lens_tmp = torch.clamp(
                    seq_lens, max=self.sliding_window_size + 1
                )
                if seq_lens_cpu is not None:
                    seq_lens_cpu_tmp = torch.clamp(
                        seq_lens_cpu, max=self.sliding_window_size + 1
                    )
                    paged_kernel_lens_sum_tmp = seq_lens_cpu_tmp.sum().item()
                else:
                    paged_kernel_lens_sum_tmp = paged_kernel_lens_tmp.sum().item()
                kv_start_idx_tmp = seq_lens - paged_kernel_lens_tmp
            else:
                # Full attention
                paged_kernel_lens_tmp = seq_lens
                paged_kernel_lens_sum_tmp = seq_lens_sum
                seq_lens_cpu_tmp = seq_lens_cpu
                kv_start_idx_tmp = None

            use_sliding_window_kv_pool = wrapper_id == 0 and isinstance(
                self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
            )

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens_tmp,
                paged_kernel_lens_sum_tmp,
                self.kv_indptr[wrapper_id],
                kv_start_idx_tmp,
                spec_info,
                seq_lens_cpu=seq_lens_cpu_tmp,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
            else:
                # Cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                seq_lens_sum = encoder_lens.sum().item()

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                seq_lens_sum,
                self.kv_indptr[wrapper_id],
                kv_start_idx,
                spec_info,
                seq_lens_cpu=seq_lens_cpu,
            )

    def call_begin_forward(
        self,
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        if spec_info is None:
            bs = len(req_pool_indices)
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            if wrapper.is_cuda_graph_enabled:
                # Directly write to the cuda graph input buffer
                kv_indices = wrapper._paged_kv_indices_buf
            else:
                kv_indices = torch.empty(
                    paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
                )

            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
        else:
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
            bs = kv_indptr.shape[0] - 1

        if use_sliding_window_kv_pool:
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        global global_override_indptr_cpu
        locally_override = False
        if seq_lens_cpu is not None and global_override_indptr_cpu is None:
            locally_override = True
            global_override_indptr_cpu = torch.empty_like(kv_indptr, device="cpu")
            global_override_indptr_cpu[0] = 0
            global_override_indptr_cpu[1 : bs + 1] = torch.cumsum(seq_lens_cpu, dim=0)

        # Check if this specific wrapper's begin_forward has been replaced with fast_decode_plan
        # by checking if it's a partial function with fast_decode_plan as the func
        wrapper_uses_fast_decode_plan = (
            fast_decode_plan is not None
            and hasattr(wrapper.begin_forward, "func")
            and wrapper.begin_forward.func == fast_decode_plan
        )

        if wrapper_uses_fast_decode_plan:
            # When begin_forward is replaced with fast_decode_plan, pass global_override_indptr_cpu
            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                page_size=1,
                data_type=self.data_type,
                q_data_type=self.q_data_type,
                non_blocking=True,
                fixed_split_size=fixed_split_size,
                disable_split_kv=(
                    disable_split_kv if disable_split_kv is not None else False
                ),
                global_override_indptr_cpu=global_override_indptr_cpu,
            )
        # NOTE: Previously use_persistent forced BatchAttention.plan() for decode steps,
        # but this is slower at the server level because it skips fast_decode_plan and
        # CUDA graph optimizations. The persistent kernel is only used for mixed batches
        # (forward_mode.is_mixed()), so decode steps should use the regular decode wrapper.
        else:
            # When using original begin_forward, don't pass global_override_indptr_cpu
            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                page_size=1,
                data_type=self.data_type,
                q_data_type=self.q_data_type,
                non_blocking=True,
                fixed_split_size=fixed_split_size,
                disable_split_kv=(
                    disable_split_kv if disable_split_kv is not None else False
                ),
            )

        if locally_override:
            global_override_indptr_cpu = None


class FlashInferIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged
        self.wrapper_mixed = attn_backend.prefill_wrappers_mixed

        # Cached buffer for kv_indices in update_pod (avoids per-call allocation)
        self._cached_kv_indices_size = 0
        self._cached_kv_indices = None

        # Dispatch the update function
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        prefix_lens_cpu=None,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[
            Union[BatchPrefillWithPagedKVCacheWrapper, BatchAttention]
        ],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        prefix_lens_cpu=None,
    ):
        if use_ragged:
            paged_kernel_lens = prefix_lens
            if prefix_lens_cpu is not None:
                paged_kernel_lens_sum = int(sum(prefix_lens_cpu))
            else:
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()
            causal = False
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum
            causal = True

        self.call_begin_forward(
            self.prefill_wrapper_ragged,
            prefill_wrappers[0],
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            None,
            self.kv_indptr[0],
            self.qo_indptr[0],
            use_ragged,
            spec_info,
            fixed_split_size=fixed_split_size,
            multi_item_params=multi_item_params,
            causal=causal,
            seq_lens_cpu=seq_lens_cpu,
            prefix_lens_cpu=prefix_lens_cpu,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        prefix_lens_cpu=None,
    ):
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # window attention use paged only
                paged_kernel_lens = torch.minimum(
                    seq_lens,
                    torch.tensor(self.sliding_window_size) + seq_lens - prefix_lens,
                )
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()
                causal = False
            else:
                # full attention
                paged_kernel_lens = seq_lens
                paged_kernel_lens_sum = seq_lens_sum
                causal = True
            kv_start_idx = seq_lens - paged_kernel_lens
            use_sliding_window_kv_pool = wrapper_id == 0 and isinstance(
                self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
            )

            self.call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
                multi_item_params=multi_item_params,
                causal=causal,
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        prefix_lens_cpu=None,
    ):
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
                paged_kernel_lens_sum = seq_lens_sum
                causal = True
            else:
                # cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()
                causal = False

            self.call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
                fixed_split_size=fixed_split_size,
                multi_item_params=multi_item_params,
                causal=causal,
            )

    def update_pod(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        pod_wrapper: BatchPODWithPagedKVCacheWrapper,
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        page_size: int,
        num_prefill: int,
        num_decode: int,
        prefix_lens_cpu=None,
    ):
        """Update POD wrapper for mixed batches. Prefill requests come first, then decode."""
        bs = len(seq_lens)
        assert len(seq_lens) == len(req_pool_indices)
        assert num_prefill + num_decode == bs
        assert isinstance(pod_wrapper, BatchPODWithPagedKVCacheWrapper)

        if use_ragged:
            paged_kernel_lens = prefix_lens
        else:
            paged_kernel_lens = seq_lens

        # Compute paged_kernel_lens_sum on CPU (avoids GPU->CPU sync)
        if use_ragged and prefix_lens_cpu is not None:
            paged_kernel_lens_cpu = torch.tensor(prefix_lens_cpu, dtype=torch.int32) if not isinstance(prefix_lens_cpu, torch.Tensor) else prefix_lens_cpu
            paged_kernel_lens_sum = int(paged_kernel_lens_cpu.sum())
        elif not use_ragged and seq_lens_cpu is not None:
            paged_kernel_lens_sum = seq_lens_sum
        else:
            paged_kernel_lens_sum = int(paged_kernel_lens.sum().to(device="cpu"))

        # Create full kv_indptr for all requests
        kv_indptr = self.kv_indptr[0]
        kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
        kv_indptr = kv_indptr[: bs + 1]
        p_kv_indptr = kv_indptr[: num_prefill + 1]
        d_kv_indptr = kv_indptr[num_prefill:]

        # Compute qo_indptr separately for prefill and decode
        extend_lens = seq_lens - prefix_lens
        p_q_indptr = self.qo_indptr[0][: num_prefill + 1]
        if num_prefill > 0:
            p_q_indptr[1:] = torch.cumsum(extend_lens[:num_prefill], dim=0)
        d_q_indptr = self.qo_indptr[1][: num_decode + 1]
        if num_decode > 0:
            d_q_indptr[1:] = torch.cumsum(extend_lens[num_prefill:], dim=0)

        # Create kv_indices using cached buffer (avoids per-call allocation)
        needed = paged_kernel_lens_sum + 256
        if needed > self._cached_kv_indices_size:
            self._cached_kv_indices_size = needed
            self._cached_kv_indices = torch.empty(
                needed, dtype=torch.int32, device=req_pool_indices.device,
            )
        kv_indices = self._cached_kv_indices
        kv_start_idx = encoder_lens if encoder_lens is not None else None
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr,
            kv_start_idx,
            kv_indices,
            self.req_to_token.shape[1],
        )
        kv_indices = kv_indices[:paged_kernel_lens_sum]

        # Build CPU tensors for pod_wrapper.plan() to avoid GPU->CPU sync
        if seq_lens_cpu is not None and prefix_lens_cpu is not None:
            prefix_lens_cpu_t = torch.tensor(prefix_lens_cpu, dtype=torch.int32) if not isinstance(prefix_lens_cpu, torch.Tensor) else prefix_lens_cpu
            paged_kernel_lens_cpu = prefix_lens_cpu_t if use_ragged else seq_lens_cpu
            kv_indptr_cpu = torch.zeros(bs + 1, dtype=torch.int32)
            kv_indptr_cpu[1:] = torch.cumsum(paged_kernel_lens_cpu, dim=0)
            extend_lens_cpu = seq_lens_cpu - prefix_lens_cpu_t
            p_q_indptr_cpu = torch.zeros(num_prefill + 1, dtype=torch.int32)
            if num_prefill > 0:
                p_q_indptr_cpu[1:] = torch.cumsum(extend_lens_cpu[:num_prefill], dim=0)
            d_q_indptr_cpu = torch.zeros(num_decode + 1, dtype=torch.int32)
            if num_decode > 0:
                d_q_indptr_cpu[1:] = torch.cumsum(extend_lens_cpu[num_prefill:], dim=0)
            last_page_len_host_p = torch.ones(num_prefill, dtype=torch.int32)
            last_page_len_host_d = torch.ones(num_decode, dtype=torch.int32)
            p_kv_indptr_cpu = kv_indptr_cpu[:num_prefill + 1]
            d_kv_indptr_cpu = kv_indptr_cpu[num_prefill:]
        else:
            p_q_indptr_cpu = None
            d_q_indptr_cpu = None
            last_page_len_host_p = None
            last_page_len_host_d = None
            p_kv_indptr_cpu = None
            d_kv_indptr_cpu = None

        # Use same kv_indices for both, split kv_indptr
        self.p_q_indptr = p_q_indptr
        self.d_q_indptr = d_q_indptr
        self.p_kv_indptr = p_kv_indptr
        self.d_kv_indptr = d_kv_indptr
        self.kv_indices = kv_indices[: paged_kernel_lens_sum + 256]
        self.kv_last_page_len_p = self.kv_last_page_len[:num_prefill]
        self.kv_last_page_len_d = self.kv_last_page_len[num_prefill:bs]
        pod_wrapper.plan(
            # Prefill params
            p_q_indptr,
            p_kv_indptr,
            kv_indices,
            self.kv_last_page_len[:num_prefill],
            # Decode params
            d_q_indptr,
            d_kv_indptr,
            kv_indices,
            self.kv_last_page_len[num_prefill:bs],
            # Common params
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=page_size,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            # CPU tensors to avoid GPU->CPU sync
            qo_indptr_host_p=p_q_indptr_cpu,
            kv_indptr_host_p=p_kv_indptr_cpu,
            last_page_len_host_p=last_page_len_host_p,
            qo_indptr_host_d=d_q_indptr_cpu,
            kv_indptr_host_d=d_kv_indptr_cpu,
            last_page_len_host_d=last_page_len_host_d,
        )
        if use_ragged:
            qo_indptr = self.qo_indptr[0][: bs + 1]
            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            wrapper_ragged.plan(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: Union[BatchPrefillWithPagedKVCacheWrapper, BatchAttention],
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_start_idx: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[SpecInput],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        causal: bool = False,  # cached prefix should be attended to by all queries. Defaulting to True will cause wrong results for the persistent kernel.
        seq_lens_cpu: Optional[torch.Tensor] = None,
        prefix_lens_cpu=None,
    ):
        bs = len(seq_lens)
        if spec_info is None:
            assert len(seq_lens) == len(req_pool_indices)
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]  # cumulative page ranges for each req
            # For BatchAttention in CUDA graph mode, write kv_indices directly
            # to the pre-allocated buffer so the graph kernel sees the update.
            if (
                isinstance(wrapper_paged, BatchAttention)
                and wrapper_paged.is_cuda_graph_enabled
            ):
                kv_indices = wrapper_paged._paged_kv_indices_buf
            else:
                kv_indices = torch.empty(
                    paged_kernel_lens_sum + 256,
                    dtype=torch.int32,
                    device=req_pool_indices.device,
                )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
        else:
            assert not isinstance(wrapper_paged, BatchAttention), (
                "Persistent kernel doesn't support custom mask yet; "
                "there's a bug in the dispatch logic"
            )
            assert isinstance(spec_info, SpecInput)
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    self.req_to_token,
                )
            )

        if use_ragged:
            # extend part
            wrapper_ragged.plan(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

        if use_sliding_window_kv_pool:
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        # cached part
        # Conditionally set multi-item parameters
        if multi_item_params is not None and multi_item_params.is_enabled():
            if isinstance(wrapper_paged, BatchAttention):
                raise NotImplementedError(
                    "Multi-item scoring is not supported for persistent attention yet"
                )
            # Multi-item scoring is active - use specialized parameters and disable generic custom_mask
            use_custom_mask = None
            prefix_len_ptr = multi_item_params.prefix_len_ptr
            token_pos_in_items_ptr = multi_item_params.token_pos_in_items_ptr
            token_pos_in_items_len = multi_item_params.token_pos_in_items_len
            max_item_len_ptr = multi_item_params.max_item_len_ptr
        else:
            # No multi-item scoring - use standard parameters
            use_custom_mask = custom_mask
            prefix_len_ptr = None
            token_pos_in_items_ptr = None
            token_pos_in_items_len = 0
            max_item_len_ptr = None

        if self.attn_backend.use_persistent and isinstance(wrapper_paged, BatchAttention):
            # Work ordering comes from the same measured gate that picked the
            # persistent kernel for this batch; the sweep shows the non-flipped
            # ordering is the better default on the cached-prefix composition,
            # with flipped winning only in a measured decode-dominant pocket.
            flipped = getattr(self.attn_backend, "_mixed_gate_flipped", None)
            if flipped is None:
                flipped = select_persistent_flipped_schedule(
                    paged_kernel_lens_sum,
                    len(paged_kernel_lens),
                    self.attn_backend.no_flipped_schedule,
                )
            # Enable CUDA graph mode for persistent kernel — sglang uses CG for
            # all BatchAttention calls. This forces invariant workspace layout so
            # capture and replay have consistent offsets regardless of batch composition.
            # For CG: pass max_bs to force invariant workspace layout across capture/replay.
            max_bs = getattr(self.attn_backend, 'mixed_cuda_graph_max_bs', 0)
            plan_qo_indptr, plan_kv_indptr, plan_kv_lens = (
                qo_indptr,
                kv_indptr,
                paged_kernel_lens,
            )
            if seq_lens_cpu is not None and prefix_lens_cpu is not None:
                # Build plan metadata on the host so BatchAttention.plan() can
                # skip its device-to-host copy and the full device synchronize
                # that copy requires. The synchronize otherwise stalls the CPU
                # behind every queued forward step on each mixed batch.
                seq_np = seq_lens_cpu.numpy().astype(np.int64, copy=False)
                prefix_np = np.asarray(prefix_lens_cpu, dtype=np.int64)
                extend_np = seq_np - prefix_np
                kv_lens_np = prefix_np if use_ragged else seq_np
                zero = np.zeros(1, dtype=np.int64)
                plan_qo_indptr = torch.from_numpy(
                    np.concatenate((zero, np.cumsum(extend_np))).astype(np.int32)
                )
                plan_kv_indptr = torch.from_numpy(
                    np.concatenate((zero, np.cumsum(kv_lens_np))).astype(np.int32)
                )
                plan_kv_lens = torch.from_numpy(kv_lens_np.astype(np.int32))
            wrapper_paged.plan(
                plan_qo_indptr,
                plan_kv_indptr,
                kv_indices,
                plan_kv_lens,
                self.num_qo_heads,
                self.num_kv_heads,
                head_dim_qk=self.head_dim,
                head_dim_vo=self.head_dim,
                page_size=1,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
                causal=causal,
                flipped_schedule=flipped,
                enable_cuda_graph=(max_bs > 0),
                cuda_graph_max_bs=max_bs,
            )
        else:
            wrapper_paged.begin_forward(
                qo_indptr,
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
                custom_mask=use_custom_mask,
                non_blocking=True,
                fixed_split_size=fixed_split_size,
                prefix_len_ptr=prefix_len_ptr,
                token_pos_in_items_ptr=token_pos_in_items_ptr,
                token_pos_in_items_len=token_pos_in_items_len,
                max_item_len_ptr=max_item_len_ptr,
            )


class FlashInferMultiStepDraftBackend:
    """
    Wrap multiple flashinfer attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices

        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        self.page_size = model_runner.page_size

        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.attn_backends: List[FlashInferAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FlashInferAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    kv_last_page_len_buf=self.kv_last_page_len,
                )
            )

        self.max_context_len = self.attn_backends[0].max_context_len

        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: torch.Tensor,
        call_fn: Callable,
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
        )

        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        # Copy the kv_indptr once to avoid multiple device-to-host copies in flashinfer's plan.
        indptr_cpu_whole = self.kv_indptr[:, : bs + 1].cpu()
        global global_override_indptr_cpu

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            global_override_indptr_cpu = indptr_cpu_whole[i]
            call_fn(i, forward_batch)

        global_override_indptr_cpu = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices = torch.empty(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int32,
            device="cuda",
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_bs * self.max_context_len),
            dtype=torch.int32,
            device="cuda",
        )

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)


def should_use_tensor_core(
    kv_cache_dtype: torch.dtype,
    num_attention_heads: int,
    num_kv_heads: int,
) -> bool:
    """
    Determine whether to use tensor cores for attention computation.

    Args:
        kv_cache_dtype: Data type of the KV cache
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key/value heads

    Returns:
        bool: Whether to use tensor cores
    """
    # Try to use environment variable first
    env_override = os.environ.get("SGLANG_FLASHINFER_USE_TENSOR_CORE")
    if env_override is not None:
        return env_override.lower() == "true"

    # Try to use _grouped_size_compiled_for_decode_kernels if available
    # This is for flashinfer <=0.1.6. Otherwise, there is an accuracy bug
    try:
        from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

        if not _grouped_size_compiled_for_decode_kernels(
            num_attention_heads,
            num_kv_heads,
        ):
            return True
        else:
            return False
    except (ImportError, AttributeError):
        pass

    # Calculate GQA group size
    gqa_group_size = num_attention_heads // num_kv_heads

    # For Flashinfer, a GQA group size of at least 4 is needed to efficiently
    # use Tensor Cores, as it fuses the head group with the token dimension in MMA.
    if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True
    elif kv_cache_dtype in (torch.float16, torch.half, torch.bfloat16):
        return gqa_group_size >= 4
    else:
        return False
