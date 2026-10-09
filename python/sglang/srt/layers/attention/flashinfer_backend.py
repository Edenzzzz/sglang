from __future__ import annotations

from sglang.srt.runtime_context import (
    get_buffer,
    get_exec,
    get_parallel,
    get_platform,
)

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch

from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.kernels.ops.attention.utils import (
    assert_buffer_fits,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    KVCacheAttentionAccessKind,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.kv_loc_plan import IdSpaceKind
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import (
    draft_kv_indices_buffer_width,
    draft_kv_indices_used_len,
    generate_draft_decode_kv_indices,
    resolve_draft_decode_window,
)
from sglang.srt.utils import (
    get_cuda_graph_max_batch_size,
    get_int_env_var,
    is_flashinfer_available,
    next_power_of_2,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.kv_loc_plan import KVLocPlan
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


if envs.SGLANG_ENABLE_TORCH_COMPILE.get():
    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True


if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
        fast_decode_plan,
    )
    from flashinfer.cascade import merge_state

    from sglang.kernels.ops.attention.merge_state import merge_state_triton

    # FlashInfer's MergeState CUDA kernel uses blockDim = (head_dim/vec_size, num_heads).
    # When num_heads is large (e.g. with DP attention where attention_tp_size=1), the
    # total threads per block can exceed CUDA's limit of 1024 and the kernel launch fails
    # with `invalid configuration argument`. Fall back to the in-tree Triton implementation,
    # which uses (token, head) as the launch grid and is therefore unaffected.
    _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK = 1024

    def _merge_state_max_safe_num_heads(head_dim: int, element_size: int) -> int:
        # Mirrors flashinfer's vec_size selection in include/flashinfer/attention/cascade.cuh.
        vec_size = max(16 // element_size, head_dim // 32)
        bdx = head_dim // vec_size
        if bdx <= 0:
            return _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK
        return _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK // bdx

    def _safe_merge_state(
        v_a: torch.Tensor,
        s_a: torch.Tensor,
        v_b: torch.Tensor,
        s_b: torch.Tensor,
    ):
        num_heads = v_a.shape[1]
        head_dim = v_a.shape[2]
        max_heads = _merge_state_max_safe_num_heads(head_dim, v_a.element_size())
        if num_heads <= max_heads:
            return merge_state(v_a, s_a, v_b, s_b)
        return merge_state_triton(v_a, s_a, v_b, s_b)


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


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
    # full->SWA translated out_cache_loc (SWA KV-store write target)
    swa_out_cache_loc: Optional[torch.Tensor] = None


@dataclass
class PrefillMetadata:
    prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper]
    use_ragged: bool
    extend_no_prefix: bool
    multi_item_params: Optional[MultiItemScoringParams] = None
    swa_out_cache_loc: Optional[torch.Tensor] = None


# Reuse this workspace buffer across all flashinfer wrappers

# Safety margin on the computed split-kv worst case for the dedicated
# full-CG prefill workspace (absorbs allocator alignment and minor
# flashinfer sizing drift across versions). Sizing logic lives in
# FlashInferAttnBackend._full_cg_prefill_workspace_bytes.
FULL_CG_PREFILL_WORKSPACE_MARGIN = 1.25

# Use as a fast path to override the indptr in flashinfer's plan function
# This is used to remove some host-to-device copy overhead.
global_override_indptr_cpu = None


def fast_prefill_plan(
    self,
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    page_size: int,
    head_dim_vo: Optional[int] = None,
    custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    window_left: int = -1,
    q_data_type: Union[str, torch.dtype] = "float16",
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
    o_data_type: Optional[Union[str, torch.dtype]] = None,
    non_blocking: bool = True,
    fixed_split_size: Optional[int] = None,
    prefix_len_ptr: Optional[torch.Tensor] = None,
    token_pos_in_items_ptr: Optional[torch.Tensor] = None,
    token_pos_in_items_len: int = 0,
    max_item_len_ptr: Optional[torch.Tensor] = None,
    # Required host-known metadata: lets us skip the per-replay device-to-host
    # copies upstream plan() always issues. Keyword-only with no default so a
    # caller that forgets them fails at the call boundary, not with a cryptic
    # None crash deeper in.
    *,
    qo_indptr_host: torch.Tensor,
    kv_indptr_host: torch.Tensor,
    kv_lens_host: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
) -> None:
    """Sync-free ``BatchPrefillWithPagedKVCacheWrapper.plan`` for the EAGLE
    draft-extend CUDA graph (FlashInfer fa2, cuda-graph mode only).

    Upstream plan() always does qo/paged_kv/last_page_len ``.to("cpu")`` to build
    its host scheduling metadata, a blocking D2H that drains the GPU queue every
    replay. The caller passes host-known qo/kv layout in, so we call the underlying
    ``_cached_module.plan`` directly with no readback; the ``_plan_info`` produced
    is identical to plan()'s.
    """
    assert self.is_cuda_graph_enabled, "fast_prefill_plan is cuda-graph only"
    assert getattr(self, "_backend", None) == "fa2", (
        "fast_prefill_plan supports the fa2 backend only"
    )
    assert getattr(self, "_cached_module", None) is not None, (
        "fast_prefill_plan requires _cached_module from a prior real plan() (capture)"
    )

    if head_dim_vo is None:
        head_dim_vo = head_dim_qk
    batch_size = len(paged_kv_last_page_len)

    total_num_rows = int(qo_indptr_host[-1])
    self._qo_indptr_last = total_num_rows
    self._max_q_len = max_q_len
    self._max_kv_len = max_kv_len

    if self._max_total_num_rows is None:
        self._max_total_num_rows = total_num_rows

    self._batch_size = batch_size
    self._num_qo_heads = num_qo_heads
    self._num_kv_heads = num_kv_heads
    self._prefix_len_ptr = prefix_len_ptr
    self._token_pos_in_items_ptr = token_pos_in_items_ptr
    self._token_pos_in_items_len = token_pos_in_items_len
    self._max_item_len_ptr = max_item_len_ptr

    # Refresh the cuda-graph input buffers (device-to-device, non-blocking).
    self._qo_indptr_buf.copy_(qo_indptr, non_blocking=non_blocking)
    self._paged_kv_indptr_buf.copy_(paged_kv_indptr, non_blocking=non_blocking)
    self._paged_kv_last_page_len_buf.copy_(
        paged_kv_last_page_len, non_blocking=non_blocking
    )
    self._paged_kv_indices_buf[: len(paged_kv_indices)].copy_(
        paged_kv_indices,
        non_blocking=(paged_kv_indices.device == self.device) and non_blocking,
    )

    self._cached_q_data_type = q_data_type
    self._cached_kv_data_type = (
        kv_data_type if kv_data_type is not None else q_data_type
    )
    self._cached_o_data_type = o_data_type
    self._block_tables = None

    args = [
        self._float_workspace_buffer,
        self._int_workspace_buffer,
        self._pin_memory_int_workspace_buffer,
        qo_indptr_host,
        kv_indptr_host,
        kv_lens_host,
        self._max_total_num_rows or total_num_rows,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        page_size,
        self.is_cuda_graph_enabled,
        head_dim_qk,
        head_dim_vo,
        causal,
        window_left,
        fixed_split_size if fixed_split_size is not None else -1,
        False,  # disable_split_kv
        0,  # num_colocated_ctas
        0,  # uniform_q_len
    ]
    self._plan_info = self._cached_module.plan(*args)


# Self-check of the split against the ordinary planned wrapper, in the same step on the same
# inputs. Off by default; it doubles attention work when on.
FUSED_SPLIT_VERIFY = os.environ.get("SGLANG_FUSED_SPLIT_VERIFY", "0") == "1"
FUSED_SPLIT_TOL = float(os.environ.get("SGLANG_FUSED_SPLIT_TOL", "0.02"))
# Localiser for the device-side index assert: synchronises and validates every index tensor
# the split builds, so the fault is attributed to the plan or the dispatch with real values
# instead of being inferred from a traceback that names an unrelated later CUDA call.
FUSED_SPLIT_DEBUG = os.environ.get("SGLANG_FUSED_SPLIT_DEBUG", "0") == "1"
_FS_NP = {torch.int32: "int32", torch.int64: "int64"}
# One-bit bisect for the post-forward index assert. The dispatch does exactly two new things:
# an FA3 call, and planning+running an EXTRA flashinfer wrapper mid-step. With this set, the
# FA2 rows are computed from the batch-wide reference wrapper and scattered instead, so the
# extra wrapper is never planned. Output stays correct either way, so a clean run isolates the
# fault to that extra plan rather than to the FA3 half.
FUSED_SPLIT_NO_FA2_WRAPPER = os.environ.get("SGLANG_FUSED_SPLIT_NOFA2", "0") == "1"


class FlashInferAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    # kv_indptr/qo_indptr are preallocated at (req pool + 1); an extend batch
    # can never carry more seqs than the pool.
    extend_dummy_seqs_capped_by_req_pool: bool = True

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
        init_new_workspace: bool = False,
    ):
        super().__init__()
        self.prefill_backend = "fa2"
        self.decode_backend = "fa2"

        # Fused FA3+FA2 SM-pre-assigned split. `fused_fa2_wrapper` is the FA2 executor for
        # the split's own row subset, planned per step; `fused_sm_margin` is the count of SMs
        # held back from FA3 and passed through upstream FA3's public `sm_margin`.
        self.fused_fa2_wrapper = None
        self.fused_split_mask = None
        self.fused_split_plan = None
        self.fused_sm_margin = int(os.environ.get("SGLANG_FUSED_SM_MARGIN", "0"))
        # Counters initialised here, not lazily via getattr: the verify block reads
        # fused_split_calls, so a lazy counter made the FIRST successful dispatch raise
        # AttributeError after both executors and the scatter had already run.
        # Grow-on-demand buffer pool for the split's per-step index arrays. Upstream
        # preallocates its own indptr/indices buffers for two reasons this path needs as well:
        # a captured CUDA graph holds absolute addresses, so per-step allocations interleaved
        # with replays can collide with graph-visible memory; and the allocation plus H2D work
        # sits on the critical path once per step, which is a prime suspect for the measured
        # 6.8% serving regression. Buffers only ever grow, so the steady state allocates nothing.
        self._fs_dev: dict = {}
        self._fs_host: dict = {}
        self.fused_split_calls = 0
        self.fused_split_onesided = 0
        self.fused_split_plans = 0
        self.fused_split_err = 0.0

        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.kv_index_translator = model_runner.kv_index_translator
        self._swa_kv_pool: Optional[BaseSWAKVPool] = self._resolve_swa_kv_pool(
            model_runner
        )
        self.use_sliding_window_kv_pool = self._swa_kv_pool is not None
        self.enable_mis = get_exec().features.enable_mis

        # FIXME: remove dllm workarounds from flashinfer
        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm_model = self.dllm_config is not None

        self.kv_cache_quant_method = self.token_to_kv_pool.get_kv_cache_quant_method()
        self.prefill_kv_access = self.kv_cache_quant_method.resolve_attention_access(
            "prefill", "flashinfer"
        )
        self.decode_kv_access = self.kv_cache_quant_method.resolve_attention_access(
            "decode", "flashinfer"
        )
        prefill_backend = model_runner.prefill_attention_backend_str
        decode_backend = model_runner.decode_attention_backend_str
        if self.__class__ is FlashInferAttnBackend:
            if prefill_backend == "flashinfer":
                self._check_kv_attention_access("prefill", self.prefill_kv_access)
            if decode_backend == "flashinfer":
                self._check_kv_attention_access("decode", self.decode_kv_access)

        self.prefill_uses_dequant_workspace = (
            self.prefill_kv_access is not None
            and self.prefill_kv_access.kind
            == KVCacheAttentionAccessKind.DEQUANT_WORKSPACE
        )
        self.decode_uses_dequant_workspace = (
            self.decode_kv_access is not None
            and self.decode_kv_access.kind
            == KVCacheAttentionAccessKind.DEQUANT_WORKSPACE
        )
        self.is_nvfp4_kvcache = any(
            access is not None and access.scale_recipe == "nvfp4"
            for access in (self.prefill_kv_access, self.decode_kv_access)
        )
        self.dq_page_table = None
        self.dq_paged_kernel_lens = None
        self.cpu_req_pool_indices = None
        # FP4 fake-quant prefill/decode exposes an FP8 workspace to FlashInfer.
        self.flashinfer_kv_cache_dtype = (
            torch.float8_e4m3fn
            if (
                self.prefill_uses_dequant_workspace
                or self.decode_uses_dequant_workspace
            )
            else model_runner.kv_cache_dtype
        )

        # Parse constants
        self.decode_use_tensor_cores = should_use_tensor_core(
            kv_cache_dtype=self.flashinfer_kv_cache_dtype,
            num_attention_heads=model_runner.model_config.num_attention_heads
            // get_parallel().attn_tp_size,
            num_kv_heads=model_runner.model_config.get_num_kv_heads(
                get_parallel().attn_tp_size, get_parallel().attn_dcp_size
            ),
        )
        self.max_context_len = model_runner.model_config.context_len
        self.page_size = model_runner.page_size
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal
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
            or "Qwen3VLForConditionalGeneration"
            in model_runner.model_config.hf_config.architectures
            or "Qwen3VLMoeForConditionalGeneration"
            in model_runner.model_config.hf_config.architectures
        ):
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(512 * 1024 * 1024)

        # When deterministic inference is enabled, tensor cores should be used for decode
        # Also set split tile sizes for prefill and decode from environment variables, and disable kv split for cuda graph
        # More information can be found here: https://github.com/flashinfer-ai/flashinfer/pull/1675
        self.enable_deterministic = (
            get_exec().deterministic.enable_deterministic_inference
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

        self.use_paged = envs.SGLANG_FLASHINFER_USE_PAGED.get()

        # Allocate buffers
        # different from flashinfer zero_init_global_workspace_buffer
        global_workspace_buffer = get_buffer(
            "flashinfer_workspace",
            lambda: torch.empty(
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                dtype=torch.uint8,
                device=model_runner.device,
            ),
        )
        if init_new_workspace:
            self.workspace_buffer = torch.empty(
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                dtype=torch.uint8,
                device=model_runner.device,
            )
        else:
            self.workspace_buffer = global_workspace_buffer
        max_bs = get_cuda_graph_max_batch_size(model_runner.req_to_token_pool.size)
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
                for _ in range(self.num_wrappers)
            ]

        fmha_backend = "auto"
        if get_platform().is_sm100:
            fmha_backend = "fa2"
            # Disable CUTLASS backend when piecewise cuda graph is enabled
            # due to TMA descriptor initialization issues on SM100 GPUs. The
            # current FlashInfer SM100 CUTLASS FMHA dispatch only instantiates
            # 64x64, 128x128, and 192x128 head dimensions. Keep unsupported
            # shapes (for example Qwen3.5's 256x256) on the FA2 fallback.
            cutlass_supported_head_dims = {
                (64, 64),
                (128, 128),
                (192, 128),
            }
            head_dims = (
                model_runner.model_config.head_dim,
                model_runner.model_config.v_head_dim,
            )
            if (
                head_dims in cutlass_supported_head_dims
                and not check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
            ):
                fmha_backend = "cutlass"
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=fmha_backend
        )

        # Two wrappers: one for sliding window attention and one for full attention.
        # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        self.prefill_wrappers_paged = []
        self.prefill_wrappers_verify = []
        self.decode_wrappers = []
        for _ in range(self.num_wrappers):
            if not skip_prefill:
                self.prefill_wrappers_paged.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend=self.prefill_backend,
                    )
                )
                self.prefill_wrappers_verify.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend=self.prefill_backend,
                    )
                )
            self.decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    backend=self.decode_backend,
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
        # Plain EXTEND under full prefill CUDA graph: one wrapper set
        # shared across all captured num_tokens buckets (bs fixed at 1).
        # Created lazily on first capture in _prepare_cuda_graph_metadata.
        self.full_cg_prefill_wrappers: Optional[
            List[BatchPrefillWithPagedKVCacheWrapper]
        ] = None

    def _check_kv_attention_access(self, phase: str, access) -> None:
        if access is not None:
            return
        method_name = getattr(self.kv_cache_quant_method, "name", "unknown")
        available = self.kv_cache_quant_method.describe_attention_accesses(phase)
        raise ValueError(
            f"KV cache method {method_name!r} does not support {phase} with "
            f"flashinfer attention backend. Available {phase} accesses: {available}."
        )

    @staticmethod
    def _resolve_swa_kv_pool(model_runner: ModelRunner) -> Optional[BaseSWAKVPool]:
        """Return the SWA KV pool to translate against, or None for non-SWA models.

        EAGLE-like draft workers share the target allocator for token bookkeeping,
        but own a separate draft KV pool. Do not use the target allocator's SWA
        mapping for that draft pool. FROZEN_KV MTP is the exception: its draft
        path reads target KV directly, so it still needs the allocator pool when
        the active pool is not SWA.
        """
        active_pool = model_runner.token_to_kv_pool
        if isinstance(active_pool, BaseSWAKVPool):
            return active_pool

        if model_runner.is_draft_worker:
            if not model_runner.spec_algorithm.is_frozen_kv_mtp():
                return None

        kvcache = model_runner.token_to_kv_pool_allocator.get_kvcache()
        return kvcache if isinstance(kvcache, BaseSWAKVPool) else None

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

        if not self.enable_mis or forward_batch.forward_mode == ForwardMode.DECODE:
            return MultiItemScoringParams()

        precomputed_indices = forward_batch.multi_item_delimiter_indices
        if precomputed_indices is None:
            return MultiItemScoringParams()

        prefix_cache_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        prefix_len_ptr, token_pos_in_items_ptr = [], []
        token_pos_in_items_len = 0
        device = forward_batch.input_ids.device

        # If no extend_seq_lens, treat whole batch as one sequence
        if extend_seq_lens is None or len(extend_seq_lens) <= 1:
            extend_seq_lens = [forward_batch.input_ids.size(0)]

        seq_start = 0
        for i, seq_len in enumerate(extend_seq_lens):
            seq_end = seq_start + seq_len
            delimiter_indices_cpu = precomputed_indices[i]
            if len(delimiter_indices_cpu) == 0:
                seq_start = seq_end
                continue

            first_delim = delimiter_indices_cpu[0].item()  # CPU .item(), no GPU sync
            delimiter_indices = delimiter_indices_cpu.to(device, non_blocking=True)
            prefix_len = first_delim + (
                prefix_cache_lens[i] if prefix_cache_lens is not None else 0
            )
            prefix_len_ptr.append(prefix_len)

            # Compute relative positions within items using searchsorted (no GPU sync).
            #   suffix_range      = [0, 1, 2, 3, 4, ...]
            #   searchsorted      = bucket index for each position
            #   last_delim        = delimiter offset at start of current bucket
            #   pos_within_item   = suffix_range - last_delim
            suffix_len = seq_len - first_delim
            relative_positions = delimiter_indices - first_delim

            suffix_range = torch.arange(suffix_len, dtype=torch.int64, device=device)
            bucket_idx = torch.searchsorted(
                relative_positions, suffix_range, right=True
            )
            last_delim = relative_positions[torch.clamp(bucket_idx - 1, min=0)]
            pos_within_item = suffix_range - last_delim

            token_pos_in_items_ptr.append(pos_within_item.to(torch.uint16))

            forward_batch.positions[seq_start + first_delim : seq_end] = (
                prefix_len + pos_within_item - 1
            )

            seq_start = seq_end

        # Pad token_pos_in_items_ptr for batch processing
        if token_pos_in_items_ptr:
            token_pos_in_items_len = max(t.numel() for t in token_pos_in_items_ptr)
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

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        seq_lens_cpu = forward_batch.seq_lens_cpu
        seq_lens_sum = forward_batch.seq_lens_sum
        encoder_lens = forward_batch.encoder_lens
        forward_mode = forward_batch.forward_mode
        spec_info = forward_batch.spec_info

        if (
            spec_info is not None
            and spec_info.ragged_verify_layout is not None
            and forward_mode.is_target_verify()
        ):
            raise NotImplementedError(
                "FlashInfer does not support ragged verify in cuda graph; "
                "disable SGLANG_RAGGED_VERIFY_MODE for this configuration."
            )

        if in_capture:
            num_tokens = forward_batch.positions.numel()
            self._prepare_cuda_graph_metadata(bs, num_tokens, forward_mode, spec_info)

        if forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                decode_wrappers=self.decode_cuda_graph_metadata[bs],
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
                fixed_split_size=None,
                disable_split_kv=self.disable_cuda_graph_kv_split,
                req_pool_indices=req_pool_indices,
                plan=forward_batch.kv_loc_plan,
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
                plan=forward_batch.kv_loc_plan,
            )
        elif forward_mode.is_dllm_extend():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=seq_lens - self.dllm_config.block_size,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=not self.use_paged,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=None,
                plan=forward_batch.kv_loc_plan,
            )
        elif forward_mode.is_draft_extend_v2():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.draft_extend_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
                plan=forward_batch.kv_loc_plan,
            )
        elif forward_mode.is_extend():
            # Plain EXTEND under full prefill CUDA graph. plan() runs
            # out-of-graph against capture-stable wrappers; captured kernels
            # read the refreshed state at replay. Must stay below the
            # target-verify / draft-extend / dllm branches (also is_extend()).
            # Split-kv must stay on — its block_valid_mask is the only
            # early-exit for the captured fixed grid's padded/stale tiles.
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=forward_batch.extend_prefix_lens[:bs],
                prefill_wrappers=self.full_cg_prefill_wrappers,
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=None,
                plan=forward_batch.kv_loc_plan,
            )
        else:
            raise ValueError("Invalid forward mode")

        if in_capture and forward_mode.is_decode_or_idle():
            # fast_decode_plan needs _cached_module from the initial begin_forward
            # above, so install it only after that first plan has run.
            for w in self.decode_cuda_graph_metadata[bs]:
                w.begin_forward = partial(fast_decode_plan, w)

        if (
            in_capture
            and forward_mode.is_draft_extend_v2()
            and self.prefill_backend == "fa2"
            # Host-rebuilt layout only matches full attention (single wrapper);
            # SWA/cross-attn keep the plain plan().
            and self.dispatch_reason is None
        ):
            # Like decode: swap in fast_prefill_plan for replay, after the real
            # plan() above set up _cached_module (host metadata supplied per-replay
            # in call_begin_forward).
            for w in self.draft_extend_cuda_graph_metadata[bs]:
                w.begin_forward = partial(fast_prefill_plan, w)

        if (
            in_capture
            and forward_mode.is_target_verify()
            and spec_info is not None
            and spec_info.spec_input_type == SpecInputType.DFLASH_VERIFY
            and getattr(spec_info, "custom_mask", None) is None
            and self.prefill_backend == "fa2"
            # Host-rebuilt layout only matches full attention (single wrapper);
            # SWA/cross-attn keep the plain plan().
            and self.dispatch_reason is None
        ):
            # DFLASH target-verify replays are shape-static per
            # (bs, draft_token_num): qo_indptr is a constant arange stride of
            # num_tokens_per_req, and the batch carries seq_lens_cpu =
            # prefix + draft_token_num (dspark_draft._run_forward /
            # dspark_verify.run_non_compact / dflash_worker_v2 all add the
            # verify window host-side), which equals the device kv length
            # generate_attn_arg_prefill produces. The host-kwargs assembly in
            # call_begin_forward therefore applies verbatim; installing the
            # sync-free plan removes three blocking .to("cpu") reads per
            # replay that otherwise stall the CPU behind the in-flight graph.
            # EAGLE target-verify keeps the plain plan(): its spec input is
            # not DFLASH_VERIFY, and this branch keys off the capture-time
            # spec_info of these per-bs wrappers.
            for w in self.prefill_cuda_graph_metadata[bs]:
                w.begin_forward = partial(fast_prefill_plan, w)

        # Refill the SWA write-target buffer from the live out_cache_loc before
        # replay (bound onto the metadata at capture below).
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            n = forward_batch.out_cache_loc.shape[0]
            self.cuda_graph_swa_out_cache_loc[n:].zero_()
            if in_capture:
                self.cuda_graph_swa_out_cache_loc[:n].zero_()
            else:
                self.cuda_graph_swa_out_cache_loc[:n].copy_(
                    self.kv_index_translator.write_ids(
                        forward_batch, IdSpaceKind.SLIDING_WINDOW
                    )
                )
            if in_capture:
                self.forward_metadata.swa_out_cache_loc = (
                    self.cuda_graph_swa_out_cache_loc[:n]
                )

    def _prepare_dequant_workspace_metadata_for_extend(
        self, forward_batch: ForwardBatch, use_ragged: bool = False
    ):
        """Prepare FlashInfer metadata for an FP4 dequant workspace.

        Some FP4 recipes store packed KV but expose an FP8 workspace to
        FlashInfer prefill. This builds the workspace page table, exact paged
        lengths, and CPU request ids needed to populate that workspace before
        the prefill kernel runs.
        """
        self.dq_page_table = None
        self.dq_paged_kernel_lens = None
        self.cpu_req_pool_indices = None
        if not (
            self.prefill_uses_dequant_workspace
            and forward_batch.forward_mode.is_extend_without_speculative()
        ):
            return

        # Ragged prefill handles current-chunk K/V with raw tensors, so the
        # paged side only contains cached prefix lengths. Non-ragged prefill
        # uses the dequant workspace for prefix + current chunk, so it needs
        # full sequence lengths. These CPU length containers may arrive as
        # Python lists or CPU tensors depending on the metadata builder.
        paged_seq_lens_cpu = (
            forward_batch.extend_prefix_lens_cpu
            if use_ragged
            else forward_batch.seq_lens_cpu
        )
        raw_paged_seq_lens = (
            paged_seq_lens_cpu
            if isinstance(paged_seq_lens_cpu, list)
            else paged_seq_lens_cpu.tolist()
        )
        paged_seq_lens = [
            int(seq_len.item()) if isinstance(seq_len, torch.Tensor) else int(seq_len)
            for seq_len in raw_paged_seq_lens
        ]
        if sum(paged_seq_lens) <= 0:
            self.cpu_req_pool_indices = forward_batch.req_pool_indices.to(
                "cpu", non_blocking=True
            )
            return

        # dq_buffer layout is page-aligned: each request occupies
        # ceil(seq_len/page_size)*page_size slots, starting after a page_size
        # dummy prefix. dq_page_table maps only actual token positions and skips
        # padding gaps; dq_paged_kernel_lens stores real lengths so FlashInfer
        # causal offsets use seq_len - q_len, not page_align(seq_len) - q_len.
        seq_lens_with_scratch = paged_seq_lens + [256]
        starts = []
        next_start = self.page_size
        for seq_len in seq_lens_with_scratch:
            starts.append(next_start)
            padded_len = (
                (seq_len + self.page_size - 1) // self.page_size
            ) * self.page_size
            next_start += padded_len

        device = forward_batch.req_pool_indices.device
        indices = [
            torch.arange(start, start + seq_len, device=device, dtype=torch.int32)
            for start, seq_len in zip(starts, seq_lens_with_scratch)
            if seq_len > 0
        ]
        self.dq_page_table = torch.cat(indices) if indices else None
        self.dq_paged_kernel_lens = torch.tensor(
            paged_seq_lens,
            dtype=torch.int32,
            device=device,
        )
        self.cpu_req_pool_indices = forward_batch.req_pool_indices.to(
            "cpu", non_blocking=True
        )

    def _kv_write_scales(self, layer: RadixAttention):
        if self.kv_cache_quant_method.needs_global_scale():
            return None, None
        return layer.k_scale, layer.v_scale

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        swa_out_cache_loc = None
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            swa_out_cache_loc = self.kv_index_translator.write_ids(
                forward_batch, IdSpaceKind.SLIDING_WINDOW
            )

        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                decode_wrappers=self.decode_wrappers,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
                fixed_split_size=self.decode_split_tile_size,
                disable_split_kv=False,
                req_pool_indices=forward_batch.req_pool_indices,
                plan=forward_batch.kv_loc_plan,
            )
            self.forward_metadata = DecodeMetadata(
                self.decode_wrappers, swa_out_cache_loc=swa_out_cache_loc
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
                plan=forward_batch.kv_loc_plan,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_verify,
                False,
                False,
                swa_out_cache_loc=swa_out_cache_loc,
            )
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            # The planner builds qo_indptr as cumsum(seq_lens - prefix_lens), while the batch's
            # token tensor is sized by extend_lens. mix_with_verify keeps those consistent at the
            # seam (its own check passes), so a disagreement here is introduced between the merge
            # and this point -- report the per-row arrays rather than leave flashinfer to compare
            # two totals several frames later.
            if os.environ.get("SGLANG_MIXED_SPEC") == "1" and prefix_lens is not None:
                _qo_from_plan = (forward_batch.seq_lens - prefix_lens)
                _plan_sum = int(_qo_from_plan.sum())
                _tok_n = (
                    int(forward_batch.input_ids.numel())
                    if forward_batch.input_ids is not None
                    else -1
                )
                if _tok_n >= 0 and _plan_sum != _tok_n:
                    logger.error(
                        "PLANWIDTH MISMATCH cumsum(seq_lens-prefix)=%d input_ids=%d "
                        "extend_num_tokens=%s | tail seq_lens=%s prefix=%s qo=%s "
                        "extend_seq_lens=%s",
                        _plan_sum,
                        _tok_n,
                        getattr(forward_batch, "extend_num_tokens", None),
                        forward_batch.seq_lens[-4:].tolist(),
                        prefix_lens[-4:].tolist(),
                        _qo_from_plan[-4:].tolist(),
                        (
                            forward_batch.extend_seq_lens[-4:].tolist()
                            if forward_batch.extend_seq_lens is not None
                            else None
                        ),
                    )

            # FA3/FA2 row partition for the SM-pre-assigned split kernel. Computed HERE, in
            # upstream's own metadata hook, from upstream's own per-row lengths -- so
            # forward_extend stays a dispatch rather than becoming a planner. Carried on the
            # backend (not on ForwardBatch) because it is backend state, and left as None when
            # the split is off so every existing path is untouched.
            # The partition itself is `mixed_dispatch.split_classes`, whose batch-level
            # decode-population guard is validated against the measured boundary surface
            # (tasks/fused_specdec_workload/boundary_28x4x128.json: 11/25 -> 23/25).
            self.fused_split_mask = None
            if (
                os.environ.get("SGLANG_FUSED_FA3_SM_SPLIT", "0") != "0"
                and getattr(self, "indices_updater_prefill", None) is not None
            ):
                try:
                    from sglang.srt.layers.attention.mixed_dispatch import split_classes

                    _qo = (forward_batch.extend_seq_lens_cpu
                           if forward_batch.extend_seq_lens_cpu is not None else None)
                    if _qo is not None and forward_batch.seq_lens_cpu is not None:
                        # Head counts and dtypes live on the indices updater, not on the
                        # backend. Reading them off `self` silently skipped the partition on
                        # every step (78/78 on the first GPU gate).
                        _iu = self.indices_updater_prefill
                        _gqa = max(1, _iu.num_qo_heads // _iu.num_kv_heads)
                        _m, _qt, _kvt = split_classes(
                            _qo, forward_batch.seq_lens_cpu.tolist(), _gqa
                        )
                        # Store the mask and the tile counts the launch budget is derived from.
                        self.fused_split_mask = (_m, _qt, _kvt)
                        # Build the index tensors and plan the FA2 wrapper once for the
                        # whole step; forward_extend then runs per layer with no host work.
                        if not self._fused_split_plan(forward_batch):
                            self.fused_split_mask = None
                except Exception as _e:
                    # A partition failure must not degrade the ordinary path silently; log and
                    # fall through to the standard single-wrapper dispatch.
                    self.fused_split_mask = None
                    logger.warning("fused split partition skipped: %s", _e)

            # Disable ragged wrapper and ensure prefix handling for multimodal and multi-item scoring
            if self.is_multimodal or self.enable_mis:
                # use_ragged = False: Multi-item scoring requires the paged wrapper because:
                # 1. Ragged wrapper doesn't support the specialized multi-item parameters
                #    (prefix_len_ptr, token_pos_in_items_ptr, etc.)
                # 2. Paged wrapper provides better control over attention masking needed
                #    for respecting item boundaries in multi-item sequences
                # 3. Custom masking logic conflicts with ragged wrapper's assumptions
                use_ragged = False
                extend_no_prefix = False
            else:
                use_ragged = (
                    not self.enable_deterministic
                    and not is_in_tc_piecewise_cuda_graph()
                    and not self.use_paged
                )
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)

            # Process multi-item scoring in attention backend instead of ForwardBatch
            multi_item_params = MultiItemScoringParams()
            if self.enable_mis:
                # Use new backend-specific implementation
                multi_item_params = self._process_multi_item_scoring(forward_batch)

            self._prepare_dequant_workspace_metadata_for_extend(
                forward_batch, use_ragged
            )

            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens,
                prefill_wrappers=self.prefill_wrappers_paged,
                use_ragged=use_ragged,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=None,
                fixed_split_size=self.prefill_split_tile_size,
                multi_item_params=multi_item_params,
                cross_attention_custom_mask=forward_batch.cross_attention_custom_mask,
                extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
                custom_kv_indices=self.dq_page_table,
                plan=forward_batch.kv_loc_plan,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_paged,
                use_ragged,
                extend_no_prefix,
                multi_item_params,
                swa_out_cache_loc=swa_out_cache_loc,
            )

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

        # SWA write-target buffer; refilled and bound onto forward_metadata in
        # init_forward_metadata_out_graph before each replay.
        self.cuda_graph_swa_out_cache_loc = (
            torch.zeros(max_num_tokens, dtype=torch.int64, device="cuda")
            if self.use_sliding_window_kv_pool
            else None
        )

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

    def _create_decode_wrappers(self, bs: int, num_tokens: int) -> list:
        return [
            BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                backend=self.decode_backend,
                use_cuda_graph=True,
                use_tensor_cores=self.decode_use_tensor_cores,
                paged_kv_indptr_buffer=self.kv_indptr[i][: num_tokens + 1],
                paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                paged_kv_last_page_len_buffer=self.kv_last_page_len[:num_tokens],
            )
            for i in range(self.num_wrappers)
        ]

    def get_cuda_graph_decode_wrappers(
        self,
        *,
        bs: int,
        num_tokens: int,
    ) -> list:
        wrappers = self.decode_cuda_graph_metadata.get(bs)
        if wrappers is None:
            self._prepare_cuda_graph_metadata(
                bs,
                num_tokens,
                ForwardMode.DECODE,
                spec_info=None,
            )
            wrappers = self.decode_cuda_graph_metadata[bs]
        return wrappers

    def _create_prefill_wrappers(self, bs: int, use_custom_mask: bool = False) -> list:
        # FlashInfer's prefill wrapper decides mask mode based on whether
        # `custom_mask_buf` is initialized (not whether a custom mask is provided).
        # For cases like DFLASH draft (ENCODER_ONLY / non-causal) we do NOT use a
        # custom mask, so we must avoid initializing `custom_mask_buf`, otherwise
        # FlashInfer will treat the (zero) buffer as a real mask and block attention.
        wrappers = []
        for i in range(self.num_wrappers):
            extra = (
                {
                    "custom_mask_buf": self.cuda_graph_custom_mask,
                    "mask_indptr_buf": self.cuda_graph_qk_indptr[i][: bs + 1],
                }
                if use_custom_mask
                else {}
            )
            wrappers.append(
                BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_cuda_graph=True,
                    backend=self.prefill_backend,
                    qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                    paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                    paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                    paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                    **extra,
                )
            )
        return wrappers

    @staticmethod
    def _full_cg_prefill_workspace_bytes(
        num_slots: int,
        max_num_tokens: int,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
    ) -> int:
        """Split-kv worst-case float-workspace demand for the plain-EXTEND
        cudagraph wrappers, mirroring flashinfer's PrefillPlan sizing
        (scheduler.cuh, enable_cuda_graph=True) for the largest captured
        bucket:

          cta_tile_q = FA2DetermineCtaTileQ(max packed qo len, head_dim)
          tiles      = ceil(max_rows * gqa / cta_tile_q) + batch_size - 1
          padded     = max(2 * num_SMs / num_kv_heads, tiles)
          tmp_v      = num_qo_heads * padded * cta_tile_q * head_dim * fp32
          tmp_s      = num_qo_heads * padded * cta_tile_q * fp32

        Split-kv must stay enabled for these wrappers — its
        block_valid_mask is what lets the padded/stale tiles of the fixed
        captured grid exit early at replay; without it every replay
        re-runs capture-sized attention (measured ~6.5 ms/layer). If a
        future flashinfer outgrows the margin, plan() fails loudly at
        startup ("Increase the workspace buffer size").
        """
        gqa_group_size = num_qo_heads // num_kv_heads
        max_qo_len = (max_num_tokens - num_slots + 1) * gqa_group_size
        if max_qo_len > 64 and head_dim < 256:
            cta_tile_q = 128
        elif max_qo_len > 16:
            cta_tile_q = 64
        else:
            cta_tile_q = 16
        tiles = -(-max_num_tokens * gqa_group_size // cta_tile_q) + num_slots - 1
        num_sm = torch.cuda.get_device_properties(device).multi_processor_count
        padded_batch_size = max((2 * num_sm) // num_kv_heads, tiles)
        per_row = num_qo_heads * padded_batch_size * cta_tile_q * 4
        tmp_v = per_row * head_dim
        tmp_s = per_row
        return int((tmp_v + tmp_s) * FULL_CG_PREFILL_WORKSPACE_MARGIN)

    def _create_full_cg_prefill_wrappers(
        self, num_slots: int, max_num_tokens: int
    ) -> list:
        """Wrappers for plain EXTEND captured under a full prefill CUDA
        graph. plan() must keep its internal state at capture-stable
        addresses (use_cuda_graph=True); the decode-side cuda-graph
        wrappers permanently pin the shared workspace via their own
        plans, so these get a dedicated workspace sized from the largest
        captured bucket. The request-slot count is fixed at capture (the
        runner pads real batches up to it with zero-length sentinel
        requests); kv indices cover up to num_slots sequences of
        max_context_len.
        """
        device = self.workspace_buffer.device
        self.full_cg_prefill_req_slots = num_slots
        upd = self.indices_updater_prefill
        workspace_bytes = self._full_cg_prefill_workspace_bytes(
            num_slots,
            max_num_tokens,
            num_qo_heads=upd.num_qo_heads,
            num_kv_heads=upd.num_kv_heads,
            head_dim=upd.head_dim,
            device=device,
        )
        logger.info(
            "Full-CG prefill workspace: %.0f MB (max bucket %d tokens, "
            "%d request slots)",
            workspace_bytes / (1024 * 1024),
            max_num_tokens,
            num_slots,
        )
        self.full_cg_prefill_workspace_buffer = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=device
        )
        self.full_cg_prefill_qo_indptr = [
            torch.zeros((num_slots + 1,), dtype=torch.int32, device=device)
            for _ in range(self.num_wrappers)
        ]
        self.full_cg_prefill_kv_indptr = [
            torch.zeros((num_slots + 1,), dtype=torch.int32, device=device)
            for _ in range(self.num_wrappers)
        ]
        # call_begin_forward materializes paged_kernel_lens_sum + 256
        # indices; size the fixed buffer for the worst case.
        self.full_cg_prefill_kv_indices = [
            torch.zeros(
                (num_slots * self.max_context_len + 256,),
                dtype=torch.int32,
                device=device,
            )
            for _ in range(self.num_wrappers)
        ]
        return [
            BatchPrefillWithPagedKVCacheWrapper(
                self.full_cg_prefill_workspace_buffer,
                "NHD",
                use_cuda_graph=True,
                backend=self.prefill_backend,
                qo_indptr_buf=self.full_cg_prefill_qo_indptr[i],
                paged_kv_indptr_buf=self.full_cg_prefill_kv_indptr[i],
                paged_kv_indices_buf=self.full_cg_prefill_kv_indices[i],
                paged_kv_last_page_len_buf=self.kv_last_page_len[:num_slots],
            )
            for i in range(self.num_wrappers)
        ]

    def _prepare_cuda_graph_metadata(
        self,
        bs: int,
        num_tokens: int,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> None:
        if forward_mode.is_decode_or_idle():
            decode_wrappers = self._create_decode_wrappers(bs, num_tokens)
            self.decode_cuda_graph_metadata[bs] = decode_wrappers
            self.forward_metadata = DecodeMetadata(decode_wrappers)
        elif forward_mode.is_target_verify() or forward_mode.is_dllm_extend():
            use_custom_mask = (
                forward_mode.is_target_verify()
                and spec_info is not None
                and getattr(spec_info, "custom_mask", None) is not None
            )
            prefill_wrappers = self._create_prefill_wrappers(bs, use_custom_mask)
            self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(
                prefill_wrappers, forward_mode.is_dllm_extend(), False
            )
        elif forward_mode.is_draft_extend_v2():
            # Draft-extend: causal paged prefill over the full sequence (no mask).
            prefill_wrappers = self._create_prefill_wrappers(bs, use_custom_mask=False)
            self.draft_extend_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
        elif forward_mode.is_extend():
            if self.full_cg_prefill_wrappers is None:
                self.full_cg_prefill_wrappers = self._create_full_cg_prefill_wrappers(
                    bs, num_tokens
                )
            self.forward_metadata = PrefillMetadata(
                self.full_cg_prefill_wrappers, False, False
            )
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def _fs_dev_buf(self, name, n, dtype, device):
        """A length-n view of a persistent device buffer, grown only when outgrown."""
        b = self._fs_dev.get(name)
        if b is None or b.numel() < n or b.dtype != dtype or b.device != device:
            b = torch.empty(max(n, 1024), dtype=dtype, device=device)
            self._fs_dev[name] = b
        return b[:n]

    def _fs_host_buf(self, name, n, dtype):
        """A length-n view of a persistent PINNED host buffer.

        Pinned and persistent together are what make the H2D copy safe to issue without a
        synchronise: the source outlives the copy because the pool owns it, unlike the
        per-step temporaries this replaced.
        """
        b = self._fs_host.get(name)
        if b is None or b.numel() < n or b.dtype != dtype:
            try:
                b = torch.empty(max(n, 1024), dtype=dtype, pin_memory=True)
            except RuntimeError:
                # Pinning can fail (no CUDA host allocator); fall back to pageable, which is
                # still persistent and still correct, just slower to copy.
                b = torch.empty(max(n, 1024), dtype=dtype)
            self._fs_host[name] = b
        return b[:n]

    def _fs_upload(self, name, arr, dtype, device):
        """Stage a numpy array through the pinned host buffer into the device buffer."""
        n = int(arr.shape[0])
        h = self._fs_host_buf(name, n, dtype)
        h.copy_(torch.from_numpy(arr.astype(_FS_NP[dtype], copy=False)))
        d = self._fs_dev_buf(name, n, dtype, device)
        d.copy_(h, non_blocking=True)
        return d

    def _fused_split_debug_check(self, where: str) -> None:
        """Synchronise, then report every split index tensor's actual device-side range.

        Four mechanism hypotheses for the IndexKernel.cu out-of-bounds assert were wrong in a
        row (shared workspace, launch-blocking, plan-copy lifetime, FA2 plan-array lifetime).
        Each was inferred from a traceback that names whatever CUDA call ran next, never the
        faulting op. This stops inferring: the synchronise pins the fault to a point in OUR
        code, and the min/max print the offending numbers.
        """
        pl = self.fused_split_plan
        if pl is None:
            return
        try:
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 - diagnostic path
            logger.error(
                "FUSED_SPLIT_DEBUG[%s] device had ALREADY faulted before validation: %s",
                where, e,
            )
            raise
        r2t = self.req_to_token_pool.req_to_token
        kv_sz = None
        for attr in ("size", "max_total_num_tokens"):
            v = getattr(self.token_to_kv_pool, attr, None)
            if isinstance(v, int):
                kv_sz = v
                break
        logger.error(
            "FUSED_SPLIT_DEBUG[%s] tok_total=%s r2t=%s kv_pool_size=%s",
            where, pl.get("tok_total"), tuple(r2t.shape), kv_sz,
        )
        named = [
            ("fa3_tok", pl.get("fa3_tok"), pl.get("tok_total")),
            ("fa2_tok", pl.get("fa2_tok"), pl.get("tok_total")),
            ("cu_q", pl.get("cu_q"), None),
            ("cache_seqlens", pl.get("cache_seqlens"), None),
            ("page_table", pl.get("page_table"), kv_sz),
        ]
        keep = pl.get("_keep") or ()
        for i, t in enumerate(keep):
            named.append((f"_keep[{i}]", t, kv_sz if i == 2 else None))
        for name, t, bound in named:
            if t is None or not torch.is_tensor(t) or t.numel() == 0:
                continue
            mn, mx = int(t.min()), int(t.max())
            bad = bound is not None and (mn < 0 or mx >= bound)
            logger.error(
                "FUSED_SPLIT_DEBUG[%s] %-14s n=%-8d min=%-12d max=%-12d bound=%-12s %s",
                where, name, t.numel(), mn, mx, bound,
                "OUT_OF_RANGE" if bad else "ok",
            )

    def _fused_split_plan(self, forward_batch: ForwardBatch) -> bool:
        """Build the split's index tensors and plan its FA2 wrapper ONCE per step.

        `forward_extend` runs per layer, so anything built there is paid ~num_layers times per
        step. All of this is layer-independent, so it belongs in the metadata hook -- the same
        division upstream already uses (plan per step, forward per layer).

        Returns True when a two-sided split is available for this step.
        """
        import numpy as np

        self.fused_split_plan = None
        mask, _q_tiles, _kv_tiles = self.fused_split_mask
        mask = np.asarray(mask, dtype=bool)
        fa3_rows, fa2_rows = np.nonzero(mask)[0], np.nonzero(~mask)[0]
        self.fused_split_plans += 1
        if len(fa3_rows) == 0 or len(fa2_rows) == 0:
            # One-sided partition: nothing to split. The decode-population guard refuses
            # fusion for pure long-prefill batches by design, so on a prefill-only workload
            # this is the common case and must not be read as a failure.
            self.fused_split_onesided += 1
            if self.fused_split_onesided % 200 == 1:
                # Say WHY, with the numbers the guard decided on. A bare "never fired" is
                # indistinguishable from a broken hook, and cost one full gate cycle.
                logger.info(
                    "FUSED_SPLIT_ONESIDED %d/%d: fa3=%d fa2=%d rows=%d "
                    "max_kv_tiles=%d need_decode_rows>=%d (guard: kv_tiles<=decode_rows/2)",
                    self.fused_split_onesided,
                    self.fused_split_plans,
                    len(fa3_rows),
                    len(fa2_rows),
                    len(mask),
                    int(max(_kv_tiles)) if len(_kv_tiles) else -1,
                    2 * int(max(_kv_tiles)) if len(_kv_tiles) else -1,
                )
            return False
        if self.fused_split_plans % 200 == 1:
            logger.info(
                "FUSED_SPLIT_PLAN ok %d: fa3=%d fa2=%d (onesided so far %d)",
                self.fused_split_plans,
                len(fa3_rows),
                len(fa2_rows),
                self.fused_split_onesided,
            )

        qo_lens = np.asarray(forward_batch.extend_seq_lens_cpu, dtype=np.int64)
        if len(qo_lens) != len(mask):
            # Partition computed for a different batch shape than the one that arrived
            # (e.g. a padded graph replay). Refuse rather than mis-slice.
            return False
        # Row-count agreement across every per-request array the plan indexes. The check above
        # only compares qo_lens against the mask, both derived from extend_seq_lens_cpu; the
        # plan also gathers forward_batch.seq_lens and req_pool_indices by the SAME row ids. If
        # those carry fewer rows (a padded extend_seq_lens_cpu, say), the gather runs off the end
        # and raises IndexKernel.cu's out-of-bounds assert -- a device-side fault that surfaces
        # at an unrelated later CUDA call, which is precisely how this presented.
        _rows = len(qo_lens)
        if (
            int(forward_batch.seq_lens.numel()) != _rows
            or int(forward_batch.seq_lens_cpu.numel()) != _rows
            or int(forward_batch.req_pool_indices.numel()) != _rows
        ):
            logger.error(
                "FUSED_SPLIT BOUNDS row-count disagreement: extend_seq_lens_cpu=%d "
                "seq_lens=%d seq_lens_cpu=%d req_pool_indices=%d -- refusing the split",
                _rows,
                int(forward_batch.seq_lens.numel()),
                int(forward_batch.seq_lens_cpu.numel()),
                int(forward_batch.req_pool_indices.numel()),
            )
            return False

        seq_lens_cpu = forward_batch.seq_lens_cpu.to(torch.int64)
        dev = forward_batch.seq_lens.device
        rp_cpu = forward_batch.req_pool_indices.tolist()

        tok_off = np.zeros(len(qo_lens) + 1, dtype=np.int64)
        np.cumsum(qo_lens, out=tok_off[1:])

        # These copies are SYNCHRONOUS on purpose. A non-blocking H2D copy requires the host
        # source to stay alive until the copy completes, but the sources here are Python
        # temporaries (`torch.from_numpy(np.concatenate(...))`, a local `out`) that are freed as
        # soon as the expression returns -- a use-after-free that copies garbage. The garbage
        # lands in INDEX tensors, so it surfaced as IndexKernel.cu's out-of-bounds assert at
        # whatever CUDA call came next, and only with CUDA_LAUNCH_BLOCKING unset: with blocking
        # on, the split ran 4501 times clean; with it off, it asserted after one call.
        # These are a few KB at most, so the sync costs nothing worth reclaiming. Making them
        # non-blocking again requires pinned staging buffers the plan owns across the step.
        def tok_idx(name, rows):
            a = np.concatenate([np.arange(tok_off[r], tok_off[r + 1]) for r in rows])
            return self._fs_upload(name, a, torch.int64, dev)

        def indptr(name, vals):
            a = np.zeros(len(vals) + 1, dtype=np.int32)
            np.cumsum(vals, out=a[1:])
            return self._fs_upload(name, a, torch.int32, dev)

        # Bounds checks at the point of construction. Every index this plan builds is consumed
        # by a torch indexing op (`qv[tok]`, `index_copy_`, `r2t[rows]`), and an out-of-range
        # value there raises a DEVICE-SIDE assert from IndexKernel.cu that surfaces at whatever
        # unrelated CUDA call happens next -- in practice a flashinfer plan() several frames
        # away, naming the victim rather than the cause. Checking here converts that into a
        # message that names the offending width.
        _tok_total = int(tok_off[-1])
        _q_rows = (
            int(forward_batch.input_ids.numel())
            if forward_batch.input_ids is not None
            else -1
        )
        if _q_rows >= 0 and _tok_total != _q_rows:
            logger.error(
                "FUSED_SPLIT BOUNDS sum(extend_seq_lens_cpu)=%d != input_ids=%d "
                "(rows=%d qo_lens tail=%s) -- refusing the split for this batch",
                _tok_total, _q_rows, len(qo_lens), qo_lens[-4:].tolist(),
            )
            return False
        _rp_max = max(rp_cpu) if rp_cpu else -1
        if _rp_max >= int(self.req_to_token_pool.req_to_token.shape[0]):
            logger.error(
                "FUSED_SPLIT BOUNDS req_pool_index max=%d >= req_to_token rows=%d",
                _rp_max, int(self.req_to_token_pool.req_to_token.shape[0]),
            )
            return False
        _max_kv_needed = int(seq_lens_cpu.max())
        if _max_kv_needed > int(self.req_to_token_pool.req_to_token.shape[1]):
            logger.error(
                "FUSED_SPLIT BOUNDS max seq_len=%d > req_to_token cols=%d",
                _max_kv_needed, int(self.req_to_token_pool.req_to_token.shape[1]),
            )
            return False

        r2t = self.req_to_token_pool.req_to_token
        # Head counts and dtypes live on the indices updater (see _fused_split_plan's caller).
        iu = self.indices_updater_prefill
        # FA3 side. page_table is indexed by REQ POOL index, not batch row: req_to_token is a
        # pool-sized table (upstream does the same at flashattention_backend.py:1028).
        max_k = int(seq_lens_cpu[fa3_rows].max())
        # Row ids staged through the pool; indexed by REQ POOL index, not batch row.
        fa3_t = self._fs_upload(
            "fa3_rows",
            np.asarray([rp_cpu[r] for r in fa3_rows], dtype=np.int64),
            torch.int64,
            dev,
        )
        # Gather into a PERSISTENT buffer: `r2t[rows, :max_k]` allocated a fresh tensor on every
        # step, and index_select with out= reuses one.
        _cols = int(r2t.shape[1])
        _pt = self._fs_dev_buf(
            "page_table", len(fa3_rows) * _cols, r2t.dtype, dev
        ).view(len(fa3_rows), _cols)
        torch.index_select(r2t, 0, fa3_t, out=_pt)
        page_table = _pt[:, :max_k]
        if self.page_size > 1:
            page_table = page_table // self.page_size
        # FA2 side, planned through flashinfer's public API with explicit indptr/indices, so no
        # second KVLocPlan is needed and the arrays come from the same req_to_token.
        n2 = len(fa2_rows)
        # One flat gather instead of a Python loop of device slices plus a cat: that loop ran
        # once per FA2 row every step and allocated a tensor per row plus the concatenation.
        _kvl = seq_lens_cpu[fa2_rows].numpy().astype(np.int64)
        _flat = (
            np.concatenate(
                [
                    np.arange(rp_cpu[r] * _cols, rp_cpu[r] * _cols + int(k))
                    for r, k in zip(fa2_rows, _kvl)
                ]
            )
            if n2
            else np.zeros(0, dtype=np.int64)
        )
        _flat_t = self._fs_upload("fa2_flat", _flat, torch.int64, dev)
        kv2_indices = self._fs_dev_buf("fa2_kv_idx", int(_flat.shape[0]), r2t.dtype, dev)
        torch.index_select(r2t.view(-1), 0, _flat_t, out=kv2_indices)
        if self.fused_fa2_wrapper is None:
            # DEDICATED workspace, never self.workspace_buffer. A flashinfer wrapper writes its
            # scheduler metadata into the workspace it was given, so planning this extra wrapper
            # mid-step into the SHARED buffer clobbers the plans of the wrappers already planned
            # for this step. The symptom was a device-side assert raised later, inside an
            # unrelated `plan` call reached from init_forward_metadata_out_graph, and it appeared
            # ONLY in runs where the split actually fired (FUSED_SPLIT_CALLS >= 1); with the
            # guard on, the split never fired and 151 merges ran clean. Upstream shares one
            # workspace across its wrappers safely because it plans them in a fixed order before
            # any of them runs -- this wrapper plans after that, so it needs its own.
            self.fused_fa2_workspace = torch.empty(
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                dtype=torch.uint8,
                device=forward_batch.seq_lens.device,
            )
            self.fused_fa2_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.fused_fa2_workspace, "NHD", backend=self.prefill_backend
            )
        # The plan arrays are BOUND TO NAMES and retained on the plan below, and the copy is
        # blocking. Both are required, for two separate reasons:
        #  * lifetime: this wrapper is planned once per step and then run once per LAYER, so the
        #    arrays must outlive this function. Passing them as inline temporaries let the
        #    caching allocator hand that memory to the next allocation while the wrapper was
        #    still using it.
        #  * ordering: with non_blocking=True flashinfer's internal copies may still be in
        #    flight when the tensors are released, so the copy can read reused memory. Garbage
        #    in an indptr then faults inside flashinfer's own plan maths -- which is why the
        #    traceback kept landing in _compute_page_mask_indptr.
        # Upstream can pass non_blocking=True here because its indptr buffers
        # (self.kv_indptr[...]) are preallocated and owned for the process lifetime; these are
        # built per step, so the same call is not safe for them.
        if FUSED_SPLIT_NO_FA2_WRAPPER:
            # Bisect mode: no extra wrapper. Record what the dispatch needs and stop here.
            self.fused_split_plan = dict(
                tok_total=_tok_total,
                fa3_tok=tok_idx("fa3_tok", fa3_rows),
                fa2_tok=tok_idx("fa2_tok", fa2_rows),
                cu_q=indptr("fa3_cu_q", qo_lens[fa3_rows]),
                max_q=int(qo_lens[fa3_rows].max()),
                page_table=page_table,
                cache_seqlens=self._fs_upload(
                    "fa3_kvlen",
                    seq_lens_cpu[fa3_rows].numpy().astype(np.int32),
                    torch.int32,
                    dev,
                ),
                n_fa3=len(fa3_rows),
                n_fa2=n2,
                no_fa2_wrapper=True,
            )
            if FUSED_SPLIT_DEBUG:
                self._fused_split_debug_check("plan")
            return True

        fa2_qo_indptr = indptr("fa2_qo", qo_lens[fa2_rows])
        fa2_kv_indptr = indptr("fa2_kv", seq_lens_cpu[fa2_rows].numpy())
        fa2_last_page = self.kv_last_page_len[:n2]
        self.fused_fa2_wrapper.begin_forward(
            fa2_qo_indptr,
            fa2_kv_indptr,
            kv2_indices,
            fa2_last_page,
            iu.num_qo_heads,
            iu.num_kv_heads,
            iu.head_dim,
            1,
            q_data_type=iu.q_data_type,
            kv_data_type=iu.data_type,
            non_blocking=False,
        )
        self.fused_split_plan = dict(
            # Retained so the planned FA2 wrapper's arrays outlive this call; see above.
            _keep=(fa2_qo_indptr, fa2_kv_indptr, kv2_indices, fa2_last_page),
            tok_total=_tok_total,
            fa3_tok=tok_idx("fa3_tok", fa3_rows),
            fa2_tok=tok_idx("fa2_tok", fa2_rows),
            cu_q=indptr("fa3_cu_q", qo_lens[fa3_rows]),
            max_q=int(qo_lens[fa3_rows].max()),
            page_table=page_table,
            cache_seqlens=self._fs_upload(
                "fa3_kvlen",
                seq_lens_cpu[fa3_rows].numpy().astype(np.int32),
                torch.int32,
                dev,
            ),
            n_fa3=len(fa3_rows),
            n_fa2=n2,
        )
        if FUSED_SPLIT_DEBUG:
            self._fused_split_debug_check("plan")
        return True

    def _fused_split_dispatch(
        self, q, layer, forward_batch, causal, logits_soft_cap, ref_wrapper=None, kv_cache=None
    ):
        """Run the FA3/FA2 SM-pre-assigned split for one layer, or return None.

        The two row subsets are DISJOINT, so the outputs combine by SCATTER (`index_copy_`) --
        unlike upstream's ragged+paged pair, which covers the same rows and merges via LSE.

        FA3's share of the GPU goes through upstream FA3's OWN public parameter, `sm_margin`
        ("can be tuned if some SMs are used for communication"), so the SM pre-assignment needs
        no private entry point on that side. Returning None is not a silent fallback for a
        broken split: it means this layer or batch is outside the split's verified scope, and
        the caller then takes the ordinary single-wrapper path.
        """
        pl = self.fused_split_plan
        if pl is None:
            return None
        # Features the split does not yet cover. Each is refused, never approximated: the
        # alternative is silently wrong attention, which a throughput A/B would not reveal.
        # TODO(swa): a sliding-window layer reads `self._swa_kv_pool` with its own cache locs
        #   (`forward_metadata.swa_out_cache_loc`), so the FA3 half needs that pool's page
        #   table rather than `req_to_token`, plus window_size=(sliding_window_size, 0) as
        #   derived in flashattention_backend.py:2030.
        # TODO(fp8kv): FA3 takes k_descale/v_descale while this backend carries
        #   layer.k_scale_float; the two conventions are reciprocal-or-not unverified here, so
        #   routing one into the other needs a reference check before it can be trusted.
        if (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        ) or layer.k_scale_float is not None or layer.v_scale_float is not None:
            return None
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            return None
        if (
            self.forward_metadata.multi_item_params is not None
            and self.forward_metadata.multi_item_params.is_enabled()
        ):
            # Item-aware masking takes precedence over the split's uniform causal mask.
            return None

        hq, hd = layer.tp_q_head_num, layer.head_dim
        qv = q.view(-1, hq, hd)
        # The plan's token indices were built from extend_seq_lens_cpu, but the tensor actually
        # indexed is this layer's q. If the two extents disagree -- a padded graph replay, say --
        # every index op here can run out of bounds, which raises a device-side assert from
        # IndexKernel.cu at whatever CUDA call comes next rather than at the fault. Compared as
        # ints (the plan carries its extent) so this costs no sync.
        if qv.shape[0] != pl["tok_total"]:
            logger.error(
                "FUSED_SPLIT BOUNDS q rows=%d != plan token extent=%d -- refusing the split "
                "for this layer (layer=%d)",
                int(qv.shape[0]), pl["tok_total"], layer.layer_id,
            )
            return None
        if FUSED_SPLIT_DEBUG:
            self._fused_split_debug_check(f"dispatch:L{layer.layer_id}")
        o = torch.empty_like(qv)

        from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache

        # The pool hangs off the BACKEND, not off ForwardBatch (upstream's own forward_extend
        # does `pool = self.token_to_kv_pool`). Reading it from forward_batch raised
        # AttributeError the first time the split actually fired.
        #
        # Use upstream's OWN accessor, not a reshape. `paged_kv_view` splits pages from the
        # buffer's STRIDES: a slot-major pool buffer need not be densely packed, and
        # flashattention_backend's own assertion spells out that such a view "carries the pool's
        # slot stride ... which no reshape of the head dimension can express". A plain
        # `.view(-1, page_size, heads, dim)` therefore silently reinterprets the buffer, so FA3
        # reads and WRITES at wrong offsets -- out of bounds of the real rows. That corrupts
        # whatever memory follows, which later surfaced as an IndexKernel.cu out-of-bounds
        # assert in unrelated code, and it produced correct numerics under
        # CUDA_LAUNCH_BLOCKING=1 only because the timing changed what occupied that memory.
        from sglang.srt.mem_cache.layout.paged_view import paged_kv_view

        _k_raw, _v_raw = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k_buf = paged_kv_view(_k_raw, self.page_size, layer.tp_k_head_num, hd)
        v_buf = paged_kv_view(_v_raw, self.page_size, layer.tp_v_head_num, layer.v_head_dim)
        o_fa3 = flash_attn_with_kvcache(
            q=qv[pl["fa3_tok"]],
            k_cache=k_buf,
            v_cache=v_buf,
            page_table=pl["page_table"],
            cache_seqlens=pl["cache_seqlens"],
            cu_seqlens_q=pl["cu_q"],
            max_seqlen_q=pl["max_q"],
            softmax_scale=layer.scaling,
            causal=causal,
            softcap=logits_soft_cap if logits_soft_cap else 0.0,
            sm_margin=self.fused_sm_margin,
        )
        o.index_copy_(0, pl["fa3_tok"], o_fa3.view(-1, hq, hd).to(o.dtype))

        if pl.get("no_fa2_wrapper"):
            if ref_wrapper is None or kv_cache is None:
                return None
            ref_all = ref_wrapper.forward(
                qv,
                kv_cache,
                causal=causal,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            ).view(-1, hq, hd)
            o.index_copy_(0, pl["fa2_tok"], ref_all[pl["fa2_tok"]].to(o.dtype))
            self.fused_split_calls += 1
            if self.fused_split_calls % 500 == 1:
                logger.info(
                    "FUSED_SPLIT_CALLS %d (NOFA2 bisect: fa3_rows=%d fa2_rows=%d)",
                    self.fused_split_calls, pl["n_fa3"], pl["n_fa2"],
                )
            return o

        o_fa2 = self.fused_fa2_wrapper.forward(
            qv[pl["fa2_tok"]],
            # Already resolved by the caller, including the fp4-dequant workspace case.
            kv_cache,
            causal=causal,
            sm_scale=layer.scaling,
            logits_soft_cap=logits_soft_cap,
        )
        o.index_copy_(0, pl["fa2_tok"], o_fa2.view(-1, hq, hd).to(o.dtype))

        # Counted here, ahead of the verify block below that reads it.
        self.fused_split_calls += 1

        if FUSED_SPLIT_VERIFY and ref_wrapper is not None and kv_cache is not None:
            # Self-check against the ordinary planned wrapper, computed in THIS step on THESE
            # inputs. A server-level ON-vs-OFF text diff cannot settle this: greedy output
            # under a concurrent workload does not reproduce run-to-run even for one unchanged
            # binary, so that comparison has no noise floor (the first GPU gate came back
            # INCONCLUSIVE for exactly this reason). Same step, same inputs, and the reference
            # is the read-only call the OFF path already makes.
            ref = ref_wrapper.forward(
                qv,
                kv_cache,
                causal=causal,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            ).view(-1, hq, hd)
            # Scale-normalised, not elementwise-relative: a clamped relative error returns
            # ~1e5 on any near-zero activation and says nothing.
            denom = ref.abs().max().clamp_min(1e-6)
            err_all = ((o - ref).abs().max() / denom).item()
            err_fa3 = (
                (o[pl["fa3_tok"]] - ref[pl["fa3_tok"]]).abs().max() / denom
            ).item()
            err_fa2 = (
                (o[pl["fa2_tok"]] - ref[pl["fa2_tok"]]).abs().max() / denom
            ).item()
            self.fused_split_err = max(self.fused_split_err, err_all)
            if err_all > FUSED_SPLIT_TOL:
                logger.error(
                    "FUSED_SPLIT_VERIFY FAIL layer=%d err=%.5f (fa3=%.5f fa2=%.5f) "
                    "rows fa3=%d fa2=%d",
                    layer.layer_id, err_all, err_fa3, err_fa2, pl["n_fa3"], pl["n_fa2"],
                )
            elif self.fused_split_calls % 500 == 0:
                logger.info(
                    "FUSED_SPLIT_VERIFY ok layer=%d err=%.5f (fa3=%.5f fa2=%.5f) worst=%.5f",
                    layer.layer_id, err_all, err_fa3, err_fa2, self.fused_split_err,
                )

        # Always-on composition counter. A serving A/B whose split never fired is
        # indistinguishable from a null result, and a counter logged only under a debug flag
        # has already produced one false pass on this project.
        if self.fused_split_calls % 500 == 1:
            logger.info(
                "FUSED_SPLIT_CALLS %d (fa3_rows=%d fa2_rows=%d sm_margin=%d onesided=%d)",
                self.fused_split_calls,
                pl["n_fa3"],
                pl["n_fa2"],
                self.fused_sm_margin,
                self.fused_split_onesided,
            )
        # The caller flattens to [-1, hq*hd]; hand back the head-shaped tensor.
        return o

    @debug_kernel_api
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

        logits_soft_cap = layer.logit_cap

        q = q.contiguous()

        assert not (self.prefill_uses_dequant_workspace and layer.is_cross_attention), (
            "FP4 dequant KV cache is not supported for cross-attention"
        )

        # We perform dequant for chunk prefill/cache reuse.
        pool = self.token_to_kv_pool
        if self.prefill_uses_dequant_workspace:
            kv_cache = pool.get_flashinfer_dequant_workspace_kv_buffer(
                layer,
                self.req_to_token_pool.req_to_token,
                self.cpu_req_pool_indices,
                forward_batch.extend_prefix_lens_cpu,
                forward_batch.extend_seq_lens_cpu,
                self.page_size,
                prepare_workspace=self.dq_page_table is not None,
                use_ragged=self.forward_metadata.use_ragged,
                k_cur=k,
                v_cur=v,
            )
        else:
            kv_cache = pool.get_kv_buffer(layer.layer_id)

        # use paged attention
        if not self.forward_metadata.use_ragged:
            if k is not None and save_kv_cache:
                assert v is not None
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc.for_layer(
                        forward_batch,
                        layer,
                        swa_loc=self.forward_metadata.swa_out_cache_loc,
                    ),
                    k,
                    v,
                    *self._kv_write_scales(layer),
                )

            causal = (
                not layer.is_cross_attention
                and layer.attn_type != AttentionType.ENCODER_ONLY
            )
            o = None
            if getattr(self, "fused_split_mask", None) is not None:
                o = self._fused_split_dispatch(
                    q,
                    layer,
                    forward_batch,
                    causal,
                    logits_soft_cap,
                    ref_wrapper=prefill_wrapper_paged,
                    kv_cache=kv_cache,
                )
            if o is None:
                o = prefill_wrapper_paged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    kv_cache,
                    causal=causal,
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
            # If `k`/`v` are not explicitly provided, fall back to the KV cache stored in
            # `self.token_to_kv_pool` for this layer. This enables attention over
            # previously cached context without re-materializing KV tensors (e.g., the
            # IQuestLoopCoder path uses token_to_kv_pool as the KV source).
            if k is None and v is None:
                assert not self.prefill_uses_dequant_workspace, (
                    "KV cache must be provided for ragged attention when using FP4 dequant KV cache"
                )
                k = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)[0]
                v = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)[1]
            causal = True
            if (
                layer.is_cross_attention
                or layer.attn_type == AttentionType.ENCODER_ONLY
            ):
                causal = False
            if not self.is_dllm_model and layer.attn_type == AttentionType.ENCODER_ONLY:
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
                swa_window_left = (
                    layer.sliding_window_size
                    if not (
                        self.forward_metadata.multi_item_params
                        and self.forward_metadata.multi_item_params.is_enabled()
                    )
                    else -1
                )
                o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=causal,
                    sm_scale=layer.scaling,
                    window_left=swa_window_left,
                    logits_soft_cap=logits_soft_cap,
                )
                o2, s2 = prefill_wrapper_paged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    kv_cache,
                    causal=False,
                    sm_scale=layer.scaling,
                    window_left=swa_window_left,
                    logits_soft_cap=logits_soft_cap,
                    # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
                    k_scale=layer.k_scale_float,
                    v_scale=layer.v_scale_float,
                )

                o, _ = _safe_merge_state(o1, s1, o2, s2)

            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc.for_layer(
                        forward_batch,
                        layer,
                        swa_loc=self.forward_metadata.swa_out_cache_loc,
                    ),
                    k,
                    v,
                    *self._kv_write_scales(layer),
                )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    @debug_kernel_api
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

        if k is not None:
            assert v is not None
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc.for_layer(
                        forward_batch,
                        layer,
                        swa_loc=self.forward_metadata.swa_out_cache_loc,
                    ),
                    k,
                    v,
                    *self._kv_write_scales(layer),
                )

        if self.decode_uses_dequant_workspace:
            kv_cache = (
                self.token_to_kv_pool.get_flashinfer_decode_dequant_workspace_kv_buffer(
                    layer,
                    self.req_to_token_pool.req_to_token,
                    forward_batch.req_pool_indices,
                    (
                        forward_batch.seq_lens_cpu
                        if forward_batch.seq_lens_cpu is not None
                        else forward_batch.seq_lens
                    ),
                )
            )
        else:
            kv_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        # Call the wrapped function
        o = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            kv_cache,
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
        # Plan with the max per-layer head count: FlashInfer bakes num_qo_heads
        # into the plan, and layers running more heads than planned are
        # silently corrupted. Over-planning is safe.
        self.num_qo_heads = (
            model_runner.model_config.get_max_num_attention_heads()
            // get_parallel().attn_tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size, get_parallel().attn_dcp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = attn_backend.flashinfer_kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self._swa_kv_pool = attn_backend._swa_kv_pool

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
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
        *,
        plan: KVLocPlan,
        req_pool_indices: torch.Tensor,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
        *,
        plan: KVLocPlan,
        req_pool_indices: torch.Tensor,
    ):
        decode_wrappers = decode_wrappers or self.decode_wrappers
        self.call_begin_forward(
            decode_wrappers[0],
            seq_lens,
            seq_lens_sum,
            self.kv_indptr[0],
            None,
            spec_info,
            seq_lens_cpu,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
            plan=plan,
            req_pool_indices=req_pool_indices,
        )

    def update_sliding_window(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
        *,
        plan: KVLocPlan,
        req_pool_indices: torch.Tensor,
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

            use_sliding_window_kv_pool = (
                wrapper_id == 0 and self._swa_kv_pool is not None
            )

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                paged_kernel_lens_tmp,
                paged_kernel_lens_sum_tmp,
                self.kv_indptr[wrapper_id],
                kv_start_idx_tmp,
                spec_info,
                seq_lens_cpu=seq_lens_cpu_tmp,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
                fixed_split_size=fixed_split_size,
                disable_split_kv=disable_split_kv,
                plan=plan,
                req_pool_indices=req_pool_indices,
            )

    def update_cross_attention(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
        *,
        plan: KVLocPlan,
        req_pool_indices: torch.Tensor,
    ):
        # Cache encoder_lens on CPU to avoid GPU→CPU transfer per call
        encoder_lens_cpu = encoder_lens.cpu() if encoder_lens is not None else None
        for wrapper_id in range(2):
            if wrapper_id == 0:
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
                kv_lens_cpu = seq_lens_cpu
            else:
                # Cross-attention: attend to encoder tokens only
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                seq_lens_sum = encoder_lens.sum().item()
                kv_lens_cpu = encoder_lens_cpu

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                paged_kernel_lens,
                seq_lens_sum,
                self.kv_indptr[wrapper_id],
                kv_start_idx,
                spec_info,
                seq_lens_cpu=kv_lens_cpu,
                fixed_split_size=fixed_split_size,
                disable_split_kv=disable_split_kv,
                plan=plan,
                req_pool_indices=req_pool_indices,
            )

    def call_begin_forward(
        self,
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
        *,
        plan: KVLocPlan,
        req_pool_indices: torch.Tensor,
    ):
        # Unified SWA wrapper-0: gather from the swa canonical directly -- its
        # entries are already swa-side physical ids, so the in-place
        # full->swa translate below must not run on top of them.
        translator = self.attn_backend.kv_index_translator
        use_swa_source = use_sliding_window_kv_pool and translator.reads_are_translated
        read_kind = IdSpaceKind.SLIDING_WINDOW if use_swa_source else IdSpaceKind.FULL
        if spec_info is None or getattr(spec_info, "kv_indptr", None) is None:
            bs = len(paged_kernel_lens)
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            if wrapper.is_cuda_graph_enabled:
                # Directly write to the cuda graph input buffer
                kv_indices = wrapper._paged_kv_indices_buf
            else:
                kv_indices = torch.empty(
                    paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
                )

            translator.pack_read_stream(
                plan,
                req_pool_indices=req_pool_indices,
                seq_lens=paged_kernel_lens,
                indptr=kv_indptr,
                out=kv_indices,
                kv_start_idx=kv_start_idx,
                kind=read_kind,
            )
        else:
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
            bs = kv_indptr.shape[0] - 1

        if use_sliding_window_kv_pool and not use_swa_source:
            assert self._swa_kv_pool is not None
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self._swa_kv_pool.translate_loc_from_full_to_swa(
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
            hasattr(wrapper.begin_forward, "func")
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
                1,
                data_type=self.data_type,
                q_data_type=self.q_data_type,
                non_blocking=True,
                fixed_split_size=fixed_split_size,
                disable_split_kv=(
                    disable_split_kv if disable_split_kv is not None else False
                ),
                global_override_indptr_cpu=global_override_indptr_cpu,
            )
        else:
            # When using original begin_forward, don't pass global_override_indptr_cpu
            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,
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
        # Plan with the max per-layer head count: FlashInfer bakes num_qo_heads
        # into the plan, and layers running more heads than planned are
        # silently corrupted. Over-planning is safe.
        self.num_qo_heads = (
            model_runner.model_config.get_max_num_attention_heads()
            // get_parallel().attn_tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size, get_parallel().attn_dcp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = attn_backend.flashinfer_kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend
        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self._swa_kv_pool = attn_backend._swa_kv_pool
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

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
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
        custom_kv_indices: Optional[torch.Tensor] = None,
        *,
        plan: KVLocPlan,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
        custom_kv_indices: Optional[torch.Tensor] = None,
        *,
        plan: KVLocPlan,
    ):
        if use_ragged:
            assert prefix_lens is not None
            paged_kernel_lens = prefix_lens
            if extend_prefix_lens_cpu is not None:
                # Host-known prefix lens; avoids a per-step D2H sync.
                paged_kernel_lens_sum = sum(extend_prefix_lens_cpu)
            else:
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

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
            seq_lens_cpu=seq_lens_cpu,
            custom_kv_indices=custom_kv_indices,
            plan=plan,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
        custom_kv_indices: Optional[torch.Tensor] = None,
        *,
        plan: KVLocPlan,
    ):
        if custom_kv_indices is not None:
            raise RuntimeError(
                "NVFP4 custom KV indices are only supported by the single-wrapper FlashInfer path."
            )
        if prefix_lens is None:
            num_accept_tokens = getattr(spec_info, "num_accept_tokens", None)
            # Spec verify keeps its query block outside seq_lens, so an unset
            # prefix means the whole sequence is already-cached prefix.
            prefix_is_full_seq = num_accept_tokens is None
            prefix_lens = (
                seq_lens
                if prefix_is_full_seq
                else seq_lens
                - num_accept_tokens[: seq_lens.shape[0]].to(
                    device=seq_lens.device, dtype=seq_lens.dtype
                )
            )
        else:
            prefix_is_full_seq = False
        sliding_window_size = self.sliding_window_size
        assert sliding_window_size is not None
        for wrapper_id in range(2):
            swa_paged_custom_mask = None
            if wrapper_id == 0:
                if use_ragged:
                    # K for extend tokens is written after the paged wrapper runs, so
                    # the paged wrapper sees prefix-only. Trim to the last `window` tokens
                    # (required for SWATokenToKVPoolAllocator; also keeps mask O(window)).
                    effective_start = torch.clamp(
                        prefix_lens - sliding_window_size, min=0
                    )
                    paged_kernel_lens = prefix_lens - effective_start
                    paged_kernel_lens_sum = paged_kernel_lens.sum().item()
                    kv_start_idx = effective_start
                    swa_paged_custom_mask = self._build_swa_prefix_custom_mask(
                        prefix_lens, seq_lens, effective_start
                    )
                else:
                    # window attention use paged only; the trim below is
                    # request-granular, exactness comes from plan-time window_left
                    paged_kernel_lens = torch.minimum(
                        seq_lens,
                        sliding_window_size + seq_lens - prefix_lens,
                    )
                    if prefix_is_full_seq and seq_lens_cpu is not None:
                        # prefix_lens is seq_lens, so the trim is min(seq_lens, window);
                        # summing the host mirror avoids draining the stream.
                        paged_kernel_lens_sum = int(
                            torch.clamp(seq_lens_cpu, max=sliding_window_size).sum()
                        )
                    else:
                        paged_kernel_lens_sum = paged_kernel_lens.sum().item()
                    kv_start_idx = seq_lens - paged_kernel_lens
            elif use_ragged:
                # Extend K/V lands after the paged pass: plan over the prefix only.
                paged_kernel_lens = prefix_lens
                if extend_prefix_lens_cpu is not None:
                    paged_kernel_lens_sum = sum(extend_prefix_lens_cpu)
                else:
                    paged_kernel_lens_sum = prefix_lens.sum().item()
                kv_start_idx = None
            else:
                # full attention
                paged_kernel_lens = seq_lens
                paged_kernel_lens_sum = seq_lens_sum
                kv_start_idx = seq_lens - paged_kernel_lens
            use_sliding_window_kv_pool = (
                wrapper_id == 0 and self._swa_kv_pool is not None
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
                fixed_split_size=fixed_split_size,
                multi_item_params=multi_item_params,
                cross_attention_custom_mask=swa_paged_custom_mask,
                # paged-only SWA path only; ragged keeps its custom prefix
                # mask, spec-verify keeps its tree mask
                window_left=(
                    sliding_window_size
                    if (wrapper_id == 0 and not use_ragged and spec_info is None)
                    else -1
                ),
                plan=plan,
            )

    def _build_swa_prefix_custom_mask(
        self,
        prefix_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_start_idx: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Custom SWA mask for the paged wrapper in the ragged merge_state EXTEND path.

        Paged KV covers absolute positions [kv_start_idx[i], prefix_lens[i]).
        Returns None when every key is in-window for every extend query.
        """
        window = self.sliding_window_size
        if window is None or window < 0:
            return None

        prefix_lens_cpu = prefix_lens.detach().cpu().tolist()
        extend_lens_cpu = (seq_lens - prefix_lens).detach().cpu().tolist()
        kv_start_cpu = kv_start_idx.detach().cpu().tolist()
        if all(p == 0 for p in prefix_lens_cpu):
            return None

        device = prefix_lens.device
        mask_parts: List[torch.Tensor] = []
        need_mask = False
        for prefix_len, extend_len, kv_start in zip(
            prefix_lens_cpu, extend_lens_cpu, kv_start_cpu
        ):
            paged_len = int(prefix_len - kv_start)  # = min(prefix_len, window)
            if paged_len == 0 or extend_len == 0:
                continue
            q_abs = torch.arange(extend_len, device=device).view(-1, 1) + prefix_len
            k_abs = torch.arange(paged_len, device=device).view(1, -1) + kv_start
            block = (k_abs >= (q_abs - window)).to(torch.uint8)
            if not bool(block.all()):
                need_mask = True
            mask_parts.append(block.view(-1))

        if not need_mask or not mask_parts:
            return None
        return torch.cat(mask_parts)

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
        custom_kv_indices: Optional[torch.Tensor] = None,
        *,
        plan: KVLocPlan,
    ):
        if custom_kv_indices is not None:
            raise RuntimeError(
                "NVFP4 custom KV indices are not supported for cross-attention."
            )
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
                paged_kernel_lens_sum = seq_lens_sum
            else:
                # cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()

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
                cross_attention_custom_mask=(
                    cross_attention_custom_mask if wrapper_id == 1 else None
                ),
                plan=plan,
            )

    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchPrefillWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: Optional[torch.Tensor],
        kv_start_idx: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[SpecInput],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        custom_kv_indices: Optional[torch.Tensor] = None,
        window_left: int = -1,
        *,
        plan: KVLocPlan,
    ):
        bs = len(seq_lens)
        # Unified SWA wrapper-0: gather from the swa canonical directly -- its
        # entries are already swa-side physical ids, so the in-place
        # full->swa translate below must not run on top of them.
        translator = self.attn_backend.kv_index_translator
        use_swa_source = use_sliding_window_kv_pool and translator.reads_are_translated
        read_kind = IdSpaceKind.SLIDING_WINDOW if use_swa_source else IdSpaceKind.FULL
        if spec_info is None:
            assert prefix_lens is not None
            assert len(seq_lens) == len(req_pool_indices)
            # Normal extend
            # custom_kv_indices uses exact dq_paged_kernel_lens so FlashInfer causal
            # offsets are based on real token counts, not page-aligned padding.
            if (
                custom_kv_indices is not None
                and self.attn_backend.dq_paged_kernel_lens is not None
            ):
                kv_indptr[1 : bs + 1] = torch.cumsum(
                    self.attn_backend.dq_paged_kernel_lens, dim=0
                )
            else:
                kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            if custom_kv_indices is not None:
                kv_indices = custom_kv_indices
            else:
                kv_indices = torch.empty(
                    paged_kernel_lens_sum + 256,
                    dtype=torch.int32,
                    device=req_pool_indices.device,
                )
                translator.pack_read_stream(
                    plan,
                    req_pool_indices=req_pool_indices,
                    seq_lens=paged_kernel_lens,
                    indptr=kv_indptr,
                    out=kv_indices,
                    kv_start_idx=kv_start_idx,
                    kind=read_kind,
                )
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]

            custom_mask = cross_attention_custom_mask
        else:
            assert isinstance(spec_info, SpecInput)
            if spec_info.spec_input_type == SpecInputType.DFLASH_VERIFY:
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        req_pool_indices=req_pool_indices,
                        paged_kernel_lens=paged_kernel_lens,
                        paged_kernel_lens_sum=paged_kernel_lens_sum,
                        translator=translator,
                        plan=plan,
                        kind=read_kind,
                        kv_start_idx=kv_start_idx,
                    )
                )
            else:
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        req_pool_indices=req_pool_indices,
                        paged_kernel_lens=paged_kernel_lens,
                        paged_kernel_lens_sum=paged_kernel_lens_sum,
                        translator=translator,
                        plan=plan,
                        kind=read_kind,
                    )
                )

        # extend part
        if use_ragged:
            wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

        if use_sliding_window_kv_pool and not use_swa_source:
            assert self._swa_kv_pool is not None
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self._swa_kv_pool.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        # cached part
        # Conditionally set multi-item parameters
        if multi_item_params is not None and multi_item_params.is_enabled():
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

        # fast_prefill_plan (installed at capture) is sync-free: it needs the
        # host-known qo/kv layout from the caller. Assert rather than silently
        # fall back to plan()'s blocking D2H on the replay hot-path.
        paged_plan_kwargs = {}
        num_tokens_per_req = (
            spec_info.num_tokens_per_req if spec_info is not None else None
        )
        uses_fast_prefill = (
            hasattr(wrapper_paged.begin_forward, "func")
            and wrapper_paged.begin_forward.func is fast_prefill_plan
        )
        if uses_fast_prefill:
            assert seq_lens_cpu is not None, (
                "fast_prefill_plan replay requires host-known seq_lens_cpu (got None)"
            )
            assert num_tokens_per_req is not None and num_tokens_per_req > 0, (
                f"fast_prefill_plan replay requires num_tokens_per_req > 0 (got {num_tokens_per_req})"
            )
            assert use_custom_mask is None, (
                "fast_prefill_plan does not support custom_mask; keep the plain plan()"
            )
            seq_lens_cpu_i32 = seq_lens_cpu.to(torch.int32)
            qo_indptr_host = torch.arange(
                0,
                (bs + 1) * num_tokens_per_req,
                step=num_tokens_per_req,
                dtype=torch.int32,
                device="cpu",
            )
            kv_indptr_host = torch.zeros(bs + 1, dtype=torch.int32, device="cpu")
            kv_indptr_host[1:] = torch.cumsum(seq_lens_cpu_i32, dim=0)
            paged_plan_kwargs = dict(
                qo_indptr_host=qo_indptr_host,
                kv_indptr_host=kv_indptr_host,
                kv_lens_host=seq_lens_cpu_i32,
                max_q_len=num_tokens_per_req,
                max_kv_len=int(seq_lens_cpu_i32.max()),
            )

        if window_left >= 0:
            # selects the module with the per-element window mask compiled in
            paged_plan_kwargs["window_left"] = window_left

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
            **paged_plan_kwargs,
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
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        self.page_size = model_runner.page_size

        max_bs = get_cuda_graph_max_batch_size(
            model_runner.req_to_token_pool.size * self.topk
        )
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
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.draft_window_size, self.draft_sink_size = resolve_draft_decode_window(
            model_runner
        )
        self.kv_index_translator = model_runner.kv_index_translator

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: torch.Tensor,
        call_fn: Callable,
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        required_kv_indices_len = draft_kv_indices_used_len(
            seq_lens_sum, self.topk, bs, self.speculative_num_steps
        )
        assert_buffer_fits(
            required_kv_indices_len,
            kv_indices_buffer.shape[1],
            "EAGLE draft kv_indices row (size max_bs * topk * max_context_len)",
            bs=bs,
            seq_lens_sum=seq_lens_sum,
        )

        src = self.kv_index_translator.read_source(
            forward_batch.kv_loc_plan,
            req_pool_indices=forward_batch.req_pool_indices,
            bs=num_seqs,
        )
        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            src.row_ids,
            src.ids,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            src.row_stride,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
            self.draft_window_size,
            self.draft_sink_size,
            ENTRY_PAGE_SIZE=src.entry_page_size,
            v2p=src.v2p,
            TRANSLATE=src.v2p is not None,
        )

        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        # Copy the kv_indptr once to avoid multiple device-to-host copies in flashinfer's plan.
        indptr_cpu_whole = self.kv_indptr[:, : bs + 1].cpu()
        global global_override_indptr_cpu

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : draft_kv_indices_used_len(seq_lens_sum, self.topk, bs, i + 1)
            ]
            global_override_indptr_cpu = indptr_cpu_whole[i]
            call_fn(i, forward_batch)

        global_override_indptr_cpu = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices_width = draft_kv_indices_buffer_width(
            forward_batch.batch_size, self.topk, self.max_context_len
        )
        kv_indices = torch.empty(
            (self.speculative_num_steps, kv_indices_width),
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
        # generate_draft_decode_kv_indices packs topk per-branch sequences per row,
        # so the row needs the topk factor -- same as the eager init_forward_metadata
        # (batch_size * topk * max_context_len). Dropping it overflows the buffer.
        kv_indices_width = draft_kv_indices_buffer_width(
            max_bs, self.topk, self.max_context_len
        )
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int32,
            device="cuda",
        )

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        bs = forward_batch.batch_size

        def call_fn(i, fb):
            inner_fb = build_inner_fb_view(fb, bs=bs, forward_mode=ForwardMode.DECODE)
            self.attn_backends[i].init_forward_metadata_out_graph(
                inner_fb, in_capture=in_capture
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata_in_graph(forward_batch)


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
