import contextlib
import os
import logging
import time
from typing import List, Optional

import torch

from sglang.kernels.ops.speculative.topk1 import (
    draft_topk1_argmax_only,
    draft_topk1_postprocess,
)
from sglang.srt.configs.model_config import get_dsa_mtp_topk_width
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
    EAGLEDraftExtendNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.kv_canary.runner.canary_manager import context_tuple
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.layers.attention.qsa.config import parse_qsa_profile
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QSAMTPSharedSparseIndices,
    QwenSparseAttnBackend,
    QwenSparseMultiStepDraftBackend,
)
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
)
from sglang.srt.layers.moe.utils import (
    draft_model_build_scope,
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.kv_loc_plan import KVLocPlan
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner import (
    DecodeCudaGraphRunner,
    get_batch_sizes_to_capture,
)
from sglang.srt.runtime_context import (
    get_context,
    get_device,
    get_exec,
    get_model,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.adaptive_spec_params import AdaptiveSpeculativeParams
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.dp_spec_prefill_coordination import (
    DPSpecPrefillCoordinationPlan,
)
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import (
    _eagle_prefill_tail_tokens,
    default_tree_mask_mode,
    get_draft_recurrent_hidden_state_spec,
    organize_draft_results,
    per_step_draft_out_cache_loc,
)
from sglang.srt.speculative.eagle_worker_common import (
    build_eagle_verify_input,
    prepare_for_draft,
    prepare_for_draft_extend,
    run_eagle_verify,
)
from sglang.srt.speculative.pp_draft_embedding import resolve_draft_embed_and_head
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    draft_pp_context,
    draft_tp_context,
    fast_sample,
    get_plan_stream,
    load_token_map,
    renorm_draft_probs,
    sample_draft_proposal,
    select_top_k_tokens,
    spec_stage_span,
)
from sglang.srt.utils.async_probe import (
    maybe_detect_inf,
    maybe_detect_nan,
    maybe_detect_oob,
)
from sglang.srt.utils.common import (
    fast_topk,
    get_available_gpu_memory,
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
    log_info_on_rank0,
)

_is_cpu = is_cpu()
_is_npu = is_npu()
_is_cuda = is_cuda()
_is_musa = is_musa()
_is_hip = is_hip()
_is_xpu = is_xpu()


logger = logging.getLogger(__name__)


def _qsa_index_share_requested(hf_config) -> bool:
    """--json-model-override-args writes top-level hf_config attributes, while
    checkpoint configs carry the flag on the nested text_config; read both."""
    text_config = getattr(hf_config, "text_config", hf_config)
    return bool(
        getattr(
            text_config,
            "index_share_for_mtp_iteration",
            getattr(hf_config, "index_share_for_mtp_iteration", False),
        )
    )


class EagleDraftWorker(EagleDraftWorkerBase):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()

        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        self.target_worker = target_worker

        # Args for easy access
        self.device = get_device().device
        self.topk = get_spec().speculative_eagle_topk
        if get_spec().speculative_use_rejection_sampling:
            assert self.topk == 1, "Chain speculative sampling supports only topk=1"
        self.speculative_num_steps = get_spec().speculative_num_steps
        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )

        self._rebuild_topk1_chain_buffers()

        # Use the same attention topology during draft construction and execution.
        self.draft_owns_attention = (
            get_parallel().attn_dp_enabled and self.speculative_algorithm.is_eagle3()
        )
        with (
            draft_tp_context(self.draft_owns_attention),
            draft_pp_context(),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            draft_model_build_scope(),
        ):
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                nccl_port=nccl_port,
                is_draft_worker=True,
                # The draft runs at absolute target positions.
                context_length=target_worker.model_runner.model_config.context_len,
                random_seed=target_worker.random_seed,
            )

        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner
        self._init_dsa_index_share_state()
        # Eager draft-extend seed buffer (graph paths use their own static ones).
        self.dsa_extend_topk_buf: Optional[torch.Tensor] = None
        self.tree_mask_mode = default_tree_mask_mode()

        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        """Allocate draft KV cache pools (called by scheduler)."""
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        self.init_token_map()
        self.init_lm_head()

        if get_spec().speculative_use_rejection_sampling:
            target_vocab_size = self.target_worker.model_config.vocab_size
            draft_vocab_size = (
                self.hot_token_id.shape[0]
                if self.hot_token_id is not None
                else target_vocab_size
            )
            # FIXME: support reduced (hot) draft vocab by scattering draft probs
            # into the target vocab via the d2t map before the sampling kernel.
            if draft_vocab_size != target_vocab_size:
                raise ValueError(
                    "--speculative-use-rejection-sampling requires the draft and "
                    f"target to share one vocab, but the draft vocab "
                    f"({draft_vocab_size}) != target vocab ({target_vocab_size})."
                )

    def init_attention_backends(self):
        with (
            draft_pp_context(),
            draft_tp_context(self.draft_owns_attention),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.draft_worker.init_attention_backends()
            self.init_attention_backend()

    def init_cuda_graphs(self):
        with (
            draft_pp_context(),
            draft_tp_context(self.draft_owns_attention),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.draft_worker.init_cuda_graphs(capture_decode_cuda_graph=False)
            if check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE):
                self.draft_runner.init_prefill_cuda_graph(force_for_draft_worker=True)
            self._capture_cuda_graphs()

        if (c := self.draft_runner.canary_manager) is not None:
            c.mark_init_finished()

    def _init_dsa_index_share_state(self) -> None:
        # Populate DSA index-share fields from the draft runner's hf_config.
        # Reused by the attention unit-test harnesses, which skip __init__.
        hf_config = self.draft_runner.model_config.hf_config
        # Reuse the first draft step's DSA indexer topk across the rest;
        # topk == 1 only (select_top_k_tokens reorders rows, desyncing indices).
        self.index_share_for_mtp_iteration = (
            getattr(hf_config, "index_share_for_mtp_iteration", False)
            and self.topk == 1
        )
        # GLM-5.2 MTP IndexShare: seed reused indexer top-k from draft-extend
        # (last verified token), not draft-decode step 0.
        self.dsa_index_topk = getattr(hf_config, "index_topk", None)
        self.dsa_seed_topk_width = (
            get_dsa_mtp_topk_width(hf_config)
            if self.index_share_for_mtp_iteration and self.dsa_index_topk is not None
            else None
        )
        self.seed_dsa_topk_from_draft_extend = (
            self.index_share_for_mtp_iteration and self.dsa_seed_topk_width is not None
        )

    def init_token_map(self):
        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if get_spec().speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif get_spec().speculative_token_map is not None:
            self.hot_token_id = load_token_map(get_spec().speculative_token_map)
        else:
            self.hot_token_id = None

    def init_lm_head(self):
        from sglang.srt.lora.layers import unwrap_lora_layer

        embed, head = self._resolve_shared_embed_and_head()
        target_lm_head = unwrap_lora_layer(
            getattr(self.target_worker.model_runner.model, "lm_head", None)
        )

        def maybe_share_target_lm_head():
            if (
                target_lm_head is not None
                and self.hot_token_id is None
                and getattr(self.draft_runner.model, "hot_token_id", None) is None
                and hasattr(self.draft_runner.model, "set_lm_head_from_target")
            ):
                self.draft_runner.model.set_lm_head_from_target(target_lm_head)

        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                self.draft_runner.model.set_embed_and_head(embed, head)
                maybe_share_target_lm_head()
            else:
                self.draft_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.hot_token_id is not None and head is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_runner.model.set_embed_and_head(embed, head)
            maybe_share_target_lm_head()

    def _resolve_shared_embed_and_head(self):
        target_runner = self.target_worker.model_runner
        return resolve_draft_embed_and_head(
            target_model=target_runner.model,
            draft_model=self.draft_runner.model,
            model_path=target_runner.model_config.model_path,
            revision=target_runner.model_config.revision,
            load_config=target_runner.load_config,
        )

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners

        self.draft_extend_attn_backend = None

        draft_backend_factory = DraftBackendFactory(
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
            seed_dsa_topk_from_draft_extend=self.seed_dsa_topk_from_draft_extend,
            qsa_profile=parse_qsa_profile(self.draft_runner.model_config.hf_config),
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        if self.draft_extend_attn_backend is not None:
            self.draft_runner.attn_backend = self.draft_extend_attn_backend
        self._configure_qsa_mtp_index_share()
        self.tree_mask_mode = default_tree_mask_mode()

        # Boot guard: every backend a draft forward reaches must carry its
        # runner's translator or the index builders emit virtual ids.
        translator = self.draft_runner.kv_index_translator
        if translator.is_translating:
            backends = [self.draft_attn_backend, self.draft_extend_attn_backend]
            if self.draft_attn_backend is not None:
                backends += self.draft_attn_backend.attn_backends
            translator.bind_and_verify_backends(backends)

    def _configure_qsa_mtp_index_share(self) -> None:
        """Reuse the draft-extend QSA selection across the MTP decode steps;
        chain speculation only: with topk > 1 decode rows are not request-major."""
        from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

        hf_config = self.draft_runner.model_config.hf_config
        if (
            not _qsa_index_share_requested(hf_config)
            or self.topk != 1
            or self.speculative_num_steps <= 1
            or not isinstance(self.draft_attn_backend, QwenSparseMultiStepDraftBackend)
            or not isinstance(self.draft_extend_attn_backend, QwenSparseAttnBackend)
        ):
            return
        if get_spec().speculative_adaptive:
            # Adaptive speculation switches SpecRuntimeState between the draft-extend
            # capture and the decode lookup; per-state index buffers would not match.
            logger.warning(
                "index_share_for_mtp_iteration is disabled under adaptive "
                "speculative decoding"
            )
            return
        layer_ids = sorted(
            {
                module.layer_id
                for module in self.draft_runner.model.modules()
                if isinstance(module, QSAIndexer)
            }
        )
        if not layer_ids:
            return
        pool = self.draft_runner.token_to_kv_pool
        # The expansion emits token_topk + ratio - 1 columns (top-k blocks
        # plus the uncompressed tail of the capture position).
        expanded_width = pool.qsa_token_topk + pool.qsa_compress_ratio - 1
        state = QSAMTPSharedSparseIndices(
            layer_ids=layer_ids,
            num_requests=self.draft_runner.req_to_token_pool.req_to_token.shape[0],
            token_topk=expanded_width,
            tail_width=get_spec().speculative_num_steps + 1,
            device=self.draft_runner.device,
        )
        for backend in (self.draft_attn_backend, self.draft_extend_attn_backend):
            backend.set_mtp_shared_sparse_indices(state)
        logger.info(
            "QSA MTP index sharing enabled: draft decode steps reuse the "
            f"draft-extend selection for layers {layer_ids}"
        )

    def _capture_cuda_graphs(self):
        """Capture the draft worker's own cuda graphs (decode + draft-extend)."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if _is_cpu or check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
            return

        if get_model().model_impl == "mindspore":
            return

        Device2DraftCudaGraphRunner = {
            "xpu": EAGLEDraftCudaGraphRunner,
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        decode_backend = get_exec().graph.cuda_graph_config.decode.backend
        capture_bs, _ = get_batch_sizes_to_capture(self.draft_runner)
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft decode CUDA graph begin. backend={decode_backend}, "
                f"num_tokens_per_req={self.topk}, bs={capture_bs}, "
                f"avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            capture_time = time.perf_counter() - tic
            self._specialized_graph_memory_usage["draft_decode"] = (
                self._specialized_graph_memory_usage.get("draft_decode", 0.0)
                + before_mem
                - after_mem
            )
            self._specialized_graph_time_usage["draft_decode"] = (
                self._specialized_graph_time_usage.get("draft_decode", 0.0)
                + capture_time
            )
            log_info_on_rank0(
                logger,
                "Capture draft decode CUDA graph end. "
                f"elapsed={capture_time:.2f} s, "
                f"mem usage={(before_mem - after_mem):.2f} GB, "
                f"avail mem={after_mem:.2f} GB.",
            )

        Device2ExtendCudaGraphRunner = {
            "xpu": EAGLEDraftExtendCudaGraphRunner,
            "npu": EAGLEDraftExtendNpuGraphRunner,
            "cuda": EAGLEDraftExtendCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        supports_hip_draft_extend_graph = False
        if _is_hip:
            # Keep imports local so non-HIP environments do not require these.
            # aiter packs draft-extend support into the decode (multi-step)
            # backend; DSV4 exposes it on the draft-extend backend itself.
            from sglang.srt.layers.attention.aiter_backend import (
                AiterMultiStepDraftBackend,
            )
            from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
                DeepseekV4HipRadixBackend,
            )
            from sglang.srt.layers.attention.dsa_backend import (
                DeepseekSparseAttnBackend,
            )

            supports_hip_draft_extend_graph = (
                isinstance(self.draft_attn_backend, AiterMultiStepDraftBackend)
                or isinstance(self.draft_extend_attn_backend, DeepseekV4HipRadixBackend)
                or isinstance(self.draft_extend_attn_backend, DeepseekSparseAttnBackend)
            )

        graph_supported_backend_types = [
            TritonAttnBackend,
            TRTLLMMLABackend,
            TRTLLMHAAttnBackend,
            TokenspeedMLABackend,
            FlashInferAttnBackend,
            QwenSparseAttnBackend,
        ]
        if _is_cuda or _is_musa:
            # DSA is CUDA-only; import lazily so non-CUDA builds don't pull in
            # deep_gemm and the rest of the sparse-attention stack at import time.
            from sglang.srt.layers.attention.dsa_backend import (
                DeepseekSparseAttnBackend,
            )

            graph_supported_backend_types.append(DeepseekSparseAttnBackend)
            from sglang.srt.layers.attention.deepseek_v4_backend import (
                DeepseekV4AttnBackend,
            )

            graph_supported_backend_types.append(DeepseekV4AttnBackend)
        if _is_cuda:
            # FlashMLA is CUDA-only; import lazily so CPU builds don't pull
            # sgl_kernel.flash_mla at import time.
            from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend

            graph_supported_backend_types.append(FlashMLABackend)

        graph_supported_backend = isinstance(
            self.draft_extend_attn_backend,
            tuple(graph_supported_backend_types),
        )
        supports_cuda_draft_extend_graph = (
            _is_cuda or _is_musa
        ) and graph_supported_backend
        # Capture extend
        # TODO: support draft extend cuda graph for more attention backends
        if (
            self.draft_extend_attn_backend
            and not envs.SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH.get()
            and (
                _is_npu
                or _is_xpu
                or supports_cuda_draft_extend_graph
                or supports_hip_draft_extend_graph
            )
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft extend CUDA graph begin. backend={decode_backend}, "
                f"num_tokens_per_req={self.speculative_num_draft_tokens}, "
                f"bs={capture_bs}, avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)
            # draft_extend is the step's last shared-buffer-reading phase; its
            # read-done event is what the scheduler's WAR barrier waits on.
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            capture_time = time.perf_counter() - tic
            self._specialized_graph_memory_usage["draft_extend"] = (
                self._specialized_graph_memory_usage.get("draft_extend", 0.0)
                + before_mem
                - after_mem
            )
            self._specialized_graph_time_usage["draft_extend"] = (
                self._specialized_graph_time_usage.get("draft_extend", 0.0)
                + capture_time
            )
            log_info_on_rank0(
                logger,
                "Capture draft extend CUDA graph end. "
                f"elapsed={capture_time:.2f} s, "
                f"mem usage={(before_mem - after_mem):.2f} GB, "
                f"avail mem={after_mem:.2f} GB.",
            )

    def draft(self, batch: ScheduleBatch, *, with_topology: bool = False):
        draft_input: EagleDraftInput = batch.spec_info
        forward_batch, can_run_decode_cuda_graph, kv_loc_plan = prepare_for_draft(
            draft_input,
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
            target_translator=self.target_worker.model_runner.kv_index_translator,
            num_draft_tokens=self.speculative_num_draft_tokens,
        )
        if (
            can_run_decode_cuda_graph
            and not forward_batch.forward_mode.is_idle()
            and self.seed_dsa_topk_from_draft_extend
            and draft_input.dsa_topk_indices is None
        ):
            can_run_decode_cuda_graph = False

        n_inner = self.speculative_num_steps - 1
        canary_outside_ctx = (
            c.with_ops_outside_graph(
                single_forward_indices=list(range(n_inner)),
                maybe_inaccurate_forward_batch=forward_batch,
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )

        with canary_outside_ctx:
            # Run draft
            if can_run_decode_cuda_graph:
                parent_list, top_scores_index, draft_tokens, draft_probs = (
                    self.cuda_graph_runner.execute(forward_batch)
                )
                if draft_probs is not None:
                    # draft_probs is the one graph output read after the target
                    # forward rather than by it, and it points into the graph's
                    # private memory pool. The pool recycles that block in the
                    # meantime -- in practice the DSA top-k mask lands there and
                    # eagle_sample sees -inf. Copy out at the boundary.
                    draft_probs = draft_probs.clone()
            else:
                if (
                    not forward_batch.forward_mode.is_idle()
                    and self.speculative_num_steps > 1
                ):
                    # Skip attention backend init for 1-step draft,
                    # `draft_forward` only does sample in this case.
                    self.draft_attn_backend.init_forward_metadata(forward_batch)
                    forward_batch.mark_forward_metadata_ready()
                parent_list, top_scores_index, draft_tokens, draft_probs = (
                    self.draft_forward(forward_batch)
                )

        verify_input = build_eagle_verify_input(
            batch,
            draft_input,
            parent_list,
            top_scores_index,
            draft_tokens,
            draft_probs,
            target_worker=self.target_worker,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            tree_mask_mode=self.tree_mask_mode,
            device=self.device,
        )
        if kv_loc_plan is not None:
            # Verify and draft extend write the window the draft planned.
            verify_input.kv_loc_plan = kv_loc_plan
            verify_input.prepared_out_cache_loc = kv_loc_plan.write_virtual
        if with_topology:
            # PP+spec relays the tree so every stage rebuilds the same verify
            # input; the mask build needs the topology this one was built from.
            # Returned rather than stashed on self so the caller owns lifetime.
            return verify_input, parent_list, top_scores_index
        return verify_input

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        if forward_batch.forward_mode.is_idle():
            return self._draft_forward_idle(forward_batch, spec_info)

        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        out_cache_loc = per_step_draft_out_cache_loc(
            out_cache_loc,
            forward_batch.batch_size,
            self.topk,
            self.speculative_num_steps,
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        # Verify commits argmax for an all-greedy batch and never reads the
        # draft distribution, so skip building it.
        needs_draft_probs = (
            get_spec().speculative_use_rejection_sampling
            and not forward_batch.sampling_info.is_all_greedy
        )
        if needs_draft_probs:
            draft_probs_list: List[torch.Tensor] = [spec_info.draft_probs]

        topk1_chain_fits = (
            self.topk == 1
            and topk_index.shape[0] <= self._topk1_parents_prealloc.shape[0]
        )
        # Materialize the chain directly only when the CUDA kernel can write
        # every subsequent column. Other topk=1 paths retain the token list and
        # assemble it with one final cat instead of launching a copy per step.
        draft_tokens_topk1 = None
        if (
            topk1_chain_fits
            and _is_cuda
            and self.hot_token_id is None
            and not needs_draft_probs
        ):
            draft_tokens_topk1 = torch.empty(
                (topk_index.shape[0], self.speculative_num_steps),
                dtype=topk_index.dtype,
                device=topk_index.device,
            )
            draft_tokens_topk1[:, :1].copy_(topk_index)

        # Forward multiple steps
        scores = None
        with IndexTopKShareState.mtp_iteration(
            forward_batch,
            enabled=self.index_share_for_mtp_iteration,
            keep_carry_seed=self.seed_dsa_topk_from_draft_extend,
        ):
            for i in range(self.speculative_num_steps):
                if draft_tokens_topk1 is not None:
                    input_ids = topk_index.flatten()
                else:
                    input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                        i, topk_p, topk_index, hidden_states, scores, self.topk
                    )
                    score_list.append(tree_info[0])
                    token_list.append(tree_info[1])
                    parents_list.append(tree_info[2])

                if i == self.speculative_num_steps - 1:
                    break

                forward_batch.input_ids = input_ids
                # Qwen3-MoE MTP uses a fused RoPE + KV-store path whose cache_loc
                # argument must be contiguous.
                if (
                    self.draft_runner.model_config.hf_config.architectures[0]
                    == "Qwen3MoeForCausalLMMTP"
                ):
                    out_cache_loc = out_cache_loc.contiguous()
                forward_batch.out_cache_loc = out_cache_loc[i]
                spec_info.hidden_states = hidden_states

                canary_index_ctx = (
                    c.with_active_single_forward_manager(i)
                    if (c := self.draft_runner.canary_manager) is not None
                    else contextlib.nullcontext()
                )
                with (
                    forward_context(
                        ForwardContext(
                            attn_backend=self.draft_attn_backend.attn_backends[i]
                        )
                    ),
                    canary_index_ctx,
                ):
                    logits_output = self.draft_runner.forward(
                        forward_batch
                    ).logits_output
                maybe_detect_nan(
                    logits_output.next_token_logits, f"draft_forward step {i}"
                )
                maybe_detect_inf(
                    logits_output.next_token_logits, f"draft_forward step {i}"
                )
                if needs_draft_probs:
                    probs, topk_p, topk_index = sample_draft_proposal(
                        logits_output.next_token_logits,
                        forward_batch.sampling_info.temperatures,
                        forward_batch.sampling_info.top_ks,
                    )
                    draft_probs_list.append(probs)
                    forward_batch.positions.add_(1)
                elif self.topk == 1:
                    if _is_cuda or _is_hip:
                        topk_p, topk_index = draft_topk1_postprocess(
                            logits_output.next_token_logits,
                            forward_batch.positions,
                            draft_tokens_topk1,
                            i + 1,
                        )
                    else:
                        topk_index = torch.argmax(
                            logits_output.next_token_logits, dim=-1, keepdim=True
                        )
                        topk_p = torch.ones_like(topk_index, dtype=torch.float32)
                        forward_batch.positions.add_(1)
                else:
                    probs = renorm_draft_probs(
                        logits_output.next_token_logits,
                        forward_batch.sampling_info,
                        needs_draft_probs,
                    )
                    topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
                    forward_batch.positions.add_(1)
                if self.draft_runner.model_config.model_is_mrope:
                    forward_batch.mrope_positions.add_(1)
                maybe_detect_oob(
                    topk_index,
                    0,
                    logits_output.next_token_logits.shape[-1],
                    f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
                )
                if self.hot_token_id is not None:
                    topk_index = self.hot_token_id[topk_index]
                hidden_states = logits_output.hidden_states

        draft_probs = (
            torch.stack(draft_probs_list, dim=1) if needs_draft_probs else None
        )

        # Organize the results
        if draft_tokens_topk1 is not None:
            bs = draft_tokens_topk1.shape[0]
            top_scores_index = self._topk1_score_indices_prealloc[:bs]
            parent_list = self._topk1_parents_prealloc[:bs]
            return parent_list, top_scores_index, draft_tokens_topk1, draft_probs

        if topk1_chain_fits:
            bs = token_list[0].shape[0]
            draft_tokens = torch.cat(token_list, dim=1)
            top_scores_index = self._topk1_score_indices_prealloc[:bs]
            parent_list = self._topk1_parents_prealloc[:bs]
            return parent_list, top_scores_index, draft_tokens, draft_probs

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        return parent_list, top_scores_index, draft_tokens, draft_probs

    def _draft_forward_idle(
        self, forward_batch: ForwardBatch, spec_info: EagleDraftInput
    ):
        """Run eager idle-rank collectives without materializing draft state."""
        input_ids = forward_batch.input_ids
        out_cache_loc = forward_batch.out_cache_loc
        hidden_states = spec_info.hidden_states

        # ModelRunner pads and unpads the empty batch on every call. Avoid the
        # normal tree/cache-layout path: idle outputs are discarded when the
        # verify input is built, but every rank must still enter each forward.
        for i in range(self.speculative_num_steps - 1):
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc
            spec_info.hidden_states = hidden_states
            canary_index_ctx = (
                c.with_active_single_forward_manager(i)
                if (c := self.draft_runner.canary_manager) is not None
                else contextlib.nullcontext()
            )
            with (
                forward_context(
                    ForwardContext(
                        attn_backend=self.draft_attn_backend.attn_backends[i]
                    )
                ),
                canary_index_ctx,
            ):
                self.draft_runner.forward(forward_batch)

        return None, None, None, None

    def draft_extend(self):
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ScheduleBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        mm_input_embeds: Optional[torch.Tensor] = None,
        kv_loc_plan: Optional[KVLocPlan] = None,
    ):
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
            kv_loc_plan: The target prefill's plan; the draft writes the same
                slots.
        """
        # Construct input_ids
        if not batch.forward_mode.is_idle():
            # Chunked-prefill-aware tail tokens (see PR #26329).
            tail_tokens = _eagle_prefill_tail_tokens(batch, next_token_ids)

            new_input_ids = torch.empty_like(batch.input_ids)
            if mm_input_embeds is not None:
                # Rotate mm embeddings the same way as input_ids: shift left by
                # one per request so they stay aligned with the rotated ids. The
                # last position per request is filled by the draft model's own
                # embed_tokens lookup on next_token_ids (see DeepseekModelNextN).
                rotated_mm = torch.empty_like(mm_input_embeds)
            pt = 0
            for i, extend_len in enumerate(batch.extend_lens):
                input_ids = batch.input_ids[pt : pt + extend_len]
                new_input_ids[pt : pt + extend_len].copy_(
                    torch.cat((input_ids[1:], tail_tokens[i].reshape(1)))
                )
                if mm_input_embeds is not None:
                    rotated_mm[pt : pt + extend_len - 1].copy_(
                        mm_input_embeds[pt + 1 : pt + extend_len]
                    )
                pt += extend_len
            assert pt == batch.input_ids.numel()
            batch.input_ids = new_input_ids
            if mm_input_embeds is not None:
                mm_input_embeds = rotated_mm

        # Draft-extend spec_info for the extend forward; carries only
        # hidden_states + shape info.
        batch.spec_info = EagleDraftExtendInput(
            hidden_states=target_hidden_states,
            # draft mode is same with decode mode, only 1 token per req
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

        # Run forward (LAST mode: only the final hidden state per request,
        # to feed the next draft step which expects [bs, hidden_dim]).
        # STANDALONE skips hidden states end-to-end.
        capture_hidden_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        forward_batch = ForwardBatch.init_new(
            batch,
            self.draft_runner,
            capture_hidden_mode=capture_hidden_mode,
            return_hidden_states_before_norm=False,
            kv_loc_plan=kv_loc_plan,
        )
        forward_batch.return_logprob = False
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds

        # Seed the first draft-decode loop from each request's last prefill
        # position. Gather last-per-req before the copy (prefill can be long).
        seed_from_extend = (
            self.seed_dsa_topk_from_draft_extend
            and not forward_batch.forward_mode.is_idle()
        )
        if seed_from_extend:
            bs = forward_batch.batch_size
            forward_batch.spec_info.dsa_seed_topk_capture = (
                self._get_dsa_extend_topk_buf(bs)
            )
            forward_batch.spec_info.dsa_seed_topk_select = (
                torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            ).long()

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            logits_output = self.draft_runner.forward(forward_batch).logits_output
        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")
        maybe_detect_inf(logits_output.next_token_logits, "draft_extend_for_prefill")

        prefill_dsa_topk = None
        if seed_from_extend:
            prefill_dsa_topk = self.dsa_extend_topk_buf[:bs].clone()

        # Assemble the next-iter draft spec_info from the extend output.
        use_rejection_sampling = get_spec().speculative_use_rejection_sampling
        probs = renorm_draft_probs(
            logits_output.next_token_logits,
            batch.sampling_info,
            use_rejection_sampling,
        )
        if use_rejection_sampling:
            topk_p, topk_index = fast_sample(probs, num_samples=1)
        else:
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        return EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            draft_probs=probs if use_rejection_sampling else None,
            hidden_states=logits_output.hidden_states,
            bonus_tokens=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
            dsa_topk_indices=prefill_dsa_topk,
            cuda_graph_compatible=not (seed_from_extend and prefill_dsa_topk is None),
        )

    def _get_dsa_extend_topk_buf(self, num_tokens: int) -> torch.Tensor:
        buf = self.dsa_extend_topk_buf
        if buf is None or buf.shape[0] < num_tokens:
            buf = torch.full(
                (num_tokens, self.dsa_seed_topk_width),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            self.dsa_extend_topk_buf = buf
        return buf[:num_tokens]

    def _draft_extend_for_decode(
        self,
        batch: ScheduleBatch,
        batch_result: GenerationBatchResult,
        *,
        kv_loc_plan: Optional[KVLocPlan] = None,
    ):
        # Cast to int64 before entering plan stream to avoid cross-stream
        # synchronization issues with .to() inside the plan stream context.
        if batch_result.prepared_draft_extend_inputs is not None:
            num_correct_drafts, select_index, next_token_ids = (
                batch_result.prepared_draft_extend_inputs
            )
        elif batch_result.accept_lens.is_cuda:
            from sglang.kernels.ops.speculative.eagle import prepare_draft_extend_inputs

            num_correct_drafts, select_index, next_token_ids = (
                prepare_draft_extend_inputs(
                    batch_result.accept_lens,
                    batch_result.next_token_ids,
                    self.speculative_num_draft_tokens,
                )
            )
        else:
            num_correct_drafts = batch_result.accept_lens - 1
            select_index = (
                torch.arange(
                    0,
                    len(batch.seq_lens) * self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens,
                    device=self.device,
                )
                + batch_result.accept_lens
                - 1
            )

            next_token_ids = batch_result.next_token_ids.to(torch.int64)

        # Batch 2: Draft extend
        draft_extend_input = EagleDraftExtendInput(
            hidden_states=batch_result.logits_output.hidden_states,
            # accept_lens includes the bonus token; correct drafts exclude it.
            num_correct_drafts=num_correct_drafts,
            num_accept_tokens=batch_result.accept_lens,
            # Draft-extend fills the whole tree width (num_draft_tokens) per req,
            # not num_steps + 1, so DP MLP-sync padding stays consistent for topk > 1.
            num_tokens_per_req=self.speculative_num_draft_tokens,
            num_tokens_for_logprob_per_req=self.speculative_num_draft_tokens,
        )
        # Prepare for draft extend in a separate stream
        if self.plan_stream:
            self.plan_stream.wait_stream(
                torch.get_device_module(self.device).current_stream()
            )
        with self.plan_stream_ctx:
            forward_batch = prepare_for_draft_extend(
                draft_extend_input,
                batch,
                next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner,
                self.cuda_graph_runner_for_draft_extend,
                return_hidden_states_before_norm=False,
                kv_loc_plan=kv_loc_plan,
            )

        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

        # Run draft extend batch in the main compute stream
        can_run_decode_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run_graph(forward_batch)
        )

        # Eager path publishes the indexer top-k into a worker buffer (the graph
        # path uses the runner's static buffer). Gathered at select_index below.
        if self.seed_dsa_topk_from_draft_extend and not can_run_decode_cuda_graph:
            forward_batch.spec_info.dsa_seed_topk_capture = (
                self._get_dsa_extend_topk_buf(forward_batch.input_ids.shape[0])
            )

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            if can_run_decode_cuda_graph:
                draft_logits_output = self.cuda_graph_runner_for_draft_extend.execute(
                    forward_batch, select_index
                )
            else:
                draft_logits_output = self.draft_runner.forward(
                    forward_batch
                ).logits_output

        maybe_detect_nan(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_run_decode_cuda_graph})",
        )
        maybe_detect_inf(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_run_decode_cuda_graph})",
        )

        # Gather the per-request last-position indexer top-k as the next loop's
        # seed (select_index already picks the last accepted position per req).
        dsa_seed_topk_indices = None
        if self.seed_dsa_topk_from_draft_extend:
            if can_run_decode_cuda_graph:
                dsa_extend_topk_capture = self.cuda_graph_runner_for_draft_extend.buffers.dsa_seed_topk_capture
            else:
                dsa_extend_topk_capture = forward_batch.spec_info.dsa_seed_topk_capture
            # Fancy indexing returns a fresh tensor (detached from the buffer).
            dsa_seed_topk_indices = dsa_extend_topk_capture[select_index]

        # Reorganize the spec info for the next batch
        if not can_run_decode_cuda_graph:
            draft_logits_output.next_token_logits = (
                draft_logits_output.next_token_logits[select_index]
            )
            if draft_logits_output.hidden_states is not None:
                draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                    select_index
                ]
        # Selected-row top-k remains worker-owned for both graph and eager
        # paths; the graph runner only moves the row selection before lm_head.
        if get_spec().speculative_use_rejection_sampling:
            ret_draft_probs, ret_topk_p, ret_topk_index = sample_draft_proposal(
                draft_logits_output.next_token_logits,
                batch.sampling_info.temperatures,
                batch.sampling_info.top_ks,
            )
        elif self.topk == 1 and _is_hip:
            ret_topk_p, ret_topk_index = draft_topk1_argmax_only(
                draft_logits_output.next_token_logits
            )
            ret_draft_probs = None
        elif self.topk == 1 and not _is_hip:
            # Gated to CUDA: see #26358 — ROCm's argmax tie-break corrupts
            # MTP draft selection on FP8 logits.
            ret_topk_index = torch.argmax(
                draft_logits_output.next_token_logits, dim=-1, keepdim=True
            )
            ret_topk_p = torch.ones_like(ret_topk_index, dtype=torch.float32)
            ret_draft_probs = None
        else:
            probs = renorm_draft_probs(
                draft_logits_output.next_token_logits,
                batch.sampling_info,
                get_spec().speculative_use_rejection_sampling,
            )
            ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
            ret_draft_probs = None
        ret_hidden_states = draft_logits_output.hidden_states

        # Construct the return values
        next_draft_input = batch_result.next_draft_input
        (
            next_draft_input.topk_p,
            next_draft_input.topk_index,
            next_draft_input.hidden_states,
        ) = (
            ret_topk_p,
            ret_topk_index,
            ret_hidden_states,
        )
        if get_spec().speculative_use_rejection_sampling:
            next_draft_input.draft_probs = ret_draft_probs
        if self.seed_dsa_topk_from_draft_extend:
            next_draft_input.dsa_topk_indices = dsa_seed_topk_indices
        next_draft_input.cuda_graph_compatible = not (
            self.seed_dsa_topk_from_draft_extend and dsa_seed_topk_indices is None
        )


class EAGLEWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()

        # Parse arguments
        self.server_args = server_args
        self.topk = get_spec().speculative_eagle_topk
        self.speculative_num_steps = get_spec().speculative_num_steps
        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = get_device().device
        self._target_worker = target_worker
        self.page_size = get_schedule().page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )

        self.enable_dp_spec_prefill_coordination = (
            envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.get()
        )

        # Only the last PP stage runs the draft; other EAGLEWorkerV2 instances
        # return proxies so scheduler dispatch remains rank-uniform.
        self._hosts_draft = get_parallel().pp_group.is_last_rank
        self._draft_worker = (
            EagleDraftWorker(
                server_args,
                gpu_id,
                nccl_port,
                target_worker,
            )
            if self._hosts_draft
            else None
        )

        # Adaptive speculative
        self.adaptive_controller: Optional[AdaptiveController] = None
        if get_spec().speculative_adaptive and self._hosts_draft:
            self.adaptive_controller = AdaptiveController(
                self,
                AdaptiveSpeculativeParams(
                    initial_steps=self.speculative_num_steps,
                    cfg_path=get_spec().speculative_adaptive_config,
                ),
            )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)

    @property
    def last_shared_read_runner(self):
        # Per the base contract: the step's last shared-buffer-reading phase is
        # draft_extend when this rank owns the draft. Non-last PP ranks execute
        # only the target prefill.
        if self._draft_worker is None:
            return self._target_worker.model_runner
        return self._draft_worker.draft_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        # Every attn backend a spec_v2 forward touches; consumed by
        # decide_needs_cpu_seq_lens to gate the seq_lens_cpu D2H.
        if self._draft_worker is None:
            return (self._target_worker.model_runner.attn_backend,)
        return (
            self._target_worker.model_runner.attn_backend,
            self._draft_worker.draft_attn_backend,
            self._draft_worker.draft_extend_attn_backend
            or self._draft_worker.draft_runner.attn_backend,
        )

    def init_cuda_graphs(self):
        super().init_cuda_graphs()
        # Build adaptive runtime states after target and draft backends exist.
        if self.adaptive_controller is not None:
            with (
                draft_tp_context(self._draft_worker.draft_owns_attention),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self.adaptive_controller.register(
                    SpecRuntimeState(
                        speculative_num_steps=self.speculative_num_steps,
                        speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                        draft_attn_backend=self._draft_worker.draft_attn_backend,
                        cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                        target_attn_backend=self._target_worker.model_runner.attn_backend,
                        target_graph_runner=self._target_worker.model_runner.decode_cuda_graph_runner,
                        draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                        cuda_graph_runner_for_draft_extend=self._draft_worker.cuda_graph_runner_for_draft_extend,
                    )
                )
                self.adaptive_controller.init_states(
                    cuda_graph_bs=(
                        None
                        if check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED)
                        else get_exec().graph.cuda_graph_bs_decode
                    ),
                )

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        # 4-mode step: the scheduler attached a running batch for this step (mixed_spec_running)
        # instead of merging it, because the merge needs draft tokens that only draft() produces.
        # WITHOUT this branch the attached batch is never consumed and its KV slots are orphaned --
        # measured as `pool memory leak detected`, identical byte-for-byte across two runs because
        # the dispatch was simply missing.
        if getattr(batch, "mixed_spec_running", None) is not None:
            return self.forward_mixed_spec(
                batch, on_publish=on_publish, pp_proxy_tensors=pp_proxy_tensors
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            if not (
                batch.is_extend_in_batch and self.enable_dp_spec_prefill_coordination
            ):
                return self._forward_prefill_batch(batch, on_publish, pp_proxy_tensors)
            else:
                if batch.dp_spec_prefill_coordination_metadata is None:
                    raise RuntimeError("Missing DP spec/prefill coordination metadata")
                plan = DPSpecPrefillCoordinationPlan(
                    *batch.dp_spec_prefill_coordination_metadata,
                    draft_width=self.topk,
                    verify_width=self.speculative_num_draft_tokens,
                )
                if not plan.heterogeneous:
                    return self._forward_prefill_batch(
                        batch, on_publish, pp_proxy_tensors
                    )
                else:
                    return self._forward_dp_spec_prefill_coordination(
                        batch, plan, on_publish, grammar_barrier, pp_proxy_tensors
                    )
        else:
            self.activate_step_by_batch(batch.seq_lens.shape[0])

            if batch.spec_info is None:
                capture_mode = (
                    CaptureHiddenMode.NULL
                    if self.speculative_algorithm.is_standalone()
                    else CaptureHiddenMode.LAST
                )
                hidden_size, hidden_dtype = get_draft_recurrent_hidden_state_spec(
                    self.draft_worker.draft_runner
                )
                batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=hidden_size,
                    dtype=hidden_dtype,
                    topk=self.topk,
                    capture_hidden_mode=capture_mode,
                    vocab_size=self.target_worker.model_config.vocab_size,
                )
            if batch.spec_info is not None and batch.spec_info.is_verify_input():
                # PP+spec: the scheduler pre-built this round's verify input
                # from relayed per-req chains — it must match what earlier
                # stages already ran, so do not re-draft here.
                verify_input = batch.spec_info
            elif self.speculative_num_steps == 0:
                # Drafting disabled (high batch size). _draft_extend below still
                # runs, keeping draft KV warm for when the batch shrinks.
                verify_input = self._build_trivial_verify_input(batch)
            else:
                with (
                    draft_tp_context(self.draft_worker.draft_owns_attention),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                    spec_stage_span("draft"),
                ):
                    verify_input: EagleVerifyInput = self.draft_worker.draft(batch)
            assert verify_input.is_verify_input()
            batch.spec_info = verify_input
            batch_output = self.verify(
                batch,
                pp_proxy_tensors=pp_proxy_tensors,
                grammar_barrier=grammar_barrier,
            )
            # Publish before draft_extend so the fence is at verify-end.
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            if (
                self.speculative_num_steps == 0
                and envs.SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND.get()
            ):
                self._stub_skipped_draft_extend(batch, batch_output)
            else:
                with (
                    draft_tp_context(self.draft_worker.draft_owns_attention),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                    spec_stage_span("draft_extend"),
                ):
                    self.draft_worker._draft_extend_for_decode(
                        batch, batch_output, kv_loc_plan=verify_input.kv_loc_plan
                    )

            if (
                get_parallel().pp_size > 1
                and not batch.forward_mode.is_idle()
                and self.speculative_num_steps > 0
            ):
                # PP tail-draft: draft the NEXT round's chain now — earlier
                # stages must have the tokens before running their half of the
                # next verify forward, so drafting cannot wait for the next
                # iteration. Mimic the head-of-iteration state draft() expects;
                # the scheduler's forward isolation reverts these SB edits, and
                # the chain rides out on batch_output.
                batch.spec_info = batch_output.next_draft_input
                batch.seq_lens = batch_output.new_seq_lens
                batch.forward_mode = ForwardMode.DECODE
                # eagle_prepare_for_verify left the verify tokens here; the
                # head-of-iteration draft always sees None (the scheduler
                # clears it), so mirror that state.
                batch.input_ids = None
                # Attention metadata planning reads the CPU copies; one D2H
                # per round (TODO: async or upper-bound estimate).
                batch.seq_lens_cpu = batch_output.new_seq_lens.to("cpu")
                batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
                with (
                    draft_tp_context(self.draft_worker.draft_owns_attention),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                    spec_stage_span("draft"),
                ):
                    next_verify_input, parent_list, top_scores_index = (
                        self.draft_worker.draft(batch, with_topology=True)
                    )
                batch_output.next_verify_chain = next_verify_input.draft_token
                # The tree shape is data-dependent once topk > 1, so the other
                # stages cannot re-derive it; relay it alongside the tokens.
                # clone(): both come out of cuda-graph-owned buffers under
                # decode replay and would be overwritten before the relay.
                batch_output.next_verify_parent_list = parent_list.clone()
                batch_output.next_verify_top_scores_index = top_scores_index.clone()

            return batch_output

    def forward_mixed_spec(self, batch, on_publish=None, pp_proxy_tensors=None):
        """4-mode step on upstream: ONE target forward over [prefill chunk | verify rows].

        Upstream already fuses a chunk with 1-token decode tails (scheduler mixed-chunk +
        `mix_with_running`'s spec branch) and runs verify as a SEPARATE forward -- measured
        2026-10-08 via the batch log (`accept len` appears only on Decode batch lines). This method
        is the remaining delta, the one the fork's serving win measured: the chunk rides in the
        VERIFY forward instead.

        Order is forced: draft() must run before the merge, because the draft tokens are the verify
        rows' input and do not exist until then -- which is why the scheduler only ATTACHES the
        running batch (`mixed_spec_running`) rather than merging it.

        Chain draft only (topk == 1): each verify row is then a plain n-token causal extend.
        """
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        import copy

        from sglang.srt.speculative.eagle_worker_common import run_eagle_verify
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.speculative.eagle_utils import (
            eagle_prepare_for_verify,
            eagle_sample,
        )

        assert self.topk == 1, "4-mode step requires --speculative-eagle-topk 1"
        running: ScheduleBatch = batch.mixed_spec_running
        batch.mixed_spec_running = None

        with (
            draft_tp_context(self.draft_worker.draft_owns_attention),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            spec_stage_span("draft"),
        ):
            verify_input = self.draft_worker.draft(running)
        assert verify_input.is_verify_input()
        running.spec_info = verify_input

        # Allocate the n KV slots per verify row and point running.out_cache_loc at them. We want
        # only that side effect; the ForwardBatch it also builds is discarded, since the merged
        # batch below produces the real one. (One wasted plan per 4-mode step -- the first thing to
        # optimise once this is correct.)
        _sl_before_prep = (
            [int(v) for v in running.seq_lens[:3]]
            if os.environ.get("SGLANG_DBG_DRAFT_COPY") == "1" else None
        )
        _vfb, _vfb_cg = eagle_prepare_for_verify(
            verify_input, self.req_to_token_pool, running, self.target_worker
        )

        # Gate (b), v2: take the reference BEFORE the merged forward, reusing the verify batch
        # the step already prepared. v1 called eagle_prepare_for_verify again for the
        # reference, which REALLOCATES cache locs and mutates `running` -- two identical
        # references then disagreed as much as merged-vs-reference, so the gate was measuring
        # its own side effects. prepare -> forward here is exactly the normal verify path, and
        # the merged forward rewrites the same KV slots with the same values afterwards.
        _ref_ver = None
        if os.environ.get("SGLANG_MIXED_VERIFY_CHECK") == "1" and _vfb is not None:
            _ref_ver = (
                self.target_worker.forward_batch_generation(
                    batch=None, forward_batch=_vfb, is_verify=True
                )
                .logits_output.next_token_logits.detach()
                .clone()
            )

        # Merge onto a COPY, leaving `batch` as the pure prefill batch. Mutating `batch` in place
        # (upstream's merge does) meant every later consumer of the prefill rows -- the prefill draft
        # extend above all -- saw merged-size fields, and patching that field by field produced an
        # endless tail (`index 4 out of bounds`, then `tensor a (4) vs b (5)`). A shallow copy with
        # its own reqs list is enough: merge_batch reassigns attributes on the copy, and the
        # original keeps the prefill-only view the fork relied on.
        n = int(verify_input.draft_token_num)
        _sl_after_prep = (
            [int(v) for v in running.seq_lens[:3]]
            if os.environ.get("SGLANG_DBG_DRAFT_COPY") == "1" else None
        )
        merged = copy.copy(batch)
        merged.reqs = list(batch.reqs)
        merged.mix_with_verify(running, verify_input)
        if os.environ.get("SGLANG_DBG_DRAFT_COPY") == "1":
            # The three values that decide the off-by-n: the row's seq_lens before prepare,
            # after prepare, and what the merge produced for the tail. Production (OFFCONV)
            # puts a verify row's first draft token AT the committed prefix.
            logger.error(
                "SLTRACE n=%s before_prep=%s after_prep=%s merged_tail_seq=%s "
                "vfb_seq=%s vfb_pos0=%s",
                n, _sl_before_prep, _sl_after_prep,
                [int(v) for v in merged.seq_lens[-(merged.mixed_verify_rows or 1):]],
                None if getattr(_vfb, "seq_lens", None) is None
                else [int(v) for v in _vfb.seq_lens[:3]],
                None if getattr(_vfb, "positions", None) is None else int(_vfb.positions[0]),
            )
        n_pre_rows = merged.batch_size() - merged.mixed_verify_rows
        t_pre = merged.extend_num_tokens - merged.mixed_verify_tokens

        fb = ForwardBatch.init_new(
            merged,
            self.target_worker.model_runner,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            return_hidden_states_before_norm=False,
        )
        batch_output = self.target_worker.forward_batch_generation(
            batch=None, forward_batch=fb, is_verify=True, pp_proxy_tensors=pp_proxy_tensors
        )
        # Split the single forward's output. Row order follows the merged batch, which
        # mix_with_verify builds as [chunk rows | verify rows], so the prefill rows are the first
        # n_pre_rows of next_token_logits and the first t_pre of hidden_states.
        lo = batch_output.logits_output
        pre_logits = lo.next_token_logits[:n_pre_rows]
        ver_logits = lo.next_token_logits[n_pre_rows:]
        pre_hidden = ver_hidden = None
        if lo.hidden_states is not None:
            # Bound the verify slice explicitly. These are FULL per-token states, and under a
            # prefill-graph bucket the tensor is PADDED past the real token count, so an
            # open-ended [t_pre:] handed the verify draft extend padding rows as hidden
            # states -- corrupting the next draft tokens silently instead of failing.
            pre_hidden = lo.hidden_states[:t_pre]
            ver_hidden = lo.hidden_states[t_pre : t_pre + merged.mixed_verify_tokens]
        pre_out = LogitsProcessorOutput(
            next_token_logits=pre_logits, hidden_states=pre_hidden
        )
        ver_out = LogitsProcessorOutput(
            next_token_logits=ver_logits, hidden_states=ver_hidden
        )

        if _ref_ver is not None:
            def _err(a, b):
                a = a.float(); b = b.float()
                if tuple(a.shape) != tuple(b.shape):
                    return float("nan")
                return float((a - b).abs().max() / a.abs().max().clamp_min(1e-6))
            _flips = lambda a, b: (
                int((a.float().argmax(-1) != b.float().argmax(-1)).sum())
                if tuple(a.shape) == tuple(b.shape) else -1
            )
            # Null case: the same comparison on the prefill rows, which are known good.
            # The verdict is only readable while this reads 0.
            _pre_err = _pre_flips = None
            try:
                _pfb = ForwardBatch.init_new(
                    batch,
                    self.target_worker.model_runner,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    return_hidden_states_before_norm=False,
                )
                _pref = self.target_worker.forward_batch_generation(
                    batch=None, forward_batch=_pfb, is_verify=True
                ).logits_output.next_token_logits
                _pre_err = _err(_pref, pre_logits)
                _pre_flips = _flips(_pref, pre_logits)
            except Exception as _e:
                _pre_err = repr(_e)[:60]
            # Per-POSITION agreement: a verify row holds n draft tokens, and only the
            # accepted prefix affects emitted text. Which position flips localises the fault --
            # position 0 wrong means the row's prefix KV length is wrong (seq_lens convention);
            # only positions >= 1 wrong means the intra-row causal structure or the RoPE
            # offsets (positions must be prefix + i).
            _nrows = ver_logits.shape[0]
            _n = int(verify_input.draft_token_num)
            _ar = _ref_ver.float().argmax(-1)
            _ag = ver_logits.float().argmax(-1)
            _bypos = [
                int((_ar[i::_n] != _ag[i::_n]).sum()) if _nrows >= _n else -1
                for i in range(_n)
            ]
            _errpos = [
                round(float((_ref_ver[i::_n].float() - ver_logits[i::_n].float()).abs().max()
                            / _ref_ver[i::_n].float().abs().max().clamp_min(1e-6)), 4)
                if _nrows >= _n else -1.0
                for i in range(_n)
            ]
            # The verify rows' field conventions, merged batch vs the reference verify batch.
            # EXTEND wants seq_lens = prefix + n with extend_prefix_lens = prefix;
            # TARGET_VERIFY keeps seq_lens at the committed prefix and masks the tree itself.
            def _tail(x, k):
                if x is None:
                    return None
                try:
                    return [int(v) for v in (x[-k:] if hasattr(x, "__getitem__") else [])]
                except Exception:
                    return "?"
            _k = merged.mixed_verify_rows or 1
            logger.error(
                "CONV merged: seq_lens=%s ext_prefix=%s ext_seq=%s pos0=%s || "
                "ref: seq_lens=%s ext_prefix=%s ext_seq=%s pos0=%s",
                _tail(getattr(fb, "seq_lens", None), _k),
                _tail(getattr(fb, "extend_prefix_lens", None), _k),
                _tail(getattr(fb, "extend_seq_lens", None), _k),
                None if getattr(fb, "positions", None) is None else int(fb.positions[-_k * _n]),
                _tail(getattr(_vfb, "seq_lens", None), _k),
                _tail(getattr(_vfb, "extend_prefix_lens", None), _k),
                _tail(getattr(_vfb, "extend_seq_lens", None), _k),
                None if getattr(_vfb, "positions", None) is None else int(_vfb.positions[0]),
            )

            # accept_lens is not available yet here (run_eagle_verify runs below); it is
            # logged per step by MIXEDSUM.
            logger.error(
                "BYPOS n=%s rows=%s flips_by_pos=%s err_by_pos=%s",
                _n, _nrows, _bypos, _errpos,
            )
            logger.error(
                "VERIFYCHECK3 ver_rows=%s rows=%s | NULL prefill err=%s flips=%s | "
                "merged_vs_ref err=%.3e flips=%s/%s | scale ref=%.3f got=%.3f",
                merged.mixed_verify_rows, tuple(ver_logits.shape),
                _pre_err, _pre_flips,
                _err(_ref_ver, ver_logits), _flips(_ref_ver, ver_logits),
                ver_logits.shape[0],
                float(_ref_ver.float().abs().max()),
                float(ver_logits.float().abs().max()),
            )

        if os.environ.get("SGLANG_DBG_DRAFT_COPY") == "1":
            logger.error(
                "MIXEDPRE pre_reqs=%d run_reqs=%d merged_bs=%s ver_rows=%s ver_toks=%s n=%s "
                "n_pre_rows=%d t_pre=%d ntl=%s pre_logits=%s ver_logits=%s dtn=%s",
                len(batch.reqs), len(running.reqs), merged.batch_size(),
                merged.mixed_verify_rows, merged.mixed_verify_tokens, n,
                n_pre_rows, t_pre, tuple(lo.next_token_logits.shape),
                tuple(pre_logits.shape), tuple(ver_logits.shape),
                verify_input.draft_token_num,
            )
            # Checksums localise run-to-run nondeterminism: if these already differ between
            # two runs at the same step index, the merged forward (or its batch) is the
            # source; if they match while outputs diverge, the fault is downstream in
            # accept/commit/draft-extend.
            logger.error(
                "MIXEDSUM step_ver_rows=%s ver_logits_sum=%.6f ver_hidden_sum=%.6f "
                "pre_logits_sum=%.6f seq_lens=%s",
                merged.mixed_verify_rows,
                float(ver_logits.float().sum()),
                -1.0 if ver_hidden is None else float(ver_hidden.float().sum()),
                float(pre_logits.float().sum()),
                running.seq_lens.tolist(),
            )
            logger.error(
                "MIXEDHS hs=%s pre_hidden=%s ver_hidden=%s",
                None if lo.hidden_states is None else tuple(lo.hidden_states.shape),
                None if pre_hidden is None else tuple(pre_hidden.shape),
                None if ver_hidden is None else tuple(ver_hidden.shape),
            )

        # Verify rows: hand the slice to upstream's own accept path, paired with the RUNNING batch
        # (eagle_sample reads batch.seq_lens and verify_input fields sized to that batch, not to
        # the merged one).
        running.forward_mode = ForwardMode.TARGET_VERIFY
        # Hand-rolling sample + commit + result meant chasing every field upstream sets, one
        # failed run each: accept_lens, new_seq_lens, bonus_tokens, next_draft_input, ... So run
        # upstream's OWN verify with the merged forward's verify slice injected, and let it do
        # the whole post-forward half.
        running.spec_info = verify_input
        ver_result = run_eagle_verify(
            running,
            target_worker=self.target_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            plan_stream=self.plan_stream,
            plan_stream_ctx=self.plan_stream_ctx,
            topk=self.topk,
            num_draft_tokens=self.speculative_num_draft_tokens,
            device=self.device,
            metadata_ready_pre_pad=False,
            finalize_tree_path=True,
            precomputed_logits_output=ver_out,
            precomputed_forward_output=batch_output,
        )
        # Sample the prefill rows with the PREFILL batch's own sampling_info, not the merged
        # one. mix_with_verify snapshots it as mixed_pre_sampling_info precisely because the
        # merge overwrites self.sampling_info with a merged-width object; this swap was lost
        # in the commit-29 restructure, leaving the merged info in place.
        _saved_si = fb.sampling_info
        if merged.mixed_pre_sampling_info is not None:
            fb.sampling_info = merged.mixed_pre_sampling_info
        try:
            pre_next_token_ids = self.target_worker.model_runner.sample(pre_out, fb)
        finally:
            fb.sampling_info = _saved_si
        result = GenerationBatchResult(
            logits_output=pre_out,
            next_token_ids=pre_next_token_ids,
            can_run_cuda_graph=False,
        )
        # Spec_v2 convention (see _forward_prefill_batch): a prefill row's new_seq_lens is
        # batch.seq_lens. on_publish is bound to THIS batch's req_pool_indices, so it covers
        # the prefill half only; the scheduler publishes the verify half against its own rows.
        result.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(batch.seq_lens)
        result.mixed_running_batch = running
        result.mixed_running_result = ver_result
        # The scheduler parks ONLY the returned result's refs (scheduler.py:4481), and we
        # return the prefill half -- so the verify half's keep-alive refs were stranded and
        # its verify-time GPU tensors (draft_token / out_cache_loc, held transitively by the
        # verify ForwardBatch) could be recycled inside the 2-iter window. Measured as
        # run-to-run NONDETERMINISM in the 4-mode arm on requests the control reproduces
        # exactly. The merged ForwardBatch is pinned for the same reason.
        _refs = list(result.extra_keep_alive_refs or [])
        _refs.append(fb)
        _refs.extend(ver_result.extra_keep_alive_refs or [])
        result.extra_keep_alive_refs = _refs
        with (
            draft_tp_context(self.draft_worker.draft_owns_attention),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            spec_stage_span("draft_extend"),
        ):
            # `batch` is still the pure prefill batch (the merge went onto `merged`), so this needs
            # no snapshot/restore dance.
            result.next_draft_input = self.draft_worker._draft_extend_for_prefill(
                batch, pre_out.hidden_states, pre_next_token_ids
            )
            # No tensor narrowing here. Once `next_token_ids = predict` makes the draft-extend batch
            # report n tokens (raw_num_tokens = input_ids.shape[0]), EVERY field is correctly n-wide
            # and the earlier out_cache_loc narrowing (commits 19/21) becomes the bug -- the two are
            # mutually exclusive, and keeping both active reproduced the same shape error from the
            # opposite side.
            # positions: NOT settable here -- `prepare_for_draft_extend` overwrites it n-wide
            # unconditionally (assigning it accept-wide was a no-op; the mismatch stayed
            # `positions (dst 1, src 4)`). So the n-wide field is intended, and the real
            # inconsistency is that the draft-extend batch reports a TOKEN COUNT of 1 (raw_n=1,
            # which sizes the graph buffer) while its fields are built n-wide from
            # num_tokens_per_req = speculative_num_draft_tokens. Next: find what the runner derives
            # raw_num_tokens from for this batch (input_ids.numel() / extend_num_tokens /
            # num_accept_tokens) and make it bs*n, rather than resizing the tensors.
            self.draft_worker._draft_extend_for_decode(
                running, result.mixed_running_result
            )
        if os.environ.get("SGLANG_DBG_DRAFT_COPY") == "1":
            def _sh(x):
                return None if x is None else tuple(x.shape)
            _pre_di = result.next_draft_input
            _run_di = getattr(result.mixed_running_result, "next_draft_input", None)
            logger.error(
                "MIXEDSPLIT pre_reqs=%d run_reqs=%d merged_bs=%s ver_rows=%s ver_toks=%s "
                "n_pre_rows=%d t_pre=%d ntl=%s pre_ids=%s pre_di_topk_p=%s run_di_topk_p=%s "
                "accept_lens=%s",
                len(batch.reqs), len(running.reqs), merged.batch_size(),
                merged.mixed_verify_rows, merged.mixed_verify_tokens,
                n_pre_rows, t_pre, _sh(lo.next_token_logits), _sh(pre_next_token_ids),
                _sh(getattr(_pre_di, "topk_p", None)),
                _sh(getattr(_run_di, "topk_p", None)),
                getattr(result.mixed_running_result, "accept_lens", None),
            )
        return result

    def _forward_prefill_batch(
        self, batch, on_publish=None, pp_proxy_tensors=None, coordination_plan=None
    ):
        # Target prefill
        target_capture_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )
        batch_output = self.target_worker.forward_batch_generation(
            batch,
            pp_proxy_tensors=pp_proxy_tensors,
            capture_hidden_mode=target_capture_mode,
            return_kv_loc_plan=self._draft_worker is not None,
        )
        kv_loc_plan, batch_output.kv_loc_plan = batch_output.kv_loc_plan, None

        # Spec_v2 convention: batch.seq_lens = length BEFORE this iter's tokens.
        # Extend processed L prompt tokens; next verify iter expects same L.
        batch_output.new_seq_lens = batch.seq_lens
        # Publish before draft_extend so the fence is at target-end.
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        # A rank that does not host the draft (prefill-side PP builds it only on
        # the last stage) forwards the target's proxy tensors and stops here.
        if self._draft_worker is None:
            return batch_output

        if coordination_plan is not None:
            coordination_plan.apply(
                batch,
                "draft_extend",
                get_parallel().attn_dp_rank,
                local_only=(
                    len(batch.global_num_tokens) == 1
                    or self.draft_worker.draft_owns_attention
                ),
            )

        # Draft prefill
        with (
            draft_tp_context(self.draft_worker.draft_owns_attention),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            spec_stage_span("draft_extend"),
        ):
            batch_output.next_draft_input = self.draft_worker._draft_extend_for_prefill(
                batch,
                batch_output.logits_output.hidden_states,
                batch_output.next_token_ids,
                batch_output.logits_output.mm_input_embeds,
                kv_loc_plan=kv_loc_plan,
            )
            return batch_output

    def _forward_dp_spec_prefill_coordination(
        self, batch, plan, on_publish, grammar_barrier, pp_proxy_tensors
    ):
        """Run draft, target, and draft-extend with each rank's local mode."""
        rank = get_parallel().attn_dp_rank
        target_local_only = len(batch.global_num_tokens) == 1
        # The draft MoE A2A backend may gather differently from the target.
        draft_local_only = (
            target_local_only
            if batch.draft_global_num_tokens is None
            else len(batch.draft_global_num_tokens) == 1
        ) or self.draft_worker.draft_owns_attention
        is_prefill = batch.forward_mode.is_extend()
        draft_batch = batch
        if is_prefill:
            draft_batch = ScheduleBatch.init_new(
                [],
                batch.req_to_token_pool,
                batch.token_to_kv_pool_allocator,
                batch.tree_cache,
                batch.model_config,
                batch.enable_overlap,
                batch.spec_algorithm,
            )
            draft_batch.prepare_for_idle()
            draft_batch.global_num_tokens = batch.global_num_tokens
            draft_batch.global_num_tokens_for_logprob = (
                batch.global_num_tokens_for_logprob
            )
        if draft_batch.spec_info is None:
            hidden_size, hidden_dtype = get_draft_recurrent_hidden_state_spec(
                self.draft_worker.draft_runner
            )
            draft_batch.spec_info = EagleDraftInput.create_idle_input(
                device=self.device,
                hidden_size=hidden_size,
                dtype=hidden_dtype,
                topk=self.topk,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            )
        plan.apply(draft_batch, "draft", rank, local_only=draft_local_only)
        with (
            draft_tp_context(self.draft_worker.draft_owns_attention),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            spec_stage_span("draft"),
        ):
            verify_input = self.draft_worker.draft(draft_batch)

        plan.apply(batch, "target", rank, local_only=target_local_only)
        if is_prefill:
            result = self._forward_prefill_batch(
                batch, on_publish, pp_proxy_tensors, coordination_plan=plan
            )
            # Pin the temporary idle tensors through the overlap lifetime.
            result.extra_keep_alive_refs = list(result.extra_keep_alive_refs or ()) + [
                draft_batch
            ]
            return result

        batch.spec_info = verify_input
        result = self.verify(batch, grammar_barrier=grammar_barrier)
        if on_publish is not None:
            on_publish(result.new_seq_lens)
        plan.apply(batch, "draft_extend", rank, local_only=draft_local_only)
        with (
            draft_tp_context(self.draft_worker.draft_owns_attention),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            spec_stage_span("draft_extend"),
        ):
            self.draft_worker._draft_extend_for_decode(batch, result)
        return result

    def _build_trivial_verify_input(self, batch: ScheduleBatch) -> EagleVerifyInput:
        """Build a 1-node EagleVerifyInput rooted at the previous bonus token.

        Used when ``speculative_num_steps == 0`` to skip drafting while still
        routing through the existing TARGET_VERIFY graph captured at
        ``draft_token_num=1``: the kernel always accepts the root and samples
        one new bonus token from target logits -- functionally a plain decode.
        """
        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                topk=self.topk, spec_steps=0, num_verify_tokens=1, device=self.device
            )

        draft_input: EagleDraftInput = batch.spec_info
        bs = batch.seq_lens.shape[0]
        device = self.device

        retrieve_index = torch.arange(bs, dtype=torch.long, device=device).unsqueeze(1)
        retrieve_next_token = torch.full((bs, 1), -1, dtype=torch.long, device=device)
        retrieve_next_sibling = torch.full((bs, 1), -1, dtype=torch.long, device=device)

        attn_backend = self._target_worker.model_runner.attn_backend
        verify_mask = attn_backend.verify_mask
        # Every position in a 1-node tree is visible, so an all-True fill is
        # correct under either layout.
        if verify_mask is not None and verify_mask.fits(bs):
            custom_mask = verify_mask.buffer
            custom_mask.fill_(True)
        else:
            if batch.seq_lens_sum is not None:
                seq_lens_sum = batch.seq_lens_sum
            elif batch.seq_lens_cpu is not None:
                seq_lens_sum = int(batch.seq_lens_cpu.sum())
            else:
                seq_lens_sum = bs * attn_backend.max_context_len
            custom_mask = torch.ones(seq_lens_sum + bs, dtype=torch.bool, device=device)

        positions = batch.seq_lens.to(torch.int64)

        return EagleVerifyInput(
            draft_token=draft_input.bonus_tokens,
            custom_mask=custom_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=0,
            topk=self.topk,
            draft_token_num=1,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def _stub_skipped_draft_extend(
        self, batch: ScheduleBatch, batch_output: GenerationBatchResult
    ) -> None:
        """Fill shape-valid stubs on next_draft_input when draft_extend is skipped.

        ``verify`` already set ``bonus_tokens`` (the only field the next steps=0
        verify reads). The overlap FutureMap still stashes topk_p/topk_index/
        hidden_states, so provide zeroed tensors of the right shape. They are never
        consumed while at steps=0; an upshift to steps>0 would draft from this stale
        state (cold recovery), which is the documented cost of this experimental flag.
        """
        next_draft_input: EagleDraftInput = batch_output.next_draft_input
        bs = batch.seq_lens.shape[0]
        device = self.device
        next_draft_input.topk_p = torch.zeros(
            (bs, self.topk), dtype=torch.float32, device=device
        )
        next_draft_input.topk_index = torch.zeros(
            (bs, self.topk), dtype=torch.int64, device=device
        )
        hidden_size, hidden_dtype = get_draft_recurrent_hidden_state_spec(
            self.draft_worker.draft_runner
        )
        if hidden_size is not None:
            next_draft_input.hidden_states = torch.zeros(
                (bs, hidden_size),
                dtype=hidden_dtype,
                device=device,
            )

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.on_verify_complete(
                num_correct_drafts_per_req, batch_size=batch_size
            )

    def activate_step_by_batch(self, batch_size: int) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.activate_step_by_batch(batch_size)

    # -- Adaptive speculative decoding protocol --

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs=None,
    ) -> SpecRuntimeState:
        """Build a SpecRuntimeState for the given step configuration."""
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)

        with self._override_worker_state(
            speculative_num_steps,
            speculative_num_draft_tokens,
            cuda_graph_bs=cuda_graph_bs,
        ):
            self._draft_worker.init_attention_backend()
            self._draft_worker._capture_cuda_graphs()

            # Build target attention backend and CUDA graph runner
            target_model_runner = self._target_worker.model_runner
            backup_init = target_model_runner.init_new_workspace
            try:
                target_attn_backend = target_model_runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                target_model_runner.init_new_workspace = backup_init

            target_graph_runner = None
            if not check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
                TargetGraphRunnerCls = (
                    NPUGraphRunner if _is_npu else DecodeCudaGraphRunner
                )
                target_graph_before_mem = get_available_gpu_memory(
                    self.device, self.gpu_id
                )
                target_graph_tic = time.perf_counter()
                target_graph_runner = TargetGraphRunnerCls(
                    target_model_runner,
                    attn_backend=target_attn_backend,
                    speculative_num_steps=speculative_num_steps,
                    speculative_num_draft_tokens=speculative_num_draft_tokens,
                )
                target_graph_after_mem = get_available_gpu_memory(
                    self.device, self.gpu_id
                )
                target_graph_time = time.perf_counter() - target_graph_tic
                self._additional_graph_memory_usage["target_verify"] = (
                    self._additional_graph_memory_usage.get("target_verify", 0.0)
                    + target_graph_before_mem
                    - target_graph_after_mem
                )
                self._additional_graph_time_usage["target_verify"] = (
                    self._additional_graph_time_usage.get("target_verify", 0.0)
                    + target_graph_time
                )

            state = SpecRuntimeState(
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
                draft_attn_backend=self._draft_worker.draft_attn_backend,
                cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                target_attn_backend=target_attn_backend,
                target_graph_runner=target_graph_runner,
                draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                cuda_graph_runner_for_draft_extend=self._draft_worker.cuda_graph_runner_for_draft_extend,
            )

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        log_info_on_rank0(
            logger,
            f"Built adaptive runtime state steps={speculative_num_steps}: "
            f"elapsed={time.perf_counter() - tic:.2f}s, "
            f"mem={(before_mem - after_mem):.2f}GB",
        )

        return state

    def apply_runtime_state(self, state: SpecRuntimeState) -> None:
        """Apply a pre-built runtime state to this worker."""
        if self.speculative_num_steps == state.speculative_num_steps:
            return

        log_info_on_rank0(
            logger,
            "Switch adaptive runtime state: "
            f"steps {self.speculative_num_steps} -> {state.speculative_num_steps}, "
            f"draft_tokens {self.speculative_num_draft_tokens} -> "
            f"{state.speculative_num_draft_tokens}",
        )

        # Top-level
        self.speculative_num_steps = state.speculative_num_steps
        self.speculative_num_draft_tokens = state.speculative_num_draft_tokens

        # Draft side
        dw = self._draft_worker
        dw.speculative_num_steps = state.speculative_num_steps
        dw.speculative_num_draft_tokens = state.speculative_num_draft_tokens
        dw.draft_attn_backend = state.draft_attn_backend
        dw.draft_runner.draft_attn_backend = state.draft_attn_backend
        dw.cuda_graph_runner = state.cuda_graph_runner
        dw.draft_extend_attn_backend = state.draft_extend_attn_backend
        # Keep the runner's attn_backend in step with the active draft-extend
        # backend (the draft-extend forward reads draft_runner.attn_backend);
        # mirrors init_attention_backend. When None, the runner keeps its
        # initialized backend (consistent across step configs).
        if state.draft_extend_attn_backend is not None:
            dw.draft_runner.attn_backend = state.draft_extend_attn_backend
        dw.cuda_graph_runner_for_draft_extend = state.cuda_graph_runner_for_draft_extend
        dw._rebuild_topk1_chain_buffers()

        # Target side
        self._target_worker.model_runner.attn_backend = state.target_attn_backend
        self._target_worker.model_runner.decode_cuda_graph_runner = (
            state.target_graph_runner
        )

        # Sync server_args
        get_context().override(
            "adaptive_spec.restore",
            speculative_num_steps=state.speculative_num_steps,
            speculative_num_draft_tokens=state.speculative_num_draft_tokens,
        )

    @contextlib.contextmanager
    def _override_worker_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ):
        """Temporarily override server_args and worker attributes for graph capture."""
        dw = self._draft_worker
        backup = (
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            dw.speculative_num_steps,
            dw.speculative_num_draft_tokens,
            dw.draft_attn_backend,
            dw.draft_extend_attn_backend,
            dw.draft_runner.draft_attn_backend,
            dw.draft_runner.attn_backend,
            dw.cuda_graph_runner,
            dw.cuda_graph_runner_for_draft_extend,
            get_spec().speculative_num_steps,
            get_spec().speculative_num_draft_tokens,
            get_exec().graph.cuda_graph_bs_decode,
            get_exec().graph.disable_cuda_graph,
        )

        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        dw.speculative_num_steps = speculative_num_steps
        dw.speculative_num_draft_tokens = speculative_num_draft_tokens
        get_context().override(
            "adaptive_spec.capture_override",
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        if cuda_graph_bs is not None:
            # BS-aware adaptive spec may prune cuda_graph_bs to an empty list
            # for steps that no BS range uses (e.g. step=1). Disable graph
            # capture for those steps; restore in finally so subsequent steps
            # are not affected.
            get_context().override(
                "adaptive_spec.capture_override",
                cuda_graph_bs_decode=cuda_graph_bs,
                **({"disable_cuda_graph": True} if not cuda_graph_bs else {}),
            )
        dw._rebuild_topk1_chain_buffers()

        try:
            yield
        finally:
            (
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                dw.speculative_num_steps,
                dw.speculative_num_draft_tokens,
                dw.draft_attn_backend,
                dw.draft_extend_attn_backend,
                dw.draft_runner.draft_attn_backend,
                dw.draft_runner.attn_backend,
                dw.cuda_graph_runner,
                dw.cuda_graph_runner_for_draft_extend,
            ) = backup[:10]
            get_context().override(
                "adaptive_spec.capture_restore",
                speculative_num_steps=backup[10],
                speculative_num_draft_tokens=backup[11],
                cuda_graph_bs_decode=backup[12],
                disable_cuda_graph=backup[13],
            )
            dw._rebuild_topk1_chain_buffers()

    def verify(self, batch: ScheduleBatch, pp_proxy_tensors=None, grammar_barrier=None):
        return run_eagle_verify(
            batch,
            pp_proxy_tensors=pp_proxy_tensors,
            target_worker=self.target_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            plan_stream=self.plan_stream,
            plan_stream_ctx=self.plan_stream_ctx,
            topk=self.topk,
            num_draft_tokens=self.speculative_num_draft_tokens,
            device=self.device,
            metadata_ready_pre_pad=False,
            finalize_tree_path=True,
            grammar_barrier=grammar_barrier,
        )
