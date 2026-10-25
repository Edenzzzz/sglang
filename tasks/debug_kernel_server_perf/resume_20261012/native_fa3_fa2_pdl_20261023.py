"""Opt-in page-one BF16 FA3-prefill/native-FA2-row PDL serving executor."""

import itertools
import logging
import os
from types import SimpleNamespace


logger = logging.getLogger(__name__)


def float_workspace_overflow(error):
    message = str(error)
    return message.startswith("Error in function 'aligned_alloc'") and (
        "Buffer overflow when allocating memory for batch_prefill_tmp_v" in message
        or "Buffer overflow when allocating memory for batch_prefill_tmp_s" in message
    )


def grown_workspace_size(current, required, maximum):
    if any(type(value) is not int for value in (current, required, maximum)):
        raise ValueError("Workspace capacities must be host integers")
    if current <= 0 or required <= current or maximum < current:
        raise ValueError("Invalid workspace growth request")
    if required > maximum:
        raise ValueError(f"Native FA2 requires {required} scratch bytes, above the configured ceiling {maximum}")
    quantum = 64 << 20
    return min(maximum, ((required + quantum - 1) // quantum) * quantum)


def announce_backend(backend, source):
    logger.warning(
        "NATIVE_PDL_BACKEND source=%s draft=%s page_size=%d fa_version=%d heads=%d kv_heads=%d",
        source, backend.is_draft_runner, backend.page_size, backend.fa_impl_ver,
        backend.num_attention_heads, backend.num_kv_heads,
    )


def announce_metadata(backend, forward_batch):
    modes = getattr(backend, "_native_pdl_seen_modes", set())
    mode = forward_batch.forward_mode
    key = (mode, bool(getattr(forward_batch, "mixed_verify_tokens", 0)))
    if key not in modes:
        modes.add(key)
        backend._native_pdl_seen_modes = modes
        logger.warning(
            "NATIVE_PDL_METADATA mode=%s draft=%s verify_tokens=%s queries=%s prefixes=%s",
            mode, backend.is_draft_runner, getattr(forward_batch, "mixed_verify_tokens", None),
            forward_batch.extend_seq_lens_cpu, forward_batch.extend_prefix_lens_cpu,
        )


def partition_lengths(query_lengths, prefix_lengths, heads, kv_heads, packed_query_limit=32):
    if heads <= 0 or kv_heads <= 0 or heads % kv_heads:
        raise ValueError("Invalid GQA head counts")
    if len(query_lengths) != len(prefix_lengths) or not query_lengths:
        raise ValueError("Missing or mismatched host query/prefix lengths")
    if any(type(length) is not int or length <= 0 for length in query_lengths):
        raise ValueError("Query lengths must be positive host integers")
    if any(type(length) is not int or length < 0 for length in prefix_lengths):
        raise ValueError("Prefix lengths must be nonnegative host integers")
    if packed_query_limit not in (32, 64):
        raise ValueError("Native row limit must be 32 or 64 packed queries")
    maximum_row_query = packed_query_limit // (heads // kv_heads)
    prefix_rows = next(
        (index for index, length in enumerate(query_lengths) if length <= maximum_row_query),
        len(query_lengths),
    )
    if prefix_rows in (0, len(query_lengths)):
        return None
    if any(length > maximum_row_query for length in query_lengths[prefix_rows:]):
        return None
    kv_lengths = [prefix + query for prefix, query in zip(prefix_lengths, query_lengths)]
    if sum(kv_lengths) >= 2**31 or sum(query_lengths) >= 2**31:
        raise ValueError("Int32 attention metadata would overflow")
    return SimpleNamespace(
        prefix_rows=prefix_rows,
        prefix_tokens=sum(query_lengths[:prefix_rows]),
        total_tokens=sum(query_lengths),
        maximum_prefix_query=max(query_lengths[:prefix_rows]),
        maximum_row_query=max(query_lengths[prefix_rows:]),
        query_lengths=list(query_lengths),
        kv_lengths=kv_lengths,
    )


def geometry_split_policy(shape, heads, sms):
    if heads <= 0 or sms <= 0:
        raise ValueError("Positive head and SM counts are required")
    prefix_queries = shape.query_lengths[:shape.prefix_rows]
    prefix_lengths = shape.kv_lengths[:shape.prefix_rows]
    tiles = sum((length + 127) // 128 for length in prefix_queries) * heads
    maximum_prefix = max(prefix_lengths)
    prefix_splits = 1
    if shape.prefix_tokens <= 512 and maximum_prefix >= 8192:
        def score(splits):
            waves = (tiles * splits + sms - 1) // sms
            kv_tiles = (maximum_prefix + splits * 128 - 1) // (splits * 128)
            return waves * kv_tiles, splits

        prefix_splits = min((1, 2, 4, 8), key=score)
    target_chunk = max(shape.kv_lengths[shape.prefix_rows:]) // 32
    row_split_size = 1024
    while row_split_size < 8192 and row_split_size * 2 <= target_chunk:
        row_split_size *= 2
    return dict(policy="geometry_v1", prefix_splits=prefix_splits, row_split_size=row_split_size)


def geometry_split_policy_v2(shape, heads, sms):
    selected = geometry_split_policy(shape, heads, sms)
    queries = shape.query_lengths[:shape.prefix_rows]
    lengths = shape.kv_lengths[:shape.prefix_rows]
    dominant = max(range(shape.prefix_rows), key=lambda index: queries[index] * lengths[index])
    splits = 1
    if queries[dominant] <= 512 and lengths[dominant] >= 8192:
        tiles = sum((length + 127) // 128 for length in queries) * heads
        maximum_prefix = max(lengths)

        def score(count):
            waves = (tiles * count + sms - 1) // sms
            kv_tiles = (maximum_prefix + count * 128 - 1) // (count * 128)
            return waves * kv_tiles, count

        splits = min((1, 2, 4, 8, 16, 32), key=score)
    return dict(
        policy="geometry_v2", prefix_splits=splits, row_split_size=selected["row_split_size"],
        dominant_prefill_query=queries[dominant], dominant_prefill_kv=lengths[dominant],
    )


def geometry_split_policy_v3(shape, heads, kv_heads, sms):
    if kv_heads <= 0 or heads % kv_heads:
        raise ValueError("Invalid packed-GQA geometry")
    selected = geometry_split_policy_v2(shape, heads, sms)
    queries = shape.query_lengths[:shape.prefix_rows]
    lengths = shape.kv_lengths[:shape.prefix_rows]
    group = heads // kv_heads
    tiles = sum((length * group + 127) // 128 for length in queries) * kv_heads
    splits = 1
    if selected["dominant_prefill_query"] <= 512 and selected["dominant_prefill_kv"] >= 8192:
        maximum_prefix = max(lengths)

        def score(count):
            waves = (tiles * count + sms - 1) // sms
            kv_tiles = (maximum_prefix + count * 128 - 1) // (count * 128)
            return waves * kv_tiles, count

        splits = min((1, 2, 4, 8, 16, 32), key=score)
    return dict(selected, policy="geometry_v3", prefix_splits=splits, packed_prefill_tiles=tiles)


class NativePdlExecutor:
    def __init__(self, heads, kv_heads, device, sm_budget=None, verify=False, prefix_splits=1, row_split_size=None, workspace_bytes=128 << 20, adaptive_splits=False, policy_version="geometry_v1", fp32_partials=False, direct_bf16_output=False, packed_query_limit=None, max_workspace_bytes=None):
        import torch
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        self.torch = torch
        self.heads, self.kv_heads, self.device = heads, kv_heads, device
        self.packed_query_limit = int(os.environ.get("NATIVE_PDL_PACKED_QUERY_LIMIT", "32")) if packed_query_limit is None else packed_query_limit
        if self.packed_query_limit not in (32, 64):
            raise ValueError("Native row limit must be 32 or 64 packed queries")
        self.verify = verify
        self.fp32_partials = fp32_partials
        self.direct_bf16_output = direct_bf16_output
        if direct_bf16_output and not fp32_partials:
            raise ValueError("Direct BF16 merge requires FP32 partials")
        self.row_namespace = "native_fa2_q32_fp32" if fp32_partials else "native_fa2_q32"
        if not 1 <= prefix_splits <= 32 or (row_split_size is not None and row_split_size <= 0):
            raise ValueError("Invalid native SplitKV configuration")
        self.prefix_splits, self.row_split_size = prefix_splits, row_split_size
        self.adaptive_splits = adaptive_splits
        if policy_version not in ("geometry_v1", "geometry_v2", "geometry_v3"):
            raise ValueError("Unknown adaptive native split policy")
        self.policy_version = policy_version
        # SGLANG_ROLE_SPLIT_FA3=1 (2026-10-23): run the mixed step as TWO FA3 launches -- prefill rows on a side stream,
        # decode/verify rows on the current stream with num_splits=SGLANG_ROLE_SPLIT_ROW_SPLITS -- instead of FA3 + FA2.
        # On 96 trace-proportional H200 captures this fixed role split was 1-2% faster than the FA3+FA2 path (and
        # 1.23-1.57x faster than sglang's single varlen FA3 call). No FA2 planning is done in this mode.
        self.role_split = os.environ.get("SGLANG_ROLE_SPLIT_FA3") == "1"
        self.role_row_splits = int(os.environ.get("SGLANG_ROLE_SPLIT_ROW_SPLITS", "32"))
        self.side_stream = torch.cuda.Stream(device=device) if self.role_split else None
        self.calls = 0
        self.plans = 0
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.sm_budget = sm_budget if sm_budget is not None else sms
        if not 0 < self.sm_budget <= sms:
            raise ValueError("FA3 SM budget exceeds device capacity")
        for namespace, variable in (
            ("stock_fa3_pdl", "NATIVE_PDL_FA3_LIBRARY"),
            (self.row_namespace, "NATIVE_PDL_FA2_FP32_LIBRARY" if fp32_partials else "NATIVE_PDL_FA2_LIBRARY"),
        ):
            if not hasattr(getattr(torch.ops, namespace), "run"):
                torch.ops.load_library(os.environ[variable])
        if direct_bf16_output and not hasattr(torch.ops.native_fa2_q32_fp32_direct, "run"):
            torch.ops.load_library(os.environ["NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY"])
        if workspace_bytes <= 0:
            raise ValueError("Native workspace size must be positive")
        self.max_workspace_bytes = workspace_bytes if max_workspace_bytes is None else max_workspace_bytes
        if self.max_workspace_bytes < workspace_bytes:
            raise ValueError("Maximum native workspace is below its initial size")
        self.workspace_grows = 0
        self.workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, kv_layout="NHD", backend="fa2"
        )
        self.plan_copy_done = None
        self.empty_lse = torch.empty(0, dtype=torch.float32, device=device)
        self.empty_timeline = torch.empty(0, dtype=torch.int64, device=device)

    def _plan_rows(self, arguments, options):
        try:
            self.wrapper.plan(*arguments, **options)
            return
        except Exception as error:
            if not float_workspace_overflow(error) or self.workspace.numel() >= self.max_workspace_bytes:
                raise
            if self.torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Native workspace cannot grow inside graph capture") from error
            required_float, required_integer = self.wrapper.workspace_size(*arguments, **options)
            if required_integer > self.wrapper._int_workspace_buffer.numel():
                raise RuntimeError("Native FA2 integer workspace is too small") from error
            capacity = grown_workspace_size(self.workspace.numel(), int(required_float), self.max_workspace_bytes)
            replacement = self.torch.empty(capacity, dtype=self.torch.uint8, device=self.device)
            self.wrapper.reset_workspace_buffer(replacement, self.wrapper._int_workspace_buffer)
            previous = self.workspace.numel()
            self.workspace = replacement
            self.workspace_grows += 1
            logger.warning("NATIVE_PDL_WORKSPACE_GROW previous=%d required=%d allocated=%d ceiling=%d", previous, required_float, capacity, self.max_workspace_bytes)
        self.wrapper.plan(*arguments, **options)

    def plan(self, query_lengths, prefix_lengths, page_table, cache_lengths, cumulative_query):
        torch = self.torch
        shape = partition_lengths(query_lengths, prefix_lengths, self.heads, self.kv_heads, self.packed_query_limit)
        if shape is None:
            return None
        selector = geometry_split_policy if self.policy_version == "geometry_v1" else geometry_split_policy_v2
        shape.split_settings = (
            selector(shape, self.heads, self.sm_budget) if self.adaptive_splits
            else dict(policy="explicit", prefix_splits=self.prefix_splits, row_split_size=self.row_split_size)
        )
        if self.adaptive_splits and self.policy_version == "geometry_v3":
            shape.split_settings = geometry_split_policy_v3(shape, self.heads, self.kv_heads, self.sm_budget)
        shape.prefix_splits = shape.split_settings["prefix_splits"]
        row_split_size = shape.split_settings["row_split_size"]
        if page_table.ndim != 2 or page_table.shape[0] != len(query_lengths):
            raise ValueError("Page table does not cover the mixed batch")
        if page_table.shape[1] < max(shape.kv_lengths):
            raise ValueError("Page-one table is shorter than the host KV lengths")
        if cache_lengths.numel() != len(query_lengths) or cumulative_query.numel() != len(query_lengths) + 1:
            raise ValueError("GPU metadata does not cover the mixed batch")
        if self.plan_copy_done is not None:
            self.plan_copy_done.synchronize()
        row_queries = query_lengths[shape.prefix_rows:]
        row_lengths = shape.kv_lengths[shape.prefix_rows:]
        if self.role_split:
            return self._plan_role_split(shape, row_queries, row_lengths, page_table, cache_lengths, cumulative_query)
        query_indptr = torch.tensor([0, *itertools.accumulate(row_queries)], dtype=torch.int32)
        kv_indptr = torch.tensor([0, *itertools.accumulate(row_lengths)], dtype=torch.int32)
        page_indices = torch.cat([
            page_table[index, :length]
            for index, length in enumerate(shape.kv_lengths)
            if index >= shape.prefix_rows
        ]).contiguous()
        last_page = torch.ones(len(row_lengths), dtype=torch.int32)
        self._plan_rows(
            (query_indptr, kv_indptr, page_indices, last_page, self.heads, self.kv_heads, 128, 1),
            dict(causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                 **({"fixed_split_size": row_split_size} if row_split_size is not None else {})),
        )
        self.plan_copy_done = torch.cuda.Event()
        self.plan_copy_done.record()
        packed_query = shape.maximum_row_query * (self.heads // self.kv_heads)
        shape.native_tile = 32 if self.wrapper._plan_info[3] == 64 and 16 < packed_query <= 32 else 0
        if self.fp32_partials:
            plan_values = self.wrapper._plan_info
            if len(plan_values) != 15:
                raise ValueError("Unsupported native FA2 planner ABI")
            if plan_values[14]:
                partial_rows = self.heads * plan_values[0] * plan_values[3]
                if plan_values[11] < plan_values[10] + partial_rows * 128 * 4:
                    raise ValueError("Planner does not reserve FP32 partial outputs")
                if self.workspace.numel() < plan_values[11] + partial_rows * 4:
                    raise ValueError("Insufficient FP32 partial workspace")
        shape.page_table = page_table
        shape.cache_lengths = cache_lengths
        shape.cumulative_query = cumulative_query
        shape.prefix_pages = page_table[:shape.prefix_rows, :max(shape.kv_lengths[:shape.prefix_rows])].contiguous()
        shape.prefix_lengths = cache_lengths[:shape.prefix_rows].contiguous()
        shape.prefix_indptr = cumulative_query[:shape.prefix_rows + 1].contiguous()
        shape.lse = torch.empty((self.heads, shape.prefix_tokens), dtype=torch.float32, device=self.device)
        shape.partial_lse = torch.empty(
            (shape.prefix_splits, self.heads, shape.prefix_tokens), dtype=torch.float32, device=self.device
        )
        shape.partial_output = torch.empty(
            (shape.prefix_splits, self.heads, shape.prefix_tokens, 128), dtype=torch.float32, device=self.device
        )
        shape.split_counts = torch.full((shape.prefix_rows,), shape.prefix_splits, dtype=torch.int32, device=self.device)
        shape.tile_counter = torch.zeros(1, dtype=torch.int32, device=self.device)
        shape.direct_row_output = self.direct_bf16_output and bool(self.wrapper._plan_info[14])
        shape.float_row_output = (
            torch.empty((sum(row_queries), self.heads, 128), dtype=torch.float32, device=self.device)
            if self.fp32_partials and not shape.direct_row_output else None
        )
        self.plans += 1
        if self.plans % 50 == 1:
            logger.info(
                "NATIVE_PDL_PLAN plans=%d fa3_rows=%d fa2_rows=%d fa3_tokens=%d row_q=%d row_max_kv=%d tile=%d sm_budget=%d",
                self.plans, shape.prefix_rows, len(row_lengths), shape.prefix_tokens,
                shape.maximum_row_query, max(row_lengths), shape.native_tile, self.sm_budget,
            )
            logger.info("NATIVE_PDL_SPLIT_POLICY %s", shape.split_settings)
        return NativePdlPlan(self, shape)


    def _plan_role_split(self, shape, row_queries, row_lengths, page_table, cache_lengths, cumulative_query):
        shape.native_tile = 0
        shape.page_table = page_table
        shape.cache_lengths = cache_lengths
        shape.cumulative_query = cumulative_query
        shape.prefix_pages = page_table[:shape.prefix_rows, :max(shape.kv_lengths[:shape.prefix_rows])].contiguous()
        shape.prefix_lengths = cache_lengths[:shape.prefix_rows].contiguous()
        shape.prefix_indptr = cumulative_query[:shape.prefix_rows + 1].contiguous()
        shape.row_pages = page_table[shape.prefix_rows:, :max(row_lengths)].contiguous()
        shape.row_lengths = cache_lengths[shape.prefix_rows:].contiguous()
        shape.row_indptr = (cumulative_query[shape.prefix_rows:] - cumulative_query[shape.prefix_rows]).to(self.torch.int32).contiguous()
        self.plans += 1
        if self.plans % 50 == 1:
            logger.info(
                "ROLE_SPLIT_FA3_PLAN plans=%d prefix_rows=%d row_count=%d prefix_tokens=%d row_q=%d row_max_kv=%d row_splits=%d",
                self.plans, shape.prefix_rows, len(row_lengths), shape.prefix_tokens,
                shape.maximum_row_query, max(row_lengths), self.role_row_splits,
            )
        return NativePdlPlan(self, shape)


class NativePdlPlan:
    def __init__(self, executor, shape):
        self.executor, self.shape = executor, shape

    def _run_role_split(self, query, key, value, scale, output):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        executor, shape = self.executor, self.shape
        torch = executor.torch
        key_cache, value_cache = key.unsqueeze(1), value.unsqueeze(1)
        current = torch.cuda.current_stream()
        side = executor.side_stream
        side.wait_stream(current)
        with torch.cuda.stream(side):
            flash_attn_with_kvcache(
                q=query[:shape.prefix_tokens], k_cache=key_cache, v_cache=value_cache,
                page_table=shape.prefix_pages, cache_seqlens=shape.prefix_lengths, cu_seqlens_q=shape.prefix_indptr,
                max_seqlen_q=shape.maximum_prefix_query, softmax_scale=scale, causal=True,
                out=output[:shape.prefix_tokens],
            )
        flash_attn_with_kvcache(
            q=query[shape.prefix_tokens:], k_cache=key_cache, v_cache=value_cache,
            page_table=shape.row_pages, cache_seqlens=shape.row_lengths, cu_seqlens_q=shape.row_indptr,
            max_seqlen_q=shape.maximum_row_query, softmax_scale=scale, causal=True,
            num_splits=executor.role_row_splits, out=output[shape.prefix_tokens:],
        )
        current.wait_stream(side)
        # The side stream read and wrote tensors owned by the current stream; keep them alive past its work.
        for t in (query, output, shape.prefix_pages, shape.prefix_lengths, shape.prefix_indptr):
            t.record_stream(side)

    def run(self, query, key, value, scale, output=None, pdl=True, timeline=None, query_tile=None):
        executor, shape = self.executor, self.shape
        torch = executor.torch
        if query.shape != (shape.total_tokens, executor.heads, 128) or query.dtype != torch.bfloat16:
            raise ValueError("PDL query differs from its planned BF16 head128 shape")
        if key.ndim == 4:
            if key.shape[1] != 1 or value.shape[1] != 1:
                raise ValueError("PDL requires page size one")
            key, value = key[:, 0], value[:, 0]
        if output is None:
            output = torch.empty_like(query)
        if executor.role_split:
            self._run_role_split(query, key, value, scale, output)
            executor.calls += 1
            if executor.calls % 100 == 1:
                logger.info(
                    "NATIVE_PDL_CALL calls=%d role_split=1 row_q=%d verify_tokens=%d row_splits=%d",
                    executor.calls, shape.maximum_row_query, getattr(shape, "mixed_verify_tokens", 0),
                    executor.role_row_splits,
                )
            return output
        timeline = executor.empty_timeline if timeline is None else timeline
        primary_arguments = (
            query[:shape.prefix_tokens], key, value, output[:shape.prefix_tokens], shape.lse,
            shape.prefix_indptr, shape.prefix_lengths, shape.prefix_pages,
            shape.partial_output, shape.partial_lse, shape.split_counts, shape.tile_counter,
            shape.prefix_splits, shape.maximum_prefix_query, executor.sm_budget, scale,
        )
        torch.ops.stock_fa3_pdl.run(*primary_arguments, 2 if pdl else 1, timeline)
        wrapper = executor.wrapper
        selected_tile = shape.native_tile if query_tile is None else query_tile
        row_output = shape.float_row_output if executor.fp32_partials and not shape.direct_row_output else output[shape.prefix_tokens:]
        row_namespace = "native_fa2_q32_fp32_direct" if shape.direct_row_output else executor.row_namespace
        getattr(torch.ops, row_namespace).run(
            query[shape.prefix_tokens:], key, value, wrapper._qo_indptr_buf,
            wrapper._paged_kv_indptr_buf, wrapper._paged_kv_indices_buf,
            wrapper._paged_kv_last_page_len_buf, row_output, executor.empty_lse,
            wrapper._float_workspace_buffer, wrapper._int_workspace_buffer,
            list(wrapper._plan_info), scale, pdl, pdl, timeline,
            selected_tile, shape.maximum_row_query,
        )
        if executor.fp32_partials and not shape.direct_row_output:
            output[shape.prefix_tokens:].copy_(row_output)
        if shape.prefix_splits > 1:
            torch.ops.stock_fa3_pdl.run(*primary_arguments, 3, timeline)
        executor.calls += 1
        if executor.verify:
            from sgl_kernel.flash_attn import flash_attn_with_kvcache

            snapshot = output.clone()
            expected = flash_attn_with_kvcache(
                q=query, k_cache=key.unsqueeze(1), v_cache=value.unsqueeze(1),
                page_table=shape.page_table, cache_seqlens=shape.cache_lengths,
                cu_seqlens_q=shape.cumulative_query, max_seqlen_q=max(shape.query_lengths),
                softmax_scale=scale, causal=True,
            )
            relative = (snapshot.float() - expected.float()).norm(dim=-1) / expected.float().norm(dim=-1).clamp_min(1e-6)
            worst = float(relative.max())
            if not torch.isfinite(snapshot).all() or not torch.isfinite(expected).all() or worst >= 0.02:
                raise RuntimeError(f"NATIVE_PDL_VERIFY failed worst_relative={worst}")
            if executor.calls % 50 == 1:
                logger.info("NATIVE_PDL_VERIFY calls=%d worst_relative=%.6g", executor.calls, worst)
        if executor.calls % 100 == 1:
            logger.info(
                "NATIVE_PDL_CALL calls=%d row_q=%d verify_tokens=%d row_tile=%d",
                executor.calls, shape.maximum_row_query, getattr(shape, "mixed_verify_tokens", 0), selected_tile,
            )
        return output


class UnifiedServingPlan:
    """A6 in serving: one unified plan per forward batch (shared by every layer), the same run() interface as
    NativePdlPlan. FA3 rows run as stock flash-attention at the plan's split count; FA2 rows run v3's BatchPrefill body
    with the plan's in-launch merge schedule (FA2 groups and FA3 split rows); no merge launch."""

    def __init__(self, plan):
        self.plan = plan
        self.shape = SimpleNamespace()
        self.calls = 0

    total_calls = 0

    def run(self, query, key, value, scale, output, pdl=True, query_tile=None, timeline=None):
        if output is None:  # the backend passes no preallocated output on some paths, as for NativePdlPlan
            import torch
            output = torch.empty_like(query)
        self.plan.run(query, key, value, scale, output)
        self.calls += 1
        UnifiedServingPlan.total_calls += 1
        if UnifiedServingPlan.total_calls % 100 == 1:
            st = self.plan.stats
            logger.info("NATIVE_PDL_CALL calls=%d a6_unified=1 fa3_rows=%d items16=%d items32=%d fa3_splits=%d "
                        "verify_tokens=%d", UnifiedServingPlan.total_calls, getattr(self.plan, "fa3_rows", -1),
                        st["items16"], st["items32"], st["max_splits_fa3"],
                        getattr(self.shape, "mixed_verify_tokens", 0))
        return output


def prepare_unified(backend, queries, prefixes, metadata):
    executor = getattr(backend, "_a6_unified_executor", None)
    if executor is None:
        import torch
        torch.ops.load_library(os.environ["UNIFIED_SO"])
        import unified_executor_20261024 as unified
        executor = unified.UnifiedExecutor(
            backend.num_attention_heads, backend.num_kv_heads, backend.device,
            fa3_min_packed=int(os.environ.get("UNIFIED_FA3_MIN", "33")))
        backend._a6_unified_executor = executor
        logger.info("A6_UNIFIED executor heads=%d kv_heads=%d fa3_min=%d stock_fa3=%s policy=%s",
                    backend.num_attention_heads, backend.num_kv_heads, executor.fa3_min_packed,
                    os.environ.get("UNIFIED_FA3_STOCK"), os.environ.get("UNIFIED_SPLIT_POLICY"))
    kv_lengths = [p + q for p, q in zip(prefixes, queries)]
    plan = executor.plan(list(queries), kv_lengths, metadata.page_table, metadata.cu_seqlens_q,
                         metadata.cache_seqlens_int32)
    backend._a6_unified_plans = getattr(backend, "_a6_unified_plans", 0) + 1
    if backend._a6_unified_plans <= 3:
        # The device metadata replaces the plan's host copies: check it against the host lengths (one sync each, first
        # three steps only).
        import torch
        host_q = torch.tensor([0, *queries], dtype=torch.int32).cumsum(0, dtype=torch.int32)
        if not (torch.equal(metadata.cu_seqlens_q.cpu(), host_q) and
                torch.equal(metadata.cache_seqlens_int32.cpu(), torch.tensor(kv_lengths, dtype=torch.int32))):
            raise RuntimeError("A6: device metadata does not match the host query / KV lengths")
    if backend._a6_unified_plans % 200 == 1:
        logger.info("A6_UNIFIED_PLAN plans=%d stats=%s", backend._a6_unified_plans, plan.stats)
    return UnifiedServingPlan(plan)


def prepare_native_pdl(backend, forward_batch, metadata):
    if backend.page_size != 1 or backend.use_mla or backend.topk > 1 or backend.fa_impl_ver != 3:
        return None
    queries = forward_batch.extend_seq_lens_cpu
    prefixes = forward_batch.extend_prefix_lens_cpu
    if queries is None or prefixes is None:
        raise RuntimeError("Native PDL mixed step requires exact host query/prefix lengths")
    heads = backend.num_attention_heads
    kv_heads = backend.num_kv_heads
    packed_query_limit = int(os.environ.get("NATIVE_PDL_PACKED_QUERY_LIMIT", "32"))
    if partition_lengths(queries, prefixes, heads, kv_heads, packed_query_limit) is None:
        if not getattr(backend, "_native_pdl_bypass_logged", False):
            backend._native_pdl_bypass_logged = True
            logger.warning("NATIVE_PDL_BYPASS queries=%s heads=%d kv_heads=%d", queries, heads, kv_heads)
        return None
    if os.environ.get("SGLANG_A6_UNIFIED") == "1":
        return prepare_unified(backend, queries, prefixes, metadata)
    fp32_partials = os.environ.get("SGLANG_NATIVE_PDL_FP32_PARTIALS") == "1"
    direct_bf16_output = os.environ.get("SGLANG_NATIVE_PDL_DIRECT_BF16_OUTPUT") == "1"
    executor_name = "_native_pdl_fp32_executor" if fp32_partials else "_native_pdl_executor"
    if direct_bf16_output:
        executor_name += "_direct"
    if packed_query_limit != 32:
        executor_name += f"_rows{packed_query_limit}"
    executor = getattr(backend, executor_name, None)
    if executor is None:
        budget = os.environ.get("SGLANG_NATIVE_PDL_FA3_SMS")
        policy = os.environ.get("SGLANG_NATIVE_PDL_SPLIT_POLICY", "explicit")
        if policy not in ("explicit", "geometry_v1", "geometry_v2", "geometry_v3"):
            raise ValueError(f"Unknown native split policy: {policy}")
        executor = NativePdlExecutor(
            heads, kv_heads, backend.device,
            sm_budget=int(budget) if budget else None,
            verify=os.environ.get("SGLANG_NATIVE_PDL_VERIFY") == "1",
            adaptive_splits=policy != "explicit",
            policy_version=policy if policy != "explicit" else "geometry_v1",
            workspace_bytes=(512 if policy != "explicit" else 128) << 20,
            max_workspace_bytes=int(os.environ.get("SGLANG_NATIVE_PDL_MAX_WORKSPACE_MIB", "4096" if policy != "explicit" else "128")) << 20,
            fp32_partials=fp32_partials,
            direct_bf16_output=direct_bf16_output,
            packed_query_limit=packed_query_limit,
        )
        setattr(backend, executor_name, executor)
    plan = executor.plan(queries, prefixes, metadata.page_table, metadata.cache_seqlens_int32, metadata.cu_seqlens_q)
    plan.shape.mixed_verify_tokens = getattr(forward_batch, "mixed_verify_tokens", 0)
    return plan
