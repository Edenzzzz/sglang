"""A5.0: FA3-only timing of our dedicated persistent FA3 primary vs flashinfer's FA3 BatchPrefill ("stock") and
upstream sgl-kernel FA3 (flash_attn_with_kvcache, the kernel serving's split runs), on one fixture, all outputs
checked against the fp32 causal reference. Copy of tma_multicast_reference_20261013.py plus the upstream arm.
"""
import itertools
import json
import math
import os

import torch


PAGE_SIZE = 128
QUERY_HEADS = 28
KV_HEADS = 4
HEAD_DIM = 128


def parse_requests(value):
    requests = [tuple(map(int, item.split(":"))) for item in value.split(",") if item]
    if any(len(item) != 2 or item[0] <= 0 or item[1] < item[0] for item in requests):
        raise ValueError("Requests must have positive query lengths no larger than KV lengths")
    return requests


def build_fixture(primary, native, device="cuda", seed=31):
    requests = primary + native
    if not requests:
        raise ValueError("At least one request is required")
    query_lengths = [item[0] for item in requests]
    kv_lengths = [item[1] for item in requests]
    page_counts = [(length + PAGE_SIZE - 1) // PAGE_SIZE for length in kv_lengths]
    query_indptr = torch.tensor([0, *itertools.accumulate(query_lengths)], dtype=torch.int32)
    page_indptr = torch.tensor([0, *itertools.accumulate(page_counts)], dtype=torch.int32)
    torch.manual_seed(seed)
    query = torch.randn(sum(query_lengths), QUERY_HEADS, HEAD_DIM, device=device,
                        dtype=torch.bfloat16) * 0.1
    cache_shape = (sum(page_counts), PAGE_SIZE, KV_HEADS, HEAD_DIM)
    key = torch.randn(cache_shape, device=device, dtype=torch.bfloat16) * 0.1
    value = torch.randn(cache_shape, device=device, dtype=torch.bfloat16) * 0.1
    generator = torch.Generator().manual_seed(seed + 1)
    indices_cpu = torch.randperm(sum(page_counts), generator=generator, dtype=torch.int64)
    for request, length in enumerate(kv_lengths):
        valid = (length - 1) % PAGE_SIZE + 1
        if valid < PAGE_SIZE:
            final_page = int(indices_cpu[int(page_indptr[request + 1]) - 1])
            key[final_page, valid:].fill_(float("nan"))
            value[final_page, valid:].fill_(float("nan"))
    return {
        "query": query, "key": key, "value": value,
        "query_lengths": query_lengths, "kv_lengths": kv_lengths,
        "query_indptr": query_indptr, "page_indptr": page_indptr,
        "indices_cpu": indices_cpu, "indices": indices_cpu.to(device=device, dtype=torch.int32),
        "primary_requests": len(primary),
    }


def reference_output(fixture):
    query = fixture["query"]
    group = QUERY_HEADS // KV_HEADS
    reference = torch.empty_like(query, dtype=torch.float32)
    for request, (query_length, kv_length) in enumerate(
            zip(fixture["query_lengths"], fixture["kv_lengths"])):
        query_start = int(fixture["query_indptr"][request])
        page_start = int(fixture["page_indptr"][request])
        page_end = int(fixture["page_indptr"][request + 1])
        pages = fixture["indices"][page_start:page_end].long()
        key = fixture["key"][pages].reshape(-1, KV_HEADS, HEAD_DIM)[:kv_length].float()
        value = fixture["value"][pages].reshape(-1, KV_HEADS, HEAD_DIM)[:kv_length].float()
        key = key.permute(1, 2, 0)
        value = value.permute(1, 0, 2)
        positions = torch.arange(kv_length, device=query.device)
        for begin in range(0, query_length, 32):
            end = min(begin + 32, query_length)
            chunk = query[query_start + begin:query_start + end].float()
            chunk = chunk.reshape(end - begin, KV_HEADS, group, HEAD_DIM)
            chunk = chunk.permute(1, 0, 2, 3).reshape(KV_HEADS, -1, HEAD_DIM)
            scores = torch.matmul(chunk, key) / math.sqrt(HEAD_DIM)
            query_positions = torch.arange(begin, end, device=query.device).repeat_interleave(group)
            masked = positions[None, :] > (kv_length - query_length + query_positions[:, None])
            scores.masked_fill_(masked, float("-inf"))
            output = torch.matmul(torch.softmax(scores, dim=-1), value)
            output = output.reshape(KV_HEADS, end - begin, group, HEAD_DIM)
            reference[query_start + begin:query_start + end] = output.permute(1, 0, 2, 3).reshape(
                end - begin, QUERY_HEADS, HEAD_DIM)
    return reference


def verify(output, reference, label):
    assert torch.isfinite(output).all(), f"{label}: nonfinite output"
    error = torch.linalg.vector_norm(output.float() - reference, dim=-1)
    scale = torch.linalg.vector_norm(reference, dim=-1).clamp_min(1e-8)
    relative = error / scale
    bad = int((relative > 0.02).sum())
    maximum = float(relative.max())
    print(f"CHECK {label} bad_heads={bad}/{relative.numel()} worst={maximum:.6f}", flush=True)
    assert bad == 0, (label, bad, maximum)


def main():
    import flashinfer
    from sglang.srt.layers.attention.fused_fa3_loader import load_fused_fa3

    torch.backends.cuda.matmul.allow_tf32 = False
    primary = parse_requests(os.environ.get("FA3_REQUESTS", "512:4096"))
    native = parse_requests(os.environ.get("FA2_REQUESTS", "4:8192,4:4097"))
    fixture = build_fixture(primary, native, seed=int(os.environ.get("SEED", "31")))
    namespace = load_fused_fa3(verbose=True, cta_q=128)
    operations = getattr(torch.ops, namespace)
    query, key, value = (fixture[name] for name in ("query", "key", "value"))
    output = torch.empty_like(query)
    empty_lse = torch.empty(0, dtype=torch.float32, device="cuda")
    empty_timeline = torch.empty(0, dtype=torch.int64, device="cuda")
    primary_count = len(primary)
    primary_tokens = sum(length for length, _ in primary)
    primary_pages = int(fixture["page_indptr"][primary_count])
    sm_split = int(os.environ.get("SM_SPLIT", "112"))
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    primary_arguments = finish_arguments = native_arguments = None
    wrappers = []
    for begin, end, backend in ((0, primary_count, "fa3"), (primary_count, len(primary + native), "fa2")):
        if begin == end:
            continue
        token_begin = int(fixture["query_indptr"][begin])
        token_end = int(fixture["query_indptr"][end])
        page_begin = int(fixture["page_indptr"][begin])
        page_end = int(fixture["page_indptr"][end])
        query_indptr = (fixture["query_indptr"][begin:end + 1] - token_begin).to("cuda")
        page_indptr = (fixture["page_indptr"][begin:end + 1] - page_begin).to("cuda")
        last_page = torch.tensor([(length - 1) % PAGE_SIZE + 1 for length in fixture["kv_lengths"][begin:end]],
                                 dtype=torch.int32, device="cuda")
        indices = fixture["indices"][page_begin:page_end]
        workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
        graph_buffers = {}
        if backend == "fa2":
            graph_buffers = dict(use_cuda_graph=True, qo_indptr_buf=query_indptr.clone(),
                                 paged_kv_indptr_buf=page_indptr.clone(),
                                 paged_kv_indices_buf=indices.clone(),
                                 paged_kv_last_page_len_buf=last_page.clone())
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", backend=backend, **graph_buffers)
        wrapper.plan(query_indptr, page_indptr, indices, last_page,
                     num_qo_heads=QUERY_HEADS, num_kv_heads=KV_HEADS, head_dim_qk=HEAD_DIM,
                     page_size=PAGE_SIZE, causal=True, q_data_type=torch.bfloat16,
                     kv_data_type=torch.bfloat16)
        query_view = query[token_begin:token_end]
        output_view = output[token_begin:token_end]
        wrappers.append((wrapper, query_view, output_view))
        if backend == "fa2":
            native_arguments = (
                query_view, key, value, wrapper._qo_indptr_buf, wrapper._paged_kv_indptr_buf,
                wrapper._paged_kv_indices_buf, wrapper._paged_kv_last_page_len_buf,
                output_view, empty_lse, wrapper._float_workspace_buffer,
                wrapper._int_workspace_buffer, list(wrapper._plan_info), 1.0 / math.sqrt(HEAD_DIM),
            )
    if primary_count:
        integer_workspace = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        pinned_workspace = torch.empty_like(integer_workspace, device="cpu", pin_memory=True)
        float_workspace = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
        query_indptr = fixture["query_indptr"][:primary_count + 1].clone()
        page_indptr = fixture["page_indptr"][:primary_count + 1].clone()
        lengths = torch.tensor(fixture["kv_lengths"][:primary_count], dtype=torch.int32)
        plan = operations.build_fused_plan(
            integer_workspace, pinned_workspace, float_workspace, query_indptr, page_indptr,
            lengths, torch.ones(primary_count, dtype=torch.bool), QUERY_HEADS, KV_HEADS,
            HEAD_DIM, HEAD_DIM, PAGE_SIZE, sm_split, num_sms, True)
        float_workspace.fill_(255)
        primary_arguments = (
            query[:primary_tokens], key, value, fixture["indices"][:primary_pages], output[:primary_tokens],
            integer_workspace, float_workspace, plan, QUERY_HEADS, KV_HEADS, PAGE_SIZE,
            1.0 / math.sqrt(HEAD_DIM), sm_split, primary_count,
            max((length + PAGE_SIZE - 1) // PAGE_SIZE for _, length in primary),
            max(length for length, _ in primary), lengths.to("cuda"), query_indptr.to("cuda"),
            page_indptr.to("cuda"),
        )
        finish_arguments = (output[:primary_tokens], integer_workspace, float_workspace,
                            plan, primary_count, QUERY_HEADS, KV_HEADS)
        print(f"TMA_PRIMARY_PLAN splits={plan[54]} merge_rows={plan[-2]}", flush=True)

    def run(timeline=empty_timeline):
        if primary_arguments is not None:
            operations.fused_primary_compact(*primary_arguments, timeline)
        if native_arguments is not None:
            operations.native_fa2_run(*native_arguments, True, True, timeline)
        if finish_arguments is not None:
            operations.fused_primary_finish(*finish_arguments)

    def run_stock():
        for wrapper, query_view, output_view in wrappers:
            wrapper.run(query_view, (key, value), out=output_view)

    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    assert not native, "A5 fixtures are FA3-only"
    max_pages = max(int(fixture["page_indptr"][r + 1] - fixture["page_indptr"][r]) for r in range(len(primary)))
    page_table = torch.zeros(len(primary), max_pages, dtype=torch.int32, device="cuda")
    for r in range(len(primary)):
        b, e = int(fixture["page_indptr"][r]), int(fixture["page_indptr"][r + 1])
        page_table[r, :e - b] = fixture["indices"][b:e]
    cache_seqlens = torch.tensor(fixture["kv_lengths"][:len(primary)], dtype=torch.int32, device="cuda")
    cu_seqlens_q = fixture["query_indptr"][:len(primary) + 1].to("cuda")
    max_seqlen_q = max(length for length, _ in primary)
    upstream = {}

    fa3only = None
    if os.environ.get("FA3_UPSTREAM") == "1":
        # Our build of flash-attention's kernel on upstream's own persistent scheduler (fa3_only_run): separates our
        # scheduler adapter from our compile configuration.
        n = len(primary)
        iopt = dict(dtype=torch.int32, device="cuda")
        fo = dict(
            lse=torch.empty(int(query.shape[0]) * QUERY_HEADS, dtype=torch.float32, device="cuda"),
            page_table=torch.full((n * max_pages,), -1, **iopt),
            partial_o=torch.empty(1, dtype=torch.float32, device="cuda"),
            partial_lse=torch.empty(1, dtype=torch.float32, device="cuda"),
            nsd=torch.ones(n, **iopt), sem=torch.zeros(1, **iopt),
            batch_list=torch.arange(n, **iopt), page_indptr=fixture["page_indptr"][:n + 1].to("cuda"))
        fa3only = fo

    def run_fa3only():
        fo = fa3only
        operations.fa3_only_run(
            query, key, value, output, fo["lse"], fixture["indices"], fo["page_indptr"], cu_seqlens_q, cache_seqlens,
            fo["batch_list"], QUERY_HEADS, KV_HEADS, PAGE_SIZE, 1.0 / math.sqrt(HEAD_DIM), max_seqlen_q, max_pages, 1,
            fo["page_table"], fo["partial_o"], fo["partial_lse"], fo["nsd"], fo["sem"], True)

    def run_upstream():
        # Kept, not copied: under graph capture the returned tensor is the graph's static output.
        upstream["out"] = flash_attn_with_kvcache(
            query, key, value, page_table=page_table, cache_seqlens=cache_seqlens, cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q, softmax_scale=1.0 / math.sqrt(HEAD_DIM), causal=True)

    reference = reference_output(fixture)
    torch.cuda.synchronize()
    output.fill_(float("nan"))
    run()
    torch.cuda.synchronize()
    verify(output, reference, "candidate_eager")
    output.fill_(float("nan"))
    run_stock()
    torch.cuda.synchronize()
    verify(output, reference, "stock_eager")
    run_upstream()
    torch.cuda.synchronize()
    verify(upstream["out"], reference, "upstream_eager")
    if fa3only is not None:
        output.fill_(float("nan"))
        run_fa3only()
        torch.cuda.synchronize()
        verify(output, reference, "fa3only_eager")
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    original_query = query.clone()
    original_reference = reference
    for repeat in range(3):
        if repeat == 1:
            query.mul_(-7)
        elif repeat == 2:
            query.add_(0.25)
        if repeat:
            reference = reference_output(fixture)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        verify(output, reference, f"candidate_graph_{repeat}")
    query.copy_(original_query)
    reference = original_reference

    if os.environ.get("NATIVE_PDL_REQUIRE_OVERLAP") == "1":
        assert primary_arguments is not None and native_arguments is not None
        initial = torch.tensor([torch.iinfo(torch.int64).max, 0, torch.iinfo(torch.int64).max,
                                torch.iinfo(torch.int64).max], dtype=torch.int64, device="cuda")
        timeline = initial.clone()
        for graph_mode in (False, True):
            timeline.copy_(initial)
            if graph_mode:
                trace_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(trace_graph):
                    run(timeline)
                trace_graph.replay()
            else:
                run(timeline)
            torch.cuda.synchronize()
            stamps = timeline.tolist()
            assert 0 < stamps[0] < stamps[3] < stamps[1]
            assert 0 < stamps[2] <= stamps[3] < torch.iinfo(torch.int64).max
            verify(output, reference, f"overlap_graph_{graph_mode}")
            print(f"TMA_PDL_OVERLAP graph={graph_mode} stamps={stamps}", flush=True)

    if os.environ.get("TIME_AB", "1") == "1":
        assert os.environ.get("FUSED_FA3_TMA_TRACE", "0") == "0"
        iterations = int(os.environ.get("TIME_ITERS", "100"))
        assert iterations > 0
        arms = [("candidate", run), ("stock", run_stock), ("upstream", run_upstream)]
        if fa3only is not None:
            arms.append(("fa3only", run_fa3only))
        for label, function in arms:
            for _ in range(3):
                function()
            torch.cuda.synchronize()
            timing_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(timing_graph):
                function()
            for _ in range(10):
                timing_graph.replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                timing_graph.replay()
            end.record()
            end.synchronize()
            verify(upstream["out"] if label == "upstream" else output, reference, f"timed_{label}")
            print(f"TIME {label} graph=True ms={start.elapsed_time(end) / iterations:.6f}", flush=True)
    print("FIXTURE " + json.dumps({"primary": primary, "native": native, "page_size": PAGE_SIZE,
                                    "poisoned_tails": True, "shuffled_pages": True}), flush=True)
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
