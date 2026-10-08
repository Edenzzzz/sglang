#!/usr/bin/env python3
"""
Test whether cudaLaunchCooperativeKernel blocks concurrent kernels on other streams.

Launches a cooperative kernel on stream A and a regular kernel on stream B.
Measures if they overlap or serialize.
"""
import torch
import torch.cuda
import time

# Simple long-running kernel via torch
def run_test():
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create two streams
    stream_a = torch.cuda.Stream(device)
    stream_b = torch.cuda.Stream(device)

    # Large tensors for long-running matmuls
    N = 4096
    a = torch.randn(N, N, device=device, dtype=torch.float32)
    b = torch.randn(N, N, device=device, dtype=torch.float32)
    c = torch.randn(N, N, device=device, dtype=torch.float32)
    d = torch.randn(N, N, device=device, dtype=torch.float32)

    torch.cuda.synchronize()

    # Baseline: two matmuls on same stream (sequential)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        torch.mm(a, b)
        torch.mm(c, d)
    end.record()
    torch.cuda.synchronize()
    sequential_ms = start.elapsed_time(end)
    print(f"Sequential (same stream): {sequential_ms:.2f} ms")

    # Test: two matmuls on different streams (should overlap)
    start_a = torch.cuda.Event(enable_timing=True)
    end_a = torch.cuda.Event(enable_timing=True)
    start_b = torch.cuda.Event(enable_timing=True)
    end_b = torch.cuda.Event(enable_timing=True)

    start_a.record(stream_a)
    with torch.cuda.stream(stream_a):
        for _ in range(10):
            torch.mm(a, b)
    end_a.record(stream_a)

    start_b.record(stream_b)
    with torch.cuda.stream(stream_b):
        for _ in range(10):
            torch.mm(c, d)
    end_b.record(stream_b)

    torch.cuda.synchronize()
    concurrent_a_ms = start_a.elapsed_time(end_a)
    concurrent_b_ms = start_b.elapsed_time(end_b)
    wall_ms = start_a.elapsed_time(end_b)
    print(f"Concurrent (two streams): A={concurrent_a_ms:.2f}ms, B={concurrent_b_ms:.2f}ms, wall={wall_ms:.2f}ms")
    print(f"  Overlap ratio: {(concurrent_a_ms + concurrent_b_ms) / wall_ms:.2f}x (>1 = overlap)")

    # Now test with flashinfer persistent (cooperative) kernel
    try:
        import flashinfer
        print(f"\nflashinfer: {flashinfer.__version__}")

        # Setup a persistent kernel workload
        num_kv_heads = 8
        num_qo_heads = 32
        head_dim = 128
        page_block_size = 1

        # Small mixed batch: 1 prefill + 50 decode
        p_qo_lens = [2048]
        p_kv_lens = [2048]
        d_qo_lens = [1] * 50
        d_kv_lens = [2048] * 50

        seq_lens = torch.tensor(d_kv_lens + p_kv_lens, dtype=torch.int32)
        q_lens = torch.tensor(d_qo_lens + p_qo_lens, dtype=torch.int32)
        d_bs = len(d_kv_lens)

        seq_lens_blocks = torch.ceil(seq_lens / page_block_size).int()
        q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
        kv_indptr = torch.cat([torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0).int()
        num_blocks = kv_indptr[-1].item()

        q = torch.rand(q_indptr[-1].item(), num_qo_heads, head_dim, device=device, dtype=torch.bfloat16)
        kv_data = torch.randn(num_blocks, 2, page_block_size, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)

        workspace_buffer = torch.empty(156 * 1024 * 1024, dtype=torch.uint8, device=device)

        # Persistent (cooperative launch) kernel
        wrapper_persistent = flashinfer.BatchAttention(kv_layout="NHD")
        wrapper_persistent.plan(
            q_indptr.to(device), kv_indptr.to(device),
            torch.arange(num_blocks, dtype=torch.int32, device=device),
            seq_lens.to(device),
            num_qo_heads, num_kv_heads, head_dim, head_dim,
            page_block_size, causal=True,
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
        )

        # Warmup
        for _ in range(5):
            wrapper_persistent.run(q, kv_data)
        torch.cuda.synchronize()

        # Test: persistent kernel on stream A, matmul on stream B
        e = torch.randn(2048, 2048, device=device, dtype=torch.float32)
        f = torch.randn(2048, 2048, device=device, dtype=torch.float32)

        start_a = torch.cuda.Event(enable_timing=True)
        end_a = torch.cuda.Event(enable_timing=True)
        start_b = torch.cuda.Event(enable_timing=True)
        end_b = torch.cuda.Event(enable_timing=True)

        # Launch persistent on stream A
        start_a.record(stream_a)
        with torch.cuda.stream(stream_a):
            for _ in range(20):
                wrapper_persistent.run(q, kv_data)
        end_a.record(stream_a)

        # Launch matmul on stream B concurrently
        start_b.record(stream_b)
        with torch.cuda.stream(stream_b):
            for _ in range(20):
                torch.mm(e, f)
        end_b.record(stream_b)

        torch.cuda.synchronize()
        coop_a_ms = start_a.elapsed_time(end_a)
        coop_b_ms = start_b.elapsed_time(end_b)
        wall_ms = start_a.elapsed_time(end_b)
        print(f"\nCooperative kernel + matmul (two streams):")
        print(f"  Persistent (stream A): {coop_a_ms:.2f}ms")
        print(f"  Matmul (stream B): {coop_b_ms:.2f}ms")
        print(f"  Wall time: {wall_ms:.2f}ms")
        print(f"  Sum: {coop_a_ms + coop_b_ms:.2f}ms")
        print(f"  Overlap ratio: {(coop_a_ms + coop_b_ms) / wall_ms:.2f}x (1.0 = serialized, >1 = overlap)")

        if (coop_a_ms + coop_b_ms) / wall_ms < 1.1:
            print("  >>> SERIALIZED: Cooperative kernel blocks other streams!")
        else:
            print("  >>> CONCURRENT: Cooperative kernel allows overlap")

    except Exception as ex:
        print(f"flashinfer test failed: {ex}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
