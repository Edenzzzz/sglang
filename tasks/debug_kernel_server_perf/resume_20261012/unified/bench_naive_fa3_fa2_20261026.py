"""The naive two-kernel baseline for the kernel claim, on the same trace-sampled (and synthetic) captures and with the
same graph-replay timing and FP32 gate as bench_unified_trace_20261024.py:

  naive_fa3_fa2_serial   stock flash-attention FA3 on the prefill rows (sglang's call, default splits), then STOCK
                         flashinfer FA2 (BatchPrefillWithPagedKVCacheWrapper, backend="fa2", flashinfer's own plan,
                         tile choice, split-KV and merge) on the decode / verify rows, one stream
  naive_fa3_fa2_streams  the same two calls on two streams (the obvious way to overlap them without PDL)

Run in its own process with PYTHONPATH WITHOUT the port07 tree, so `import flashinfer` is the pip release
(0.7.0.post1), not our modified BatchPrefill. Records append to the same kernel_results.json, so
analyze_trace_sampled_20261023.py compares any arm (A6, v3, sglang FA3) against these.
  python bench_naive_fa3_fa2_20261026.py <captures dir> <kernel_results.json> [--iterations N]
Claude, 2026-10-26.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/dev/shm/w/ub")
import bench_captured_mixed_20261023 as B  # noqa: E402
import flashinfer  # noqa: E402
from sgl_kernel.flash_attn import flash_attn_with_kvcache  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("captures", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--iterations", type=int, default=20)
args = parser.parse_args()
if "port07" in flashinfer.__file__ or "/dev/shm/w/fi" in flashinfer.__file__:
    raise SystemExit(f"not the stock flashinfer release: {flashinfer.__file__}")
torch.backends.cuda.matmul.allow_tf32 = False
report = json.loads(args.output.read_text()) if args.output.exists() else dict(records=[], captures=[])
report.setdefault("naive_flashinfer", dict(version=flashinfer.__version__, path=flashinfer.__file__))
workspace = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
side = torch.cuda.Stream()
for path in sorted(args.captures.glob("*.pt")):
    case = B.load_case(path)
    expected = B.fp32_reference(case)
    g = case.geometry
    heads, kv_heads = g["heads"], g["kv_heads"]
    pr, pt = case.prefix_rows, case.prefix_tokens
    lens = case.cache_lengths[pr:]
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), lens.cumsum(0, dtype=torch.int32)])
    kv_indices = torch.cat([case.page_table[pr + i, :int(lens[i])] for i in range(lens.numel())]).to(torch.int32)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    wrapper.plan(case.row_cumulative_query.to(torch.int32), kv_indptr, kv_indices,
                 torch.ones(lens.numel(), dtype=torch.int32, device="cuda"), heads, kv_heads, g["key_shape"][-1], 1,
                 causal=True, sm_scale=g["scale"], q_data_type=case.query.dtype)
    k4, v4 = case.key, case.value  # [pages, 1, kv_heads, d] == NHD with page size 1

    def fa3_prefill():
        flash_attn_with_kvcache(
            q=case.query[:pt], k_cache=case.key, v_cache=case.value, page_table=case.prefix_pages,
            cache_seqlens=case.cache_lengths[:pr], cu_seqlens_q=case.cumulative_query[:pr + 1],
            max_seqlen_q=max(g["query_lengths"][:pr]), out=case.output[:pt], softmax_scale=g["scale"], causal=True)

    def fa2_rows():
        wrapper.run(case.query[pt:], (k4, v4), out=case.output[pt:])

    def serial():
        fa3_prefill()
        fa2_rows()

    def streams():
        cur = torch.cuda.current_stream()
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            fa3_prefill()
        fa2_rows()
        cur.wait_stream(side)

    for arm, fn in (("naive_fa3_fa2_serial", serial), ("naive_fa3_fa2_streams", streams)):
        for order in ("forward", "reverse"):
            r = B.measure(case, fn, expected, "graph", args.iterations)
            r.update(capture=path.name, mode=g["mode"], arm=arm, style="graph", order=order, bound=False)
            report["records"].append(r)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    ms = {a: [r["milliseconds"] for r in report["records"] if r["capture"] == path.name and r["arm"] == a]
          for a in ("naive_fa3_fa2_serial", "naive_fa3_fa2_streams")}
    print(f"NAIVE_BENCH {path.name} serial={sum(ms['naive_fa3_fa2_serial']) / 2:.4f} "
          f"streams={sum(ms['naive_fa3_fa2_streams']) / 2:.4f} rows={lens.numel()} fa3_rows={pr}", flush=True)
    del case, expected, wrapper
    torch.cuda.empty_cache()
print("NAIVE_BENCH_DONE", flush=True)
