# Full graph capture notes
Currently, SGLang only supports piecewise cuda graph capture. See [roadmap](https://github.com/sgl-project/sglang/issues/11490) and [initial PR](https://github.com/sgl-project/sglang/pull/10062).

This is because in SGLang's mixed chunk prefill, prefill and decode requests are packed into one batch and processed in one BatchPrefill kernel to avoid long prefill blocking decode requests. The #CTAs launched is unrelated to kv length (page table) but related to query length. While we could copy the query tokens of prefill and decode into one buffer, different p/d ratio can lead to different #CTAs launched (decode is 1 CTA/token while prefill is 1 CTA/128 token, where 128 is the tile size of the prefill kernel). Therefore, it [breaks the graph at the attention forward op](https://github.com/sgl-project/sglang/blob/9ec6031d7066cc25d976f289c554f7c4ce94adad/python/sglang/srt/compilation/backend.py#L395).

We can achieve full graph capture by replacing the BatchPrefill kernel with our custom persistent kernel (with overlap schedule), which launches 2 CTAs per SM. An alternative approach is to only capture a small number of prefill batch sizes (e.g. 512, 1024, 2048, 4096), so that we only need 4 * #decode_graphs. We will adopt the persistent full graph capture approach first. All other kernels also need to be launch CTAs based on token count/fixed #CTAs, though seems that the `apply_rope_pos_ids_cos_sin_cache` already achieves this.

## Development branches
We will develop on the following branches. As this is a large feature, **please make PRs instead of directly pushing the code**.
- SGLANG: [attn_fuse_](https://github.com/Edenzzzz/sglang-dev/tree/attn_fuse_)
   - `pip install -e "python"`
- FlashInfer: [persistent_overlap_](https://github.com/Edenzzzz/flashinfer/tree/persistent_overlap_)
   - To install, run `pip install -e . --no-build-isolation`. To recompile, just edit the code in the main repo and clear cache `python3 -m flashinfer clear-cache`. If you edit `.cu` files in csrc, it might be necessary to run `rm -rf ~/.cache/flashinfer`.
We need to capture all the operations in the forward pass of the [flashinfer backend](python/sglang/srt/layers/attention/flashinfer_backend.py), including kv cache saving operations.

## Implementation
Please refer to [init_forward_metadata_capture_cuda_graph](python/sglang/srt/layers/attention/flashinfer_backend.py) in flashinfer backend, [cuda graph runner](python/sglang/srt/model_executor/cuda_graph_runner.py) and [piecewise cuda graph runner](python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py).

## Benchmarking
You can reference the workloads in [my PR](https://github.com/sgl-project/sglang/pull/8368) and [the official benchmarking docs](https://docs.sglang.ai/developer_guide/benchmark_and_profiling.html) before making the PR. We plan to benchmark on one multi-turn chat dataset (ShareGPT, medium benefit for our method), one low-benefit dataset (LooGLE), and two high-benefit (decode-heavy/balanced) datasets: codebase analysis/debugging with reasoning (e.g. Cursor) and Agent/tool calling.  **Better use H100/H800 to benchmark as that's what we will use for the final experiments**
You may also use/modify the nsys profiler tools in [examples/profiler/nsys_profile_tools/README.md](examples/profiler/nsys_profile_tools/README.md), which decomposes kernel runtime into different categories and compare it against total CPU runtime. For example, it may be useful to know how much kernel launch gap exists in the attention phase.
You can use either one of the following workloads:
* Launch server:
```
python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B  --enable-mixed-chunk  --attention-backend flashinfer
```
* Benchmark:
```
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 8192 --random-output 400 --request-rate 8 --num-prompt 480
python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --request-rate 8 --sharegpt-context-len 131072 --num-prompts 480
```
