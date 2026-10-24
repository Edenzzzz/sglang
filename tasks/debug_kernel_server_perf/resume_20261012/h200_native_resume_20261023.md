# H200 restored for native-kernel serving work

The existing single-GPU allocation is now running: job `34c7bd503d461dc1`,
trial `304487505`, control plane `i18n-bd`, name **vibecuda**, requested queue
`h100-sxm-80gb.vmss-eastus2-vlmh200.agent.merge`. No replacement allocation
was submitted. The worker exposes exactly one NVIDIA H200, 132 SMs, UUID
`GPU-819511a1-a9bf-d8d7-e639-c2e62671fcbc`.

The frozen serving tree is restored at `/dev/shm/w/up/python`, based on
upstream `3fef129ff8a32c32da5f6a0efd33485f6cc6233c` plus the preserved overlay.
Torch is `2.14.1+cu130`, FlashInfer `0.7.0.post1`, and `sglang-kernel` `0.4.9`.
The native FA3 backend hash matches H20. All four target weight shards, the
target config, trained EAGLE3 weights, TraceLab data and ShareGPT data are
verified against the H20 inputs, rather than assumed identical from names.

Both native CUDA libraries are freshly compiled on H200. **All 13 GPU gates
pass**: five native-FA2 tests, three FA3 producer tests, and five combined
tests covering numerical correctness, SplitKV, graph replay, full-output
consumers, long contexts and adaptive replanning. Device timestamps show
overlap on the nontrivial cached-prefill fixture, not on every tiny prefill.

The initial combined gate hit a **test-fixture allocation error** in its
unused full-batch FA2 reference plan for the 24/2-head geometry: H200's
planner needed 136,839,168 bytes versus the fixture's 128-MiB buffer. No
candidate numerical comparison failed there. The fixture now generates
its Q/K/V and physical mapping directly, without planning an unrelated
full-batch FA2 execution. The actual native-row planner and all tolerances
are unchanged. The initial failure is preserved alongside the successful
retry; the original controller's failed status is not overwritten.

Native libraries:

- FA2: `/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so`
- FA3: `/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so`

Source/build tree: `/dev/shm/w/native_kernel_sources_20261023`.
Successful combined retry: `/dev/shm/w/combined_h200_retry_20261023`.
Workload drivers/data are restored under `/dev/shm/w`. The copied `perf`
scripts come from the earlier workload bundle and must be refreshed before
using the revised feature/predictability analysis.

Local evidence: `../../pd_colocation/results_20261023/h200_native_gates/`,
`h200_native_gates.tgz`, and `h200_model_fingerprints.txt` in that result
directory. These are kernel/restore gates, **not H200 serving-performance
results**. Same-input serving-output disagreements remain open on H20;
the new selector has not been promoted.
