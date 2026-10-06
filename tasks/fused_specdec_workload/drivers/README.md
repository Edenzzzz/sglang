# Pod drivers

These ran on the H200 pod under `/dev/shm/w` and were lost when the trial expired
on 2026-10-05; they live here now so that costs nothing next time.

| script | what it does |
|---|---|
| `uab.sh` | unified queue vs baseline serving A/B, one extension dir per arm so the `-D` flags take effect |
| `lcab.sh` | the same A/B at 65536 input / 256 output, 16 prompts, concurrency 8 — the long-context regime where attention dominates the step (56 GiB of KV, fits at `--mem-fraction-static 0.80`) |
| `dmp.sh` | runs the unified arm under `SGLANG_FUSED_DUMP` to capture a faulting batch as an `.npz` |
| `trsw5.sh` | four-arm driver with the GPU-exclusivity guard; refuses to start if `nvidia-smi --query-compute-apps` is non-empty |
| `trsw9.sh` | asymmetric split sweep (`sm_fa3` 64/96/116) against a fused-off baseline |
| `noproxy.sh` | strips proxy variables; `merlin-cli` fails with EOF through the proxy |
| `px.py` | chunked base64 push + `execute-script` exec helper (set `SID` to the trial). Strips proxy vars itself, and `probe_trial()` reports trial status |

## `{"error": "EOF"}` has TWO causes, and they look identical

1. **Proxy.** `NO_PROXY` (uppercase) omits `.bytedance.net` / `.byteintl.net` while lowercase
   `no_proxy` lists them, and Go prefers the uppercase one, so `merlin-cli` proxies these
   hosts and the proxy fails TLS. Any call made without stripping the proxy vars can fail
   this way even when everything is healthy. `px.py` and `noproxy.sh` strip them.
2. **The pod is gone.** `execute-script` against an expired trial returns the same bare
   `EOF`, with nothing saying the pod no longer exists.

So on any `EOF`, run `px.probe_trial()` (or `job-v2 runs get --json '{"sid":"..."}'` with the
proxy stripped) **before** concluding anything about the network. Three trials have now
expired mid-measurement, so also stream results off the pod as they land rather than
collecting at the end -- `/dev/shm` dies with the trial.

Every one of them assumes the working set described under "Re-establishing the
GPU working set" in `../../debug_kernel_server_perf_4_mode.md`: a torch
2.8/cu128 venv at `/dev/shm/w/venv`, flashinfer at `/dev/shm/w/fi`, the sglang
`python/` tree at `/dev/shm/w/python`, and weights at `/dev/shm/w/qwen7b`.

The arms they define are only valid with the three pre-flight checks in memory:
the fused composition counter is non-empty, the numerics pass in that exact mode
and split, and the compile flags are present in the generated `build.ninja`.
