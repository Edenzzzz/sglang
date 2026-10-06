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

## Warm the JIT cache before any arm

`test_fused_server_batch.py` globs `~/.cache/flashinfer/*/generated/batch_attention_with*`
and indexes `gen[0]` at `load()` time (line ~191). A fresh pod has no such cache, so every
arm dies with `IndexError: list index out of range` before a kernel runs -- an error that
names neither flashinfer nor the cache. Run `warm_jit_cache.py` once first; it drives one
tiny `BatchAttention` plan+run to generate those sources and asserts the glob is non-empty.

Run it with plain `python`, NOT `python -I`: isolated mode ignores `PYTHONPATH`, so the
flashinfer package on `PYTHONPATH` becomes unimportable and it fails with
`ModuleNotFoundError: No module named 'flashinfer'`.

## Capture full logs, not grep lines

`ab_fa3_traits.sh` writes each arm's complete output to `res/<arm>.full` and derives the
summary from it. An earlier version piped through
`grep -iE '...|Traceback|...'`, which matched the word `Traceback` but dropped the frames
after it -- so a real fault showed as a bare exception with no location and needed a manual
unfiltered re-run to diagnose. On a fault the driver now prints the tail of the full log.

## After pushing the flashinfer tree, RELINK `flashinfer/data/`

`flashinfer/data/{csrc,cutlass,include,spdlog,tvm_binding}` are **absolute** symlinks into
the dev box's `/opt/tiger/flashinfer`. `tar` preserves them as symlinks, so on the pod they
dangle. The failure is not a missing-file error but:

    FileExistsError: [Errno 17] File exists: '/dev/shm/w/fi/flashinfer/data/csrc'

because `jit/core.py:17-18` calls `os.makedirs(..., exist_ok=True)`, and `exist_ok` only
suppresses the error when the path is a *directory* -- a dangling symlink exists but is not
one. Every arm fails instantly and identically, which looks like a harness bug rather than a
restore defect. Fix after each flashinfer push:

    cd /dev/shm/w/fi/flashinfer/data
    rm -f csrc cutlass include spdlog tvm_binding
    ln -s /dev/shm/w/fi/csrc csrc
    ln -s /dev/shm/w/fi/3rdparty/cutlass cutlass
    ln -s /dev/shm/w/fi/include include
    ln -s /dev/shm/w/fi/3rdparty/spdlog spdlog
    ln -s /dev/shm/w/fi/tvm_binding tvm_binding
    for l in csrc cutlass include; do readlink -e $l >/dev/null || echo "DANGLING $l"; done

## Restore order: the kernel A/B does not need the sglang tree

`tests/test_fused_server_batch.py` imports only `os`, `sys`, `glob`, `numpy` and `torch` --
**not sglang**. So the kernel-level arms can run as soon as the flashinfer tree, cutlass,
flash-attention and the venv are in place. The sglang `python/` tree (5.8 MB, the slowest
push at ~13 s per 60 KB chunk, roughly 30 min) is needed only for the server-integration
items. Push it *after* launching the kernel A/B, or in parallel -- not before, which idles
the GPU for half an hour behind a dependency the harness does not have.

Push order that respects this: venv -> cutlass/flash-attention clones (pod-side, fast) ->
flashinfer fork tree -> **launch kernel A/B** -> sglang tree.

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
