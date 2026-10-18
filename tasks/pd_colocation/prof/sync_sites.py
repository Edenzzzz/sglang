"""Map GPU idle gaps and device-to-host copies in a torch-profiler trace (CPU+GPU, with_stack) to Python call sites.

For each DtoH memcpy and each GPU gap > 50 us, take the runtime launch of the kernel/copy AFTER it (correlation id),
and the innermost python_function frames enclosing that launch on the CPU thread; aggregate gap time by call site.
  python sync_sites.py <trace.json.gz>"""
import bisect, collections, gzip, json, sys

t = json.load(gzip.open(sys.argv[1]))
evs = t["traceEvents"]
gpu = sorted([e for e in evs if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")],
             key=lambda e: e["ts"])
rt = {e["args"]["correlation"]: e for e in evs if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver")
      and "correlation" in e.get("args", {})}
py = [e for e in evs if e.get("ph") == "X" and e.get("cat") == "python_function"]
by_tid = collections.defaultdict(list)
for e in py:
    by_tid[e["tid"]].append(e)
for v in by_tid.values():
    v.sort(key=lambda e: e["ts"])
starts = {k: [e["ts"] for e in v] for k, v in by_tid.items()}


def site(launch):
    v = by_tid.get(launch["tid"], [])
    i = bisect.bisect_right(starts.get(launch["tid"], []), launch["ts"])
    frames = [e for e in v[max(0, i - 4000):i] if e["ts"] + e["dur"] >= launch["ts"]]
    frames.sort(key=lambda e: e["ts"])
    names = [f["name"] for f in frames if "sglang" in f["name"] or "flashinfer" in f["name"]]
    return " < ".join(n.split("python/")[-1][:90] for n in names[-3:][::-1]) or "?"


gap_by = collections.defaultdict(lambda: [0, 0.0])
sync_by = collections.defaultdict(lambda: [0, 0.0])
end = gpu[0]["ts"]
for e in gpu:
    g = e["ts"] - end
    launch = rt.get(e.get("args", {}).get("correlation"))
    if g > 50 and launch is not None:
        k = site(launch)
        gap_by[k][0] += 1; gap_by[k][1] += g
    if "DtoH" in e["name"] and launch is not None:
        k = site(launch)
        sync_by[k][0] += 1; sync_by[k][1] += e["dur"]
    end = max(end, e["ts"] + e["dur"])
print("GPU gaps > 50 us by the call site that launched the work AFTER the gap:")
for k, (n, d) in sorted(gap_by.items(), key=lambda x: -x[1][1])[:25]:
    print(f"  {d/1e3:7.1f}ms n={n:4d}  {k}")
print("DtoH copies by call site:")
for k, (n, d) in sorted(sync_by.items(), key=lambda x: -x[1][0])[:20]:
    print(f"  n={n:4d}  {k}")
