#!/usr/bin/env python3
"""B1 ceiling: where a serving step's GPU time goes, from a torch-profiler trace.

Buckets every GPU kernel by name into attention / GEMM / sampling+verify / index+copy /
elementwise+norm / communication / other, and reports each bucket's share of kernel time and the
GPU-idle share of the profiled span. Fusing the small-op tail (B1) can save at most the
sampling+verify, index+copy and elementwise buckets' launch-bound part, so their share is the
ceiling to compare against the 30%/10% bar before building anything. Also reports launch counts per
bucket and the time-weighted estimated achieved occupancy kineto attaches to each kernel.

Usage: parse_kernel_tail.py <trace.json[.gz]> [more traces...]
"""
import gzip, json, re, sys
from collections import defaultdict

BUCKETS = [
    ("attention", r"flash|fmha|attention|attn|persistent_fa3|FA2Executor|BatchPrefill|BatchDecode|"
                  r"merge_state|mla|paged"),
    ("gemm", r"gemm|cutlass|cublas|sm90_xmma|nvjet|matmul|wgmma|_mm_|splitk"),
    ("sampling_verify", r"argmax|topk|top_k|sample|softmax|verify|accept|tree|eagle|draft|"
                        r"multinomial|sort|cumsum|log_softmax"),
    ("index_copy", r"index|gather|scatter|copy|memcpy|memset|cat|slice|fill|arange|"
                   r"embedding|unrolled_elementwise_kernel.*copy"),
    ("elementwise_norm", r"norm|rope|rotary|silu|gelu|act_and_mul|elementwise|add|mul|"
                         r"reduce|vectorized"),
    ("communication", r"nccl|allreduce|all_reduce|custom_ar"),
]


def bucket(name):
    low = name.lower()
    for label, pattern in BUCKETS:
        if re.search(pattern, low, re.I):
            return label
    return "other"


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def main():
    per_bucket = defaultdict(float)
    launches = defaultdict(int)
    occ_weighted = defaultdict(float)
    occ_time = defaultdict(float)
    per_kernel = defaultdict(lambda: [0.0, 0])
    span_lo, span_hi, busy = None, None, 0.0
    intervals = []
    for path in sys.argv[1:]:
        for ev in load(path).get("traceEvents", []):
            if ev.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset") or "dur" not in ev:
                continue
            name, ts, dur = ev.get("name", "?"), float(ev["ts"]), float(ev["dur"])
            b = bucket(name)
            per_bucket[b] += dur
            launches[b] += 1
            occ = (ev.get("args") or {}).get("est. achieved occupancy %")
            if occ is not None:
                occ_weighted[b] += float(occ) * dur
                occ_time[b] += dur
            k = per_kernel[name[:90]]
            k[0] += dur
            k[1] += 1
            intervals.append((ts, ts + dur))
    if not intervals:
        sys.exit("no GPU kernel events in the trace (was the GPU activity recorded?)")
    intervals.sort()
    cur_lo, cur_hi = intervals[0]
    for lo, hi in intervals[1:]:
        if lo > cur_hi:
            busy += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    busy += cur_hi - cur_lo
    span_lo, span_hi = intervals[0][0], max(hi for _, hi in intervals)
    total = sum(per_bucket.values())
    span = span_hi - span_lo
    print(f"profiled span {span / 1e3:.1f} ms, GPU busy {busy / 1e3:.1f} ms "
          f"({100 * busy / span:.1f}%), kernel time {total / 1e3:.1f} ms")
    for label, us in sorted(per_bucket.items(), key=lambda kv: -kv[1]):
        occ = (f"{occ_weighted[label] / occ_time[label]:5.1f}%" if occ_time[label] else "  n/a")
        print(f"BUCKET {label:18s} {us / 1e3:9.2f} ms  {100 * us / total:5.1f}% of kernel time  "
              f"launches={launches[label]:7d}  est_occupancy={occ}")
    print("top kernels:")
    for name, (us, n) in sorted(per_kernel.items(), key=lambda kv: -kv[1][0])[:25]:
        print(f"  {us / 1e3:8.2f} ms  n={n:6d}  avg={us / max(n, 1):7.1f} us  [{bucket(name)}] {name}")


if __name__ == "__main__":
    main()
