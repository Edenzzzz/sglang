"""In-server prefill throughput from STEPTRACE lines (SGLANG_STEP_TRACE=1).

For steps carrying exactly one prefill row (np=1, ptok >= 1024), the prefill FLOPs are
causal attention 4*D*H*L*(ptok*pf + ptok^2/2) plus linear 2*P*ptok (Qwen2.5-7B: 28 layers,
28 query heads, head_dim 128, 7.6e9 params), divided by the step's GPU time (gpu=, ms). This
counts the co-scheduled decode/verify rows' time against prefill, so it is a lower bound on the
prefill kernel rate and the number that bounds TTFT. Reported per prefix bucket.
Usage: steptrace_prefill_rate.py <server_log> [...]"""
import re
import sys
from collections import defaultdict

L, H, D, P = 28, 28, 128, 7.6e9
PAT = re.compile(r"STEPTRACE (\w+) .*?np=(\d+) ptok=(\d+) pf=(\d+) nd=(\d+) dkv=(\d+).*?gpu=([\d.]+)")
for path in sys.argv[1:]:
    buckets = defaultdict(lambda: [0.0, 0.0, 0])
    total_gpu = 0.0
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if not m:
            continue
        mode, np_, ptok, pf, nd, dkv, gpu = m.groups()
        np_, ptok, pf, nd, gpu = int(np_), int(ptok), int(pf), int(nd), float(gpu)
        total_gpu += gpu
        if np_ != 1 or ptok < 1024 or gpu <= 0:
            continue
        flops = 4 * D * H * L * (ptok * pf + ptok * ptok / 2.0) + 2 * P * ptok
        key = "<32K" if pf < 32768 else "32-128K" if pf < 131072 else "128-256K" if pf < 262144 else ">=256K"
        b = buckets[key]
        b[0] += flops
        b[1] += gpu / 1e3
        b[2] += 1
    print(path)
    for key in ("<32K", "32-128K", "128-256K", ">=256K"):
        if key in buckets:
            f, t, n = buckets[key]
            print(f"  prefix {key:9s} steps={n:6d} prefill_gpu_s={t:8.1f} rate={f / t / 1e12:6.0f} TFLOP/s")
    print(f"  all steps gpu_s={total_gpu / 1e3:.1f}")
