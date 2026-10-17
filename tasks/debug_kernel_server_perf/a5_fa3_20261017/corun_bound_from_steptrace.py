"""Upper bound on what FA3 || FA2 co-running can save in serving, from STEPTRACE lines (SGLANG_STEP_TRACE=1).

Per step: t3 = FA3 attention time of the prefill rows (causal FLOPs 4*D*H*L*(ptok*pf + ptok^2/2) at FA3_TFLOPS);
t2 = FA2 time of the decode/verify rows (KV bytes dkv * L * 2 * KVH * D * 2 at HBM_GBS). Sequential split costs t3 + t2;
an ideal co-run costs max(t3, t2), so the saving is min(t3, t2) -- the A5.3 kernel table realised ~all of it when the
roles were comparable (append 512 @ 48K + 46 verify: 0.59 of a 0.6 ms FA3 wave hidden). Reported as a share of all
steps' GPU time and of mixed (np>0 and nd>0) steps' GPU time, with the distribution of min/max.
Usage: corun_bound_from_steptrace.py <server_log> [...]"""
import re
import sys

L, H, KVH, D = 28, 28, 4, 128
FA3_TFLOPS, HBM_GBS = 500.0, 3000.0
PAT = re.compile(r"STEPTRACE (\w+) .*?np=(\d+) ptok=(\d+) pf=(\d+) nd=(\d+) dkv=(\d+).*?gpu=([\d.]+)")
for path in sys.argv[1:]:
    tot = mixed = save = att = 0.0
    n = nm = 0
    ratios = []
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if not m:
            continue
        _, np_, ptok, pf, nd, dkv, gpu = m.groups()
        np_, ptok, pf, nd, dkv, gpu = int(np_), int(ptok), int(pf), int(nd), int(dkv), float(gpu)
        tot += gpu
        n += 1
        if np_ == 0 or nd == 0:
            continue
        nm += 1
        mixed += gpu
        t3 = 4 * D * H * L * (ptok * pf + ptok * ptok / 2.0) / (FA3_TFLOPS * 1e9)
        t2 = dkv * L * 2 * KVH * D * 2 / (HBM_GBS * 1e6)
        att += t3 + t2
        save += min(t3, t2)
        ratios.append(min(t3, t2) / max(t3, t2) if max(t3, t2) > 0 else 0.0)
    ratios.sort()
    q = lambda f: ratios[int(f * (len(ratios) - 1))] if ratios else 0.0
    print(f"{path}\n  steps {n} (mixed {nm}); GPU ms total {tot:.0f}, mixed {mixed:.0f}; modelled attention in mixed "
          f"steps {att:.0f} ms ({att / max(mixed, 1e-9):.0%})")
    print(f"  ideal co-run saving {save:.0f} ms = {save / max(tot, 1e-9):.1%} of all GPU time, "
          f"{save / max(mixed, 1e-9):.1%} of mixed-step time; min/max p10/p50/p90 {q(.1):.2f}/{q(.5):.2f}/{q(.9):.2f}")
