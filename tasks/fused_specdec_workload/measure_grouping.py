#!/usr/bin/env python3
"""Corrected 4-mode grouping measurement.

An earlier run timed `mixed_decode` as the small-prefill chunk ALONE, with no
decode rows -- which is pure prefill, the one regime the persistent kernel is
not built for. `mixed_decode` must mean "small prefill + decode/draft rows
together", i.e. the LPT kernel's actual regime. This compares:

  A) 3-group: [long-prefix chunks] [verify rows] [small chunks + draft rows]
  B) 4-group: [long-prefix chunks] [verify rows] [small chunks] [draft rows]
  C) monolithic: everything in one kernel
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "debug_kernel_server_perf"))
from gate_sweep import Bench
import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def t_rows(b, bp3, heads, rows):
    if not rows:
        return None
    qs = torch.tensor([q for q, _ in rows], dtype=torch.int32)
    ks = torch.tensor([max(kv, 1) for _, kv in rows], dtype=torch.int32)
    if int(ks.sum()) > 7_500_000 or int(qs.sum()) > 32768:
        return None
    qi = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(qs, 0).int()])
    ki = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(ks, 0).int()])
    nq = int(qi[-1]); last = torch.ones_like(ks); q = b.qbuf[:nq]
    out = {}
    try:
        b.bp.plan(qi, ki, b.kv_indices, last, *heads[:2], heads[2], 1, causal=False,
                  q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, non_blocking=True)
        out["fa2"] = float(np.median(bench_gpu_time(lambda: b.bp.run(q, b.kv)))) * 1e3
        bp3.plan(qi, ki, b.kv_indices, last, *heads[:2], heads[2], 1, causal=False,
                 q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, non_blocking=True)
        out["fa3"] = float(np.median(bench_gpu_time(lambda: bp3.run(q, b.kv)))) * 1e3
        best = None
        for f in (True, False):
            b.pa.plan(qi, ki, b.kv_indices, ks, heads[0], heads[1], heads[2], heads[2], 1,
                      causal=False, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                      flipped_schedule=f)
            o = torch.empty(nq, heads[0], heads[2], device="cuda", dtype=torch.bfloat16)
            l = torch.empty(nq, heads[0], device="cuda", dtype=torch.float32)
            t = float(np.median(bench_gpu_time(lambda: b.pa.run(q, b.kv, out=o, lse=l)))) * 1e3
            best = t if best is None else min(best, t)
        out["pers"] = best
    except Exception:
        return None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="tasks/fused_specdec_workload/sets/qwen_small_h800.jsonl")
    ap.add_argument("--heads", default="28,4,128")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    heads = tuple(int(x) for x in a.heads.split(","))
    b = Bench(heads)
    ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    bp3 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD", backend="fa3")
    batches = [json.loads(l) for l in open(a.set) if l.strip()][: a.n]

    tot = dict(mono_pers=0.0, mono_best=0.0, g3=0.0, g4=0.0)
    lpt_win = 0; lpt_n = 0
    print(f"{'#':>3} {'mono pers':>10} {'MIXED_DEC(chunk+draft)':>24} {'winner':>8} {'3-grp':>8} {'4-grp':>8}")
    for i, bt in enumerate(batches):
        L  = [tuple(x) for x in bt["large_prefill"]]
        V  = [tuple(x) for x in bt["spec_verify"]]
        Sc = [tuple(x) for x in bt["mixed_decode"]]
        D  = [tuple(x) for x in bt["spec_draft"]]
        allr = L + V + Sc + D
        mono = t_rows(b, bp3, heads, allr)
        md   = t_rows(b, bp3, heads, Sc + D)          # <- the real LPT regime
        if mono is None or md is None:
            continue
        parts3 = [t_rows(b, bp3, heads, x) for x in (L, V) if x]
        parts4 = [t_rows(b, bp3, heads, x) for x in (L, V, Sc, D) if x]
        if any(p is None for p in parts3 + parts4):
            continue
        g3 = sum(min(p.values()) for p in parts3) + min(md.values())
        g4 = sum(min(p.values()) for p in parts4)
        w = min(md, key=md.get)
        lpt_n += 1; lpt_win += (w == "pers")
        tot["mono_pers"] += mono["pers"]; tot["mono_best"] += min(mono.values())
        tot["g3"] += g3; tot["g4"] += g4
        print(f"{i:>3} {mono['pers']:>10.1f} "
              f"{'fa2 %.0f fa3 %.0f pers %.0f'%(md['fa2'],md['fa3'],md['pers']):>24} "
              f"{w:>8} {g3:>8.1f} {g4:>8.1f}", flush=True)

    print(f"\n{lpt_n} batches; mixed_decode (chunk+draft) best kernel = persistent on "
          f"{lpt_win}/{lpt_n} ({100*lpt_win/max(lpt_n,1):.0f}%)")
    mp = tot["mono_pers"]
    for nm, v in (("monolithic persistent (today)", mp), ("monolithic best-of-3", tot["mono_best"]),
                  ("3-group (chunk+draft fused)", tot["g3"]), ("4-group (draft peeled off)", tot["g4"])):
        print(f"  {nm:<32} {v:>9.1f} us   {mp/v:>5.2f}x vs today")


if __name__ == "__main__":
    main()
