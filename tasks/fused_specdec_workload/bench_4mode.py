#!/usr/bin/env python3
"""Ours (4-mode dispatch, scheduler-partitioned grids) vs the real alternatives.

  fused       one persistent launch, 3 tile classes, all SMs        (prev design)
  sequential  FA3 on large prefill THEN persistent on the rest      (vLLM-style)
  dual_naive  same two kernels, two streams, BOTH at full grid      (no scheduler)
  ours        same two kernels, two streams, grids split by the scheduler

Every launch is inside the timed region, including the fork/join events.
"""
import argparse, json, os, sys
import numpy as np, torch, flashinfer
from flashinfer.testing.utils import bench_gpu_time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from four_mode_dispatch import Dispatcher, classify, route, plan_split, FA3_MIN_KV_TILES, KV_TILE

MAX_PAGES, MAX_Q = 8_000_000, 32_768


class Bench:
    def __init__(self, heads):
        self.H_QO, self.H_KV, self.D = heads
        self.kv = torch.empty(MAX_PAGES, 2, 1, self.H_KV, self.D, device="cuda",
                              dtype=torch.bfloat16).normal_()
        self.qbuf = torch.empty(MAX_Q, self.H_QO, self.D, device="cuda",
                                dtype=torch.bfloat16).uniform_()
        self.idx = torch.arange(MAX_PAGES, dtype=torch.int32, device="cuda")
        self.pa = flashinfer.BatchAttention(kv_layout="NHD")
        self.fa3 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
            kv_layout="NHD", backend="fa3")

    def _ind(self, rows):
        ql = torch.tensor([q for q, _ in rows], dtype=torch.int32)
        kl = torch.tensor([max(k, 1) for _, k in rows], dtype=torch.int32)
        qi = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(ql, 0).int()])
        ki = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(kl, 0).int()])
        return qi, ki, ql, kl

    def plan_pa(self, rows, flip):
        qi, ki, ql, kl = self._ind(rows)
        self.pa.plan(qi, ki, self.idx, kl, self.H_QO, self.H_KV, self.D, self.D, 1,
                     causal=False, q_data_type=torch.bfloat16,
                     kv_data_type=torch.bfloat16, flipped_schedule=flip)
        nq = int(qi[-1])
        return (self.qbuf[:nq],
                torch.empty(nq, self.H_QO, self.D, device="cuda", dtype=torch.bfloat16),
                torch.empty(nq, self.H_QO, device="cuda", dtype=torch.float32))

    def plan_fa3(self, rows):
        qi, ki, ql, kl = self._ind(rows)
        self.fa3.plan(qi, ki, self.idx, torch.ones_like(kl), self.H_QO, self.H_KV,
                      self.D, 1, causal=False, q_data_type=torch.bfloat16,
                      kv_data_type=torch.bfloat16, non_blocking=True)
        return self.qbuf[: int(qi[-1])]

    def fits(self, rows):
        return bool(rows) and (sum(max(k, 1) for _, k in rows) < MAX_PAGES - 256
                               and sum(q for q, _ in rows) < MAX_Q)


def uncapped():
    os.environ.pop("FLASHINFER_FA3_MAX_CTAS", None)
    os.environ.pop("FLASHINFER_PERSISTENT_MAX_CLUSTERS", None)


def med(fn):
    return float(np.median(bench_gpu_time(fn))) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="sets/qwen_small_h800_largeprefill.jsonl")
    ap.add_argument("--heads", default="28,4,128")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--oracle", action="store_true", help="sweep split ratios per batch")
    a = ap.parse_args()
    heads = tuple(int(x) for x in a.heads.split(","))
    b = Bench(heads)
    d = Dispatcher(b, heads[0], heads[1])
    gqa = heads[0] // heads[1]
    print(f"heads {heads}  FA3_MIN_KV_TILES={FA3_MIN_KV_TILES} "
          f"({FA3_MIN_KV_TILES*KV_TILE} tokens)\n")
    print(f"{'#':>3} {'fa3/pers rows':>14} {'ours mode':>13} {'fused':>9} {'seq':>9} "
          f"{'dual':>9} {'ours':>9} {'orc':>9}  oracle pick")
    res = []
    for i, bt in enumerate([json.loads(l) for l in open(a.set) if l.strip()][: a.n]):
        cls = classify(bt, gqa)
        allr = [r for c in cls.values() for r in c]
        fa3_rows, pers_rows = route(cls, gqa)
        if not b.fits(allr) or not fa3_rows or not pers_rows:
            continue
        try:
            uncapped()
            qa, oa, la = b.plan_pa(allr, False)
            t_fused = med(lambda: b.pa.run(qa, b.kv, out=oa, lse=la))
            qf = b.plan_fa3(fa3_rows)
            qp, op, lp = b.plan_pa(pers_rows, False)
            t_seq = med(lambda: (b.fa3.run(qf, b.kv),
                                b.pa.run(qp, b.kv, out=op, lse=lp)))
            d.plan(fa3_rows, pers_rows, c_fa3=132, c_pers=132, mode='conc')
            t_dual = med(d.run)
            mode, c_f, c_p = d.plan(fa3_rows, pers_rows)
            t_ours = med(d.run)
            # oracle over the whole strategy space: fused, seq, and every split
            t_orc, best_split = min((t_fused, 'fused'), (t_seq, 'seq'))[0], \
                                'fused' if t_fused < t_seq else 'seq'
            if a.oracle:
                for cf in range(8, 129, 8):
                    d.plan(fa3_rows, pers_rows, c_fa3=cf, c_pers=132 - cf, mode='conc')
                    t = med(d.run)
                    if t < t_orc:
                        t_orc, best_split = t, f'conc{cf}/{132-cf}' 
        except Exception as e:
            print(f"  [{i}] skip {str(e)[:70]}", flush=True); continue
        res.append(dict(i=i, n_fa3=len(fa3_rows), n_pers=len(pers_rows), c_fa3=c_f,
                        c_pers=c_p, mode=mode, fused=t_fused, seq=t_seq, dual=t_dual,
                        ours=t_ours, oracle=t_orc, orc_split=best_split))
        print(f"{i:>3} {f'{len(fa3_rows)}/{len(pers_rows)}':>14} {f'{mode[:4]} {c_f}/{c_p}':>13} "
              f"{t_fused:>9.1f} {t_seq:>9.1f} {t_dual:>9.1f} {t_ours:>9.1f} "
              f"{t_orc:>9.1f} {mode} {best_split}", flush=True)
    uncapped()
    json.dump(res, open(f"dispatch4_{a.tag}.json", "w"))
    if not res:
        print("no comparable batches"); return
    A = {k: np.array([r[k] for r in res]) for k in ("fused", "seq", "dual", "ours", "oracle")}
    print(f"\n{len(res)} batches")
    for k in ("fused", "seq", "dual", "ours", "oracle"):
        print(f"  {k:<10} {A[k].sum():>10.1f} us   vs seq {A['seq'].sum()/A[k].sum():>5.2f}x")
    print(f"  ours beats seq on {(A['ours']<A['seq']).sum()}/{len(res)}, "
          f"dual on {(A['ours']<A['dual']).sum()}/{len(res)}")
    print(f"  ours within 5% of oracle on {(A['ours']<=1.05*A['oracle']).sum()}/{len(res)}"
          f"   best fixed strategy: "
          f"{min(('fused',A['fused'].sum()),('seq',A['seq'].sum()),('dual',A['dual'].sum()),key=lambda x:x[1])}")
    from collections import Counter
    print(f"  modes chosen: {dict(Counter(r['mode'] for r in res))}")
    print(f"  oracle picks: {dict(Counter(str(r['orc_split']) for r in res))}")


if __name__ == "__main__":
    main()
