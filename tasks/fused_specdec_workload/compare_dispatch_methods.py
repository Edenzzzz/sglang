#!/usr/bin/env python3
"""Ours (one partitioned cooperative launch) vs the two real alternatives.

Methods compared on the same captured fused batches:

  A) monolithic      one persistent launch, all classes share all SMs (today)
  B) sequential      FA3 on the large-prefill rows, THEN persistent on the rest,
                     same stream -- what vLLM-style split execution does
  C) dual_stream     the same two kernels on two streams, overlapped
  D) ours            one persistent launch, 3 tile classes, scheduler-assigned
                     SM proportions (FLASHINFER_SM_PARTITION=1)

B and C are timed with BOTH launches inside the timed region, so their launch
and (for C) cross-stream sync costs are counted, which is the honest comparison
-- an earlier estimate of this shape summed separately-timed pieces and was not.
"""
import argparse, json, os, sys
import numpy as np, torch, flashinfer
from flashinfer.testing.utils import bench_gpu_time

MAX_PAGES, MAX_Q = 8_000_000, 32_768
LARGE_PREFIX = 8192


class Bench:
    def __init__(self, heads):
        self.H_QO, self.H_KV, self.D = heads
        self.kv = torch.empty(MAX_PAGES, 2, 1, self.H_KV, self.D,
                              device="cuda", dtype=torch.bfloat16).normal_()
        self.qbuf = torch.empty(MAX_Q, self.H_QO, self.D, device="cuda",
                                dtype=torch.bfloat16).uniform_()
        self.idx = torch.arange(MAX_PAGES, dtype=torch.int32, device="cuda")
        self.pa = flashinfer.BatchAttention(kv_layout="NHD")
        self.fa3 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
            kv_layout="NHD", backend="fa3")
        self.s2 = torch.cuda.Stream()
        self.ev = torch.cuda.Event()

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
        self.fa3.plan(qi, ki, self.idx, torch.ones_like(kl), self.H_QO, self.H_KV, self.D, 1,
                      causal=False, q_data_type=torch.bfloat16,
                      kv_data_type=torch.bfloat16, non_blocking=True)
        return self.qbuf[: int(qi[-1])]

    def fits(self, rows):
        if not rows:
            return False
        return (sum(max(k, 1) for _, k in rows) < MAX_PAGES - 256
                and sum(q for q, _ in rows) < MAX_Q)


def best_pa(b, rows):
    out = None
    for flip in (True, False):
        q, o, l = b.plan_pa(rows, flip)
        t = float(np.median(bench_gpu_time(lambda: b.pa.run(q, b.kv, out=o, lse=l)))) * 1e3
        out = t if out is None else min(out, t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="tasks/fused_specdec_workload/sets/qwen_small_h800.jsonl")
    ap.add_argument("--heads", default="28,4,128")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    heads = tuple(int(x) for x in a.heads.split(","))
    b = Bench(heads)
    part = os.environ.get("FLASHINFER_SM_PARTITION", "0") != "0"
    print(f"heads {heads}  SM_PARTITION={'on' if part else 'off'}\n")

    res = []
    print(f"{'#':>3} {'split rows':>12} {'fused':>9} {'seq':>9} {'stream':>9}")
    for i, bt in enumerate([json.loads(l) for l in open(a.set) if l.strip()][: a.n]):
        big = [tuple(x) for x in bt["large_prefill"]]
        rest = [tuple(x) for c in ("spec_verify", "mixed_decode", "spec_draft") for x in bt[c]]
        allr = big + rest
        if not b.fits(allr) or not big or not rest:
            continue
        try:
            # "fused" is one persistent launch over every class. Whether the
            # scheduler partitions SMs is a plan-time env flag, so ours vs
            # monolithic is this same column measured in two processes
            # (FLASHINFER_SM_PARTITION=1 / =0) -- not two calls here.
            fused = best_pa(b, allr)
            qf = b.plan_fa3(big)
            qp, op, lp = b.plan_pa(rest, False)

            def seq():
                b.fa3.run(qf, b.kv)
                b.pa.run(qp, b.kv, out=op, lse=lp)
            t_seq = float(np.median(bench_gpu_time(seq))) * 1e3

            # FA3 on the side stream, the cooperative persistent kernel on the
            # default stream (its cta_counter is allocated against that stream);
            # the event is created once, not per timed iteration.
            def streams():
                with torch.cuda.stream(b.s2):
                    b.fa3.run(qf, b.kv)
                    b.ev.record()
                b.pa.run(qp, b.kv, out=op, lse=lp)
                b.ev.synchronize()
            t_str = float(np.median(bench_gpu_time(streams))) * 1e3
        except Exception as e:
            print(f"  [{i}] skip {str(e)[:70]}", flush=True)
            continue
        res.append(dict(i=i, n_big=len(big), n_rest=len(rest), fused=fused,
                        seq=t_seq, stream=t_str))
        print(f"{i:>3} {f'{len(big)}+{len(rest)}':>12} {fused:>9.1f} {t_seq:>9.1f} "
              f"{t_str:>9.1f}", flush=True)

    out = f"tasks/fused_specdec_workload/dispatch_cmp_{a.tag}.json"
    json.dump(res, open(out, "w"))
    if not res:
        print("no comparable batches"); return
    A = {k: np.array([r[k] for r in res]) for k in ("fused", "seq", "stream")}
    print(f"\n{len(res)} batches with both a large-prefill row and a decode population")
    for k in ("fused", "seq", "stream"):
        print(f"  {k:<10} {A[k].sum():>10.1f} us   vs seq {A['seq'].sum()/A[k].sum():>5.2f}x")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
