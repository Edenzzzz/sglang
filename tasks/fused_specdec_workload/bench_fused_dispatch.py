#!/usr/bin/env python3
"""Fused FA3 + FA2 with SM pre-assignment, on the real MTP serving schedule.

Four execution strategies on the same captured batches:

  mono      one persistent launch over every row (today's kernel)
  seq       FA3 on the large-tile rows, then persistent on the rest (vLLM-style)
  fused     the two CO-LAUNCHED with grids pre-assigned by the scheduler:
            FA3 gets c_fa3 CTAs (FLASHINFER_FA3_MAX_CTAS), the persistent kernel
            gets the remaining SMs (FLASHINFER_PERSISTENT_MAX_CLUSTERS, 2
            clusters per SM). Fork/join is device-side so it is capturable.
  fused+sk  same, but FA3 runs split-KV: each large row is expanded into S
            KV-chunks and merged with flashinfer.merge_states, which lifts FA3's
            parallel width above h_qo*ceil(qo/128) -- worth 1.75x standalone.

Routing is on packed_qo_len (tile occupancy) and kv tiles, per mixed_dispatch.
MTP shapes: verify qo=k+1=2, draft qo=1, so on GQA models both are tile-16 and
only the prefill chunk is a large tile.
"""
import argparse, json, os, sys
import numpy as np, torch, flashinfer
from flashinfer.testing.utils import bench_gpu_time

MAXP, MAXQ = 8_000_000, 40_000
MIN_PACKED_QO, MIN_KV_TILES, KV_TILE, FA3_TILE_Q = 48, 32, 128, 128


class Bench:
    def __init__(self, hq, hkv, d):
        self.HQ, self.HKV, self.D = hq, hkv, d
        self.gqa = hq // hkv
        self.kv = torch.empty(MAXP, 2, 1, hkv, d, device="cuda", dtype=torch.bfloat16).normal_()
        self.qbuf = torch.empty(MAXQ, hq, d, device="cuda", dtype=torch.bfloat16).uniform_()
        self.idx = torch.arange(MAXP, dtype=torch.int32, device="cuda")
        self.pa = flashinfer.BatchAttention(kv_layout="NHD")
        self.fa3 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
            kv_layout="NHD", backend="fa3")
        self.s2 = torch.cuda.Stream()
        self.ev_fork, self.ev_join = torch.cuda.Event(), torch.cuda.Event()

    def _ind(self, rows):
        ql = torch.tensor([q for q, _ in rows], dtype=torch.int32)
        kl = torch.tensor([max(k, 1) for _, k in rows], dtype=torch.int32)
        qi = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(ql, 0).int()])
        ki = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(kl, 0).int()])
        return qi, ki, kl

    def plan_pa(self, rows):
        qi, ki, kl = self._ind(rows)
        self.pa.plan(qi, ki, self.idx, kl, self.HQ, self.HKV, self.D, self.D, 1, causal=False,
                     q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        n = int(qi[-1])
        return (self.qbuf[:n],
                torch.empty(n, self.HQ, self.D, device="cuda", dtype=torch.bfloat16),
                torch.empty(n, self.HQ, device="cuda", dtype=torch.float32))

    def plan_fa3(self, rows):
        qi, ki, kl = self._ind(rows)
        self.fa3.plan(qi, ki, self.idx, torch.ones_like(kl), self.HQ, self.HKV, self.D, 1,
                      causal=False, q_data_type=torch.bfloat16,
                      kv_data_type=torch.bfloat16, non_blocking=True)
        return self.qbuf[: int(qi[-1])]

    def plan_fa3_split(self, rows, S):
        """Expand each row into S KV chunks so FA3's width is S x larger."""
        exp, meta = [], []
        for q, kv in rows:
            per = max(kv // S, 1)
            bnds = [(i * per, (i + 1) * per if i < S - 1 else kv) for i in range(S)]
            for a, b in bnds:
                exp.append((q, max(b - a, 1)))
            meta.append((q, len(bnds)))
        qf = self.plan_fa3(exp)
        return qf, meta


def route(rows, gqa):
    fa3, rest = [], []
    for q, kv in rows:
        if q * gqa >= MIN_PACKED_QO and -(-kv // KV_TILE) >= MIN_KV_TILES and q >= 8:
            fa3.append((q, kv))
        else:
            rest.append((q, kv))
    return fa3, rest


def split_sms(fa3_rows, rest_rows, hq, hkv, num_sm=132):
    """Makespan balance. FA3 width is (qo tiles x q heads) with no KV split;
    the persistent side saturates bandwidth well below the full device."""
    if not fa3_rows or not rest_rows:
        return 0 if not fa3_rows else num_sm
    w_f = sum(-(-q // FA3_TILE_Q) * max(kv, 1) for q, kv in fa3_rows) * hq
    w_p = sum(max(1, -(-(q * (hq // hkv)) // KV_TILE)) * max(kv, 1) for q, kv in rest_rows) * hkv
    # rate weighting measured on H200: the small-tile mainloop retires ~2x the
    # work per SM-second of the large-tile one (it is bandwidth-bound).
    a, b = w_f / 1.0, w_p / 2.0
    return int(min(max(round(num_sm * a / (a + b)), 8), num_sm - 8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="/dev/shm/w/mtp_h200.jsonl")
    ap.add_argument("--heads", default="28,4,128")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--split-kv", type=int, default=2)
    ap.add_argument("--no-fsk", action="store_true")
    a = ap.parse_args()
    hq, hkv, d = (int(x) for x in a.heads.split(","))
    b = Bench(hq, hkv, d)
    gqa = hq // hkv
    uncap = lambda: [os.environ.pop(k, None) for k in
                     ("FLASHINFER_FA3_MAX_CTAS", "FLASHINFER_PERSISTENT_MAX_CLUSTERS")]
    med = lambda f: float(np.median(bench_gpu_time(f))) * 1e3

    steps = [json.loads(l) for l in open(a.set) if l.strip()]
    mixed = []
    for s in steps:
        rows = [tuple(x) for c in ("prefill", "verify", "draft") for x in s.get(c, [])]
        if not rows:
            continue
        f, r = route(rows, gqa)
        if f and r and sum(max(k, 1) for _, k in rows) < MAXP - 4096 \
                and sum(q for q, _ in rows) < MAXQ:
            mixed.append((s.get("step"), rows, f, r))
    print(f"{len(mixed)} of {len(steps)} MTP steps are mixed (large tile + small tiles)")
    print(f"{'step':>6} {'f/r':>8} {'SMs':>8} {'mono':>9} {'seq':>9} {'fused':>9} {'fused+sk':>9}")
    res = []
    for step, rows, f, r in mixed[: a.n]:
        c_f = split_sms(f, r, hq, hkv)
        c_p = 132 - c_f
        try:
            uncap()
            qa, oa, la = b.plan_pa(rows)
            t_mono = med(lambda: b.pa.run(qa, b.kv, out=oa, lse=la))
            qf = b.plan_fa3(f)
            qp, op, lp = b.plan_pa(r)
            t_seq = med(lambda: (b.fa3.run(qf, b.kv), b.pa.run(qp, b.kv, out=op, lse=lp)))

            def co(qf_, run_fa3):
                cur = torch.cuda.current_stream()
                b.ev_fork.record(cur)
                b.pa.run(qp, b.kv, out=op, lse=lp)
                with torch.cuda.stream(b.s2):
                    b.s2.wait_event(b.ev_fork)
                    run_fa3()
                    b.ev_join.record(b.s2)
                cur.wait_event(b.ev_join)

            os.environ["FLASHINFER_FA3_MAX_CTAS"] = str(c_f)
            os.environ["FLASHINFER_PERSISTENT_MAX_CLUSTERS"] = str(c_p * 2)
            qf = b.plan_fa3(f); qp, op, lp = b.plan_pa(r)
            t_fused = med(lambda: co(qf, lambda: b.fa3.run(qf, b.kv)))

            if a.no_fsk:
                t_fsk = float("nan")
                uncap()
                print(f"{step:>6} {f'{len(f)}/{len(r)}':>8} {f'{c_f}/{c_p}':>8} {t_mono:>9.1f} "
                      f"{t_seq:>9.1f} {t_fused:>9.1f} {'-':>9}", flush=True)
                res.append(dict(step=step, c_f=c_f, mono=t_mono, seq=t_seq, fused=t_fused,
                                fsk=t_fused))
                continue
            qfs, meta = b.plan_fa3_split(f, a.split_kv)
            qp, op, lp = b.plan_pa(r)
            def fa3_split():
                o, lse = b.fa3.run(qfs, b.kv, return_lse=True)
                off = 0
                for q, S in meta:
                    v = o[off:off + q * S].view(S, q, hq, d).transpose(0, 1).contiguous()
                    s = lse[off:off + q * S].view(S, q, hq).transpose(0, 1).contiguous()
                    flashinfer.merge_states(v, s)
                    off += q * S
            t_fsk = med(lambda: co(qfs, fa3_split))
            uncap()
        except Exception as e:
            print(f"{step:>6} skip {str(e)[:60]}", flush=True); continue
        print(f"{step:>6} {f'{len(f)}/{len(r)}':>8} {f'{c_f}/{c_p}':>8} {t_mono:>9.1f} "
              f"{t_seq:>9.1f} {t_fused:>9.1f} {t_fsk:>9.1f}", flush=True)
        res.append(dict(step=step, c_f=c_f, mono=t_mono, seq=t_seq, fused=t_fused, fsk=t_fsk))
    if not res:
        print("no comparable steps"); return
    A = {k: np.array([x[k] for x in res]) for k in ("mono", "seq", "fused", "fsk")}
    print(f"\n{len(res)} mixed MTP steps")
    for k in ("mono", "seq", "fused", "fsk"):
        print(f"  {k:<9} {A[k].sum():>10.1f} us   vs mono {A['mono'].sum()/A[k].sum():>5.3f}x")
    json.dump(res, open("/dev/shm/w/fused_dispatch.json", "w"))


if __name__ == "__main__":
    main()
