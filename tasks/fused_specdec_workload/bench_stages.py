#!/usr/bin/env python3
"""Stage-boundary cost: what the fused kernel is actually worth per SERVER step.

Speculative decoding today runs two stages per step, each with its own plan(),
host->device metadata, launch and sync:

  staged   plan(draft) -> launch -> sync ; plan(verify) -> launch -> sync
           (+ a third stage when a prefill chunk is admitted)

A 4-mode fused schedule (SparseSpec-style) puts verify(batch N-1), draft(batch N)
and the prefill chunk in ONE batch, so there is one plan and one launch:

  unified  plan(all rows) -> launch -> sync

This measures wall-clock per step INCLUDING the CPU plan time, which is what the
server sees -- a kernel-only speedup is diluted by metadata work that does not
shrink, and the stage collapse removes a boundary the two-kernel design cannot.
"""
import json, time, sys
import numpy as np, torch, flashinfer

torch.manual_seed(0)
HQ, HKV, Dh = 28, 4, 128
GQA = HQ // HKV
MAXP = 3_000_000
kv = torch.empty(MAXP, 2, 1, HKV, Dh, device="cuda", dtype=torch.bfloat16).normal_()
idx = torch.arange(MAXP, dtype=torch.int32, device="cuda")

def ind(rows):
    ql = torch.tensor([q for q, _ in rows], dtype=torch.int32)
    kl = torch.tensor([max(k, 1) for _, k in rows], dtype=torch.int32)
    qi = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(ql, 0).int()])
    ki = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(kl, 0).int()])
    return qi, ki, kl

def timed_stage(wr, rows, reps):
    """plan + launch + sync, as a server step would do it."""
    qi, ki, kl = ind(rows)
    n = int(qi[-1])
    q = torch.empty(n, HQ, Dh, device="cuda", dtype=torch.bfloat16).uniform_()
    o = torch.empty(n, HQ, Dh, device="cuda", dtype=torch.bfloat16)
    l = torch.empty(n, HQ, device="cuda", dtype=torch.float32)
    # warmup
    wr.plan(qi, ki, idx, kl, HQ, HKV, Dh, Dh, 1, causal=False,
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    wr.run(q, kv, out=o, lse=l); torch.cuda.synchronize()
    t_plan = t_tot = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        wr.plan(qi, ki, idx, kl, HQ, HKV, Dh, Dh, 1, causal=False,
                q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        t1 = time.perf_counter()
        wr.run(q, kv, out=o, lse=l)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_plan += (t1 - t0); t_tot += (t2 - t0)
    return t_plan / reps * 1e6, t_tot / reps * 1e6

steps = [json.loads(l) for l in open("/dev/shm/w/mtp_h200_chunk512.jsonl") if l.strip()]
sel = []
for s in steps:
    pre = [tuple(x) for x in s.get("prefill", [])]
    ver = [tuple(x) for x in s.get("verify", [])]
    dra = [tuple(x) for x in s.get("draft", [])]
    if ver and dra and sum(max(k,1) for _,k in pre+ver+dra) < MAXP - 4096:
        sel.append((s.get("step"), pre, ver, dra))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
sel = [sel[i] for i in np.linspace(0, len(sel)-1, min(N, len(sel))).astype(int)]
print(f"{len(sel)} steps with both verify and draft rows\n")
print(f"{'step':>6} {'v/d/p':>10} {'staged us':>10} {'(plan)':>8} {'unified us':>11} {'(plan)':>8} {'gain':>7}")

wr = flashinfer.BatchAttention(kv_layout="NHD")
tot_s = tot_u = tot_sp = tot_up = 0.0
for step, pre, ver, dra in sel:
    reps = 20
    # staged: draft stage, verify stage, prefill stage (as separate launches)
    sp = st = 0.0
    for rows in ([dra, ver] + ([pre] if pre else [])):
        p, t = timed_stage(wr, rows, reps); sp += p; st += t
    # unified: one plan, one launch over every row
    up, ut = timed_stage(wr, pre + ver + dra, reps)
    print(f"{step:>6} {f'{len(ver)}/{len(dra)}/{len(pre)}':>10} {st:>10.1f} {sp:>8.1f} "
          f"{ut:>11.1f} {up:>8.1f} {st/ut:>6.3f}x", flush=True)
    tot_s += st; tot_u += ut; tot_sp += sp; tot_up += up

print(f"\ntotals over {len(sel)} steps")
print(f"  staged  {tot_s:>9.1f} us/step   of which plan {tot_sp:>8.1f} us ({100*tot_sp/tot_s:.0f}%)")
print(f"  unified {tot_u:>9.1f} us/step   of which plan {tot_up:>8.1f} us ({100*tot_up/tot_u:.0f}%)")
print(f"  stage collapse is worth {tot_s/tot_u:.3f}x per step")
