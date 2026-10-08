"""Probe the fresh-chunk region (chunk prefix -> 0) the sweep never sampled."""
import json
import numpy as np, torch
from flashinfer.testing.utils import bench_gpu_time
from gate_sweep import Bench

HEADS=(28,4,128)
b = Bench(HEADS)

def three_way(d_bs, d_kv, q, pfx):
    d_qo=[1]*d_bs; d_kvl=[d_kv-1]*d_bs
    q_lens=torch.tensor(d_qo+[q], dtype=torch.int32)
    kv_lens=torch.tensor(d_kvl+[pfx], dtype=torch.int32)
    q_indptr=torch.cat([torch.zeros(1,dtype=torch.int32), torch.cumsum(q_lens,0).int()])
    kv_indptr=torch.cat([torch.zeros(1,dtype=torch.int32), torch.cumsum(kv_lens,0).int()])
    nq=int(q_indptr[-1]); last=torch.ones_like(kv_lens)
    qt=b.qbuf[:nq]
    b.bp.plan(q_indptr,kv_indptr,b.kv_indices,last,HEADS[0],HEADS[1],HEADS[2],1,causal=False,
              q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,non_blocking=True)
    torch.cuda.synchronize()
    bp=float(np.median(bench_gpu_time(lambda: b.bp.run(qt,b.kv))))
    pa={}
    for flip in (True,False):
        b.pa.plan(q_indptr,kv_indptr,b.kv_indices,kv_lens,HEADS[0],HEADS[1],HEADS[2],HEADS[2],1,
                  causal=False,q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,flipped_schedule=flip)
        o,l=b.out[:nq],b.lse[:nq]
        pa[flip]=float(np.median(bench_gpu_time(lambda: b.pa.run(qt,b.kv,out=o,lse=l))))
    o_bp=b.bp.run(qt,b.kv); o_pa,_=b.pa.run(qt,b.kv,out=b.out[:nq],lse=b.lse[:nq])
    err=float((o_bp-o_pa).abs().max())
    return bp,pa[True],pa[False],err

print(f"{'composition':<32} {'BP ms':>8} {'PAflip':>8} {'PAnofl':>8} {'spd':>8} {'err':>7}")
rows=[]
for d_bs,d_kv in [(8,81920),(16,81920),(32,81920),(16,32768),(32,16384),(64,8192)]:
    for q in [512,2048,4096]:
        for pfx in [0,128,512]:
            try:
                bp,pf,pn,err=three_way(d_bs,d_kv,q,pfx)
                best=min(pf,pn); rows.append(dict(d_bs=d_bs,d_kv=d_kv,q=q,pfx=pfx,bp_ms=bp,pa_flip_ms=pf,pa_noflip_ms=pn,err=err))
                print(f"{d_bs:>3}d@{d_kv//1024:>3}K+{q:>5}q@pfx{pfx:<5}{'':<6} {bp:>8.3f} {pf:>8.3f} {pn:>8.3f} {bp/best:>7.2f}x {err:>7.4f}", flush=True)
            except Exception as e:
                print(f"{d_bs}d@{d_kv} q{q} pfx{pfx}: ERROR {str(e)[:100]}", flush=True)
json.dump(rows, open('probe_fresh.json','w'))
if rows: print(f"\nmax abs err: {max(r['err'] for r in rows):.4f}")
