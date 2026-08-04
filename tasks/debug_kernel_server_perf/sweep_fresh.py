"""Extension sweep over the fresh/short-prefix chunk region (prefix < 512),
which the original gate sweep never sampled. Same JSON schema as gate_sweep3."""
import itertools, json, os, sys
import numpy as np, torch
from flashinfer.testing.utils import bench_gpu_time
from gate_sweep import Bench

HEADS = tuple(int(x) for x in os.environ.get("HEADS","28,4,128").split(","))
b = Bench(HEADS)

cases=[]
for d_bs, d_kv in itertools.product([1,2,4,8,16,32,64,128],[2048,8192,32768,98304]):
    if d_bs*d_kv > 7_000_000: continue
    for q in [128,512,2048,4096,8192]:
        for pfx in [0,128,256]:
            cases.append((d_bs,d_kv,[q],[pfx]))
# multi-prefill: one fresh chat chunk + one long continuation (true mixed tenancy)
for d_bs,d_kv in [(8,81920),(16,81920),(32,32768)]:
    for q1,p1 in [(2048,0),(4096,0)]:
        for q2,p2 in [(2048,32768),(4096,65536)]:
            if d_bs*d_kv+p1+p2 < 7_900_000:
                cases.append((d_bs,d_kv,[q1,q2],[p1,p2]))
print(f"{len(cases)} cases", flush=True)

res=[]
for i,(d_bs,d_kv,p_lens,p_prefix) in enumerate(cases):
    try:
        d_qo=[1]*d_bs; d_kvl=[d_kv-1]*d_bs
        q_lens=torch.tensor(d_qo+p_lens,dtype=torch.int32)
        kv_lens=torch.tensor(d_kvl+p_prefix,dtype=torch.int32)
        q_indptr=torch.cat([torch.zeros(1,dtype=torch.int32),torch.cumsum(q_lens,0).int()])
        kv_indptr=torch.cat([torch.zeros(1,dtype=torch.int32),torch.cumsum(kv_lens,0).int()])
        nq=int(q_indptr[-1]); last=torch.ones_like(kv_lens); qt=b.qbuf[:nq]
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
        res.append({"d_bs":d_bs,"d_kv":d_kv,"p_lens":p_lens,"p_prefix":p_prefix,
                    "bp_ms":bp,"pa_flip_ms":pa[True],"pa_noflip_ms":pa[False]})
        if i%40==0:
            print(f"[{i+1}/{len(cases)}] {d_bs}d@{d_kv} q{p_lens}+{p_prefix} spd={bp/min(pa.values()):.2f}x",flush=True)
    except Exception as e:
        print(f"[{i+1}] ERROR {str(e)[:100]}",flush=True)
json.dump(res, open(sys.argv[1],"w"))
print(f"wrote {len(res)} rows to {sys.argv[1]}")
