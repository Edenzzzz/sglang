"""Blended-KV sweep: realistic co-tenancy pools where a few long-context rows
sit beside many short-context rows. Uniform-KV sweeps cannot represent these,
yet they are the common production shape."""
import itertools, json, os, sys
import numpy as np, torch
from flashinfer.testing.utils import bench_gpu_time
from gate_sweep import Bench

H = tuple(int(x) for x in os.environ.get("HEADS","28,4,128").split(","))
b = Bench(H)

def measure(dec_kvs, p_lens, p_prefix):
    rows = [(1, k) for k in dec_kvs] + list(zip(p_lens, p_prefix))
    q_lens = torch.tensor([r[0] for r in rows], dtype=torch.int32)
    kv_lens = torch.tensor([r[1] for r in rows], dtype=torch.int32)
    qi = torch.cat([torch.zeros(1,dtype=torch.int32), torch.cumsum(q_lens,0).int()])
    ki = torch.cat([torch.zeros(1,dtype=torch.int32), torch.cumsum(kv_lens,0).int()])
    nq=int(qi[-1]); assert int(ki[-1])<=8_000_000
    qt=b.qbuf[:nq]; last=torch.ones_like(kv_lens)
    b.bp.plan(qi,ki,b.kv_indices,last,H[0],H[1],H[2],1,causal=False,
              q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,non_blocking=True)
    torch.cuda.synchronize()
    bp=float(np.median(bench_gpu_time(lambda: b.bp.run(qt,b.kv))))
    pa={}
    for flip in (True,False):
        b.pa.plan(qi,ki,b.kv_indices,kv_lens,H[0],H[1],H[2],H[2],1,causal=False,
                  q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,flipped_schedule=flip)
        o,l=b.out[:nq],b.lse[:nq]
        pa[flip]=float(np.median(bench_gpu_time(lambda: b.pa.run(qt,b.kv,out=o,lse=l))))
    return bp,pa[True],pa[False]

cases=[]
for n_long,ctx in itertools.product([2,4,8,12,16,24],[16384,32768,49152,65536]):
    for n_short in [0,16,32,48,64]:
        if n_long*ctx + n_short*2048 > 7_000_000: continue
        for q,pfx in [(512,0),(2048,0),(2048,4096)]:
            cases.append((n_long,ctx,n_short,q,pfx))
print(f"{len(cases)} blended cases",flush=True)
res=[]
for i,(n_long,ctx,n_short,q,pfx) in enumerate(cases):
    try:
        dec=[ctx-1]*n_long+[2047]*n_short
        bp,pf,pn=measure(dec,[q],[pfx])
        res.append({"dec_kvs":dec,"p_lens":[q],"p_prefix":[pfx],
                    "bp_ms":bp,"pa_flip_ms":pf,"pa_noflip_ms":pn})
        if i%25==0:
            print(f"[{i+1}/{len(cases)}] {n_long}d@{ctx//1024}K+{n_short}d@2K q{q}p{pfx} spd={bp/min(pf,pn):.2f}x",flush=True)
    except Exception as e:
        print(f"[{i+1}] ERR {str(e)[:90]}",flush=True)
json.dump(res,open(sys.argv[1],"w")); print(f"wrote {len(res)}")
