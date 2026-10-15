import json,sys,os,re
root=sys.argv[1]
arms=[("MTP","MTP_LPT"),("SPLIT_C3072","MTPMIX_SPLIT"),("EDFR50_F2048","MTPMIX_SPLIT_EDFR")]
keys=[("Output token throughput (tok/s)","tput"),("Mean TTFT (ms)","ttft"),("Median TTFT (ms)","mttft"),("P99 TTFT (ms)","p99ttft"),("Mean TPOT (ms)","tpot"),("Mean ITL (ms)","itl"),("P99 ITL (ms)","p99itl"),("Accept length","acc"),("Concurrency","inflight")]
rows=[]
for pt in sorted(os.listdir(root)):
    if not os.path.isdir(f"{root}/{pt}"): continue
    for lab,m in arms:
        for t in ("agent","chat"):
            f=f"{root}/{pt}/{lab}/bench_{m}_{t}.log"
            if not os.path.exists(f): continue
            txt=open(f).read(); r={"point":pt,"arm":lab,"tenant":t}
            for k,n in keys:
                mm=re.search(re.escape(k)+r":\s+([0-9.]+)",txt); r[n]=float(mm.group(1)) if mm else None
            rows.append(r)
cols=["point","arm","tenant"]+[n for _,n in keys]
print(",".join(cols))
for r in rows: print(",".join(str(r[c]) for c in cols))
print("#REL vs MTP (%)")
for r in rows:
    if r["arm"]=="MTP": continue
    b=[x for x in rows if x["point"]==r["point"] and x["tenant"]==r["tenant"] and x["arm"]=="MTP"][0]
    print(r["point"],r["arm"],r["tenant"]," ".join(f"{n}={100*(r[n]/b[n]-1):+.0f}" for n in ("tput","ttft","mttft","tpot","itl","p99itl") if r[n] and b[n]))
