import re,sys,os
root=sys.argv[1]
K=[("Output token throughput (tok/s)","tput"),("Mean TTFT (ms)","ttft"),("Mean TPOT (ms)","tpot"),("Mean ITL (ms)","itl"),("P99 ITL (ms)","p99itl"),("Concurrency","inflight"),("Accept length","acc")]
def rd(f):
    t=open(f).read(); return {n:float(re.search(re.escape(k)+r":\s+([0-9.]+)",t).group(1)) for k,n in K}
for pt in ["cohort_c8_chat4","cohort_c16_chat4","cohort_c8_chat8","cohort_c16_chat8"]:
    for ten in ("agent","chat"):
        m=rd(f"{root}/{pt}_MTP/bench_MTP_LPT_{ten}.log"); e=rd(f"{root}/{pt}_EDFP/bench_MTPMIX_SPLIT_EDFR_{ten}.log")
        rel=" ".join(f"{n}={100*(e[n]/m[n]-1):+.0f}%" for n in ("tput","ttft","tpot","itl","p99itl"))
        print(f"{pt} {ten}: MTP tput={m['tput']:.0f} ttft={m['ttft']:.0f} itl={m['itl']:.1f} | EDFP tput={e['tput']:.0f} ttft={e['ttft']:.0f} itl={e['itl']:.1f} acc={m['acc']}/{e['acc']} | {rel}")
