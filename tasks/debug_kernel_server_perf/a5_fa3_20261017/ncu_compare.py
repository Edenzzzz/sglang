"""Side-by-side NCU raw metrics for A5.0c: python ncu_compare.py <dir> <fixture>
Reads <dir>/<fixture>_{oursOFF,oursON,upstream}.csv (ncu --page raw --csv) and prints the metrics that separate a
mainloop gap from per-tile overhead: duration, tensor pipe, issued instructions by pipe, warp stall reasons, launch
geometry and memory traffic."""
import csv, re, sys

d, fx = sys.argv[1], sys.argv[2]
variants = ["oursOFF", "oursON", "upstream"]
PATTERNS = [r"^gpu__time_duration\.sum$", r"^launch__(grid_size|block_size|registers_per_thread|shared_mem_per_block_dynamic|waves_per_multiprocessor)$",
            r"^sm__warps_active\.avg\.pct_of_peak", r"^sm__pipe_tensor_op_gmma_cycles_active\.avg\.pct_of_peak",
            r"^sm__inst_executed_pipe_(xu|fma|alu|lsu|uniform|tensor_op_gmma|fp16|adu|cbu)\.sum$", r"^smsp__inst_executed\.sum$",
            r"^sm__throughput\.avg\.pct_of_peak", r"^dram__bytes_read\.sum$", r"^lts__t_sector_hit_rate\.pct$",
            r"^smsp__average_warp_latency_issue_stalled_[a-z_]+\.ratio$", r"^smsp__issue_active\.avg\.pct_of_peak",
            r"^l1tex__data_pipe_lsu_wavefronts_mem_shared\.sum$", r"^smsp__sass_inst_executed_op_(global|shared)_(ld|st)\.sum$"]
rows = {}
for v in variants:
    try:
        r = list(csv.reader(open(f"{d}/{fx}_{v}.csv")))
    except FileNotFoundError:
        continue
    hdr, units, vals = r[0], r[1], r[2]
    rows[v] = {h: (val, u) for h, u, val in zip(hdr, units, vals)}
keys = sorted({k for v in rows.values() for k in v if any(re.search(p, k) for p in PATTERNS)})
print(f"{'metric':72s} " + " ".join(f"{v:>14s}" for v in variants))
for k in keys:
    vals = [rows.get(v, {}).get(k, ("-", ""))[0] for v in variants]
    unit = next((rows[v][k][1] for v in rows if k in rows[v]), "")
    print(f"{k[:66] + ' ' + unit[:5]:72s} " + " ".join(f"{x[:14]:>14s}" for x in vals))
