"""
3-way power/TFLOPS comparison:
  1. triton.do_bench  — DeepGEMM bf16_gemm_nt 8192^3
  2. bench_gpu_time (event) — torch.mm 8192^3 bf16
  3. bench_gpu_time (cudagraph) — torch.mm 8192^3 bf16

Each run: DCGM traces power at 100ms, per-iter CUDA event timing.
"""
import json, os, subprocess, sys, tempfile, time, signal
import numpy as np
import torch

# DCGM Python bindings — typical install path; adjust if needed
_dcgm_path = os.path.join(os.environ.get("DCGM_HOME", "/usr/local/dcgm"), "bindings", "python3")
if os.path.isdir(_dcgm_path):
    sys.path.insert(0, _dcgm_path)

M = N = K = 8192
FLOPS = 2.0 * M * N * K

# Auto-detect physical GPU ID from CUDA_VISIBLE_DEVICES
_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
GPU_ID = int(_vis.split(",")[0]) if _vis else 0


def get_gpu_info():
    """Query TDP and max clock from nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", f"--id={GPU_ID}",
         "--query-gpu=power.limit,clocks.max.gr",
         "--format=csv,noheader,nounits"]).decode().strip()
    parts = out.split(",")
    return float(parts[0].strip()), int(parts[1].strip())  # tdp_w, max_clock_mhz


def start_dcgm(logfile):
    p = subprocess.Popen(
        ["dcgmi", "dmon", "-i", str(GPU_ID), "-e", "157,155,100", "-d", "100"],
        stdout=open(logfile, "w"), stderr=subprocess.DEVNULL)
    time.sleep(0.3)
    return p


def stop_dcgm(p):
    p.send_signal(signal.SIGTERM)
    p.wait()


def parse_dcgm(logfile):
    ip, ap, clk = [], [], []
    with open(logfile) as f:
        for line in f:
            if line.strip().startswith("GPU"):
                p = line.split()
                if len(p) >= 5:
                    ip.append(float(p[2])); ap.append(float(p[3])); clk.append(float(p[4]))
    return np.array(ip), np.array(ap), np.array(clk)


def run_do_bench_deepgemm():
    """Run triton.do_bench on DeepGEMM bf16_gemm_nt, return per-iter times."""
    import deep_gemm
    import triton.testing as tt

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)  # NT layout
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    fn = lambda: deep_gemm.bf16_gemm_nt(a, b, out)

    # do_bench returns summary; we need per-iter. Reimplement the core loop.
    from triton.runtime import driver
    di = driver.active.get_device_interface()
    cache = driver.active.get_empty_cache_for_benchmark()

    fn(); di.synchronize()
    se = di.Event(enable_timing=True); ee = di.Event(enable_timing=True)
    se.record()
    for _ in range(5): driver.active.clear_cache(cache); fn()
    ee.record(); di.synchronize()
    est = se.elapsed_time(ee) / 5
    n_warmup = max(1, int(25 / est))
    n_rep = 5000
    for _ in range(n_warmup): fn()
    starts = [di.Event(enable_timing=True) for _ in range(n_rep)]
    ends = [di.Event(enable_timing=True) for _ in range(n_rep)]
    di.synchronize()
    for i in range(n_rep):
        driver.active.clear_cache(cache)
        starts[i].record(); fn(); ends[i].record()
    di.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def run_do_bench_torchmm():
    """Run do_bench-style loop on torch.mm, return per-iter times."""
    from triton.runtime import driver
    di = driver.active.get_device_interface()
    cache = driver.active.get_empty_cache_for_benchmark()

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    fn = lambda: torch.mm(a, b)

    fn(); di.synchronize()
    se = di.Event(enable_timing=True); ee = di.Event(enable_timing=True)
    se.record()
    for _ in range(5): driver.active.clear_cache(cache); fn()
    ee.record(); di.synchronize()
    est = se.elapsed_time(ee) / 5
    n_warmup = max(1, int(25 / est))
    n_rep = 5000
    for _ in range(n_warmup): fn()
    starts = [di.Event(enable_timing=True) for _ in range(n_rep)]
    ends = [di.Event(enable_timing=True) for _ in range(n_rep)]
    di.synchronize()
    for i in range(n_rep):
        driver.active.clear_cache(cache)
        starts[i].record(); fn(); ends[i].record()
    di.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def run_bgt_event():
    from flashinfer.testing.utils import bench_gpu_time
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    return bench_gpu_time(lambda: torch.mm(a, b), repeat_iters=5000)


def run_bgt_cudagraph():
    from flashinfer.testing.utils import bench_gpu_time
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    return bench_gpu_time(lambda: torch.mm(a, b), use_cuda_graph=True, repeat_iters=5000)


def run_with_dcgm(label, run_fn):
    logfile = os.path.join(tempfile.gettempdir(), f"dcgm_{label}.log")
    torch.cuda.synchronize()
    time.sleep(3)  # cool down
    p = start_dcgm(logfile)
    time.sleep(1)  # baseline samples
    times = run_fn()
    time.sleep(1)  # cooldown samples
    stop_dcgm(p)
    ip, ap, clk = parse_dcgm(logfile)
    return np.array(times), ip, ap, clk


def make_plot(ax_pow, ax_clk, ax_tf, title, times, ip, ap, clk,
              forced_xmax=None, tdp_w=1000, max_clock_mhz=1965):
    tf = FLOPS / (times * 1e-3) / 1e12
    cum = np.cumsum(times + 0.07)

    dcgm_ms = np.arange(len(ip)) * 100.0
    si = next((i for i, p in enumerate(ip) if p > 300), len(ip) // 3)
    dcgm_ms = dcgm_ms - dcgm_ms[si]

    # Find burst end: first iter where TFLOPS drops below 95% of initial median
    initial_tf = np.median(tf[:min(20, len(tf))])
    burst_end_idx = next((i for i in range(20, len(tf)) if tf[i] < initial_tf * 0.95), len(tf))
    burst_end_ms = cum[min(burst_end_idx, len(cum) - 1)]
    end_ms = cum[-1]

    burst_tf = np.median(tf[:burst_end_idx]) if burst_end_idx > 0 else np.median(tf)
    sustained_tf = np.median(tf[burst_end_idx:]) if burst_end_idx < len(tf) else burst_tf

    xmax = (forced_xmax + 300) if forced_xmax else (end_ms + 300)
    xmin = -200

    # Shading
    for ax in [ax_pow, ax_clk, ax_tf]:
        if burst_end_idx < len(tf):
            ax.axvspan(0, burst_end_ms, alpha=0.08, color='green')
            ax.axvspan(burst_end_ms, end_ms, alpha=0.08, color='red')
            ax.axvline(x=burst_end_ms, color='darkgreen', ls='-', lw=2)

    # Power
    m = (dcgm_ms >= xmin) & (dcgm_ms <= xmax)
    if m.any():
        ax_pow.plot(dcgm_ms[m], ip[m], 'r.-', lw=1.5, ms=4, label='Instant power')
    ax_pow.axhline(y=tdp_w, color='black', ls='--', lw=1, label=f'TDP {tdp_w:.0f}W')
    ax_pow.set_ylabel('Power (W)', fontsize=10)
    ax_pow.set_ylim(0, tdp_w * 1.5)
    ax_pow.set_title(title, fontsize=11, fontweight='bold')
    ax_pow.legend(loc='upper right', fontsize=7)
    ax_pow.grid(True, alpha=0.3)

    if burst_end_idx < len(tf):
        ax_pow.annotate(f'Burst ends @ {burst_end_ms:.0f}ms',
                        xy=(burst_end_ms, tdp_w * 1.1),
                        xytext=(burst_end_ms + max(200, end_ms * 0.15), tdp_w * 1.3),
                        fontsize=9, fontweight='bold', color='darkgreen',
                        arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='darkgreen', alpha=0.9))

    if len(ip) > 0 and max(ip) > 300:
        peak_idx = np.argmax(ip)
        ax_pow.annotate(f'Peak: {max(ip):.0f}W\n({max(ip)/tdp_w*100:.0f}% TDP)',
                        xy=(dcgm_ms[peak_idx], max(ip)),
                        xytext=(dcgm_ms[peak_idx] - max(300, end_ms * 0.15), max(ip) - tdp_w * 0.1),
                        fontsize=8, fontweight='bold', color='red',
                        arrowprops=dict(arrowstyle='->', color='red', lw=1),
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='red', alpha=0.9))

    ax_pow.text(burst_end_ms/2, tdp_w * 1.43, f'Burst\n{burst_tf:.0f} TFLOPS', ha='center', fontsize=9, fontweight='bold', color='darkgreen')
    ax_pow.text((burst_end_ms+end_ms)/2, tdp_w * 1.43, f'Sustained\n{sustained_tf:.0f} TFLOPS', ha='center', fontsize=9, fontweight='bold', color='darkred')

    # Clock
    if m.any():
        ax_clk.plot(dcgm_ms[m], clk[m], 'm.-', lw=1.5, ms=4)
    ax_clk.set_ylabel('Clock (MHz)', fontsize=10)
    ax_clk.set_ylim(max_clock_mhz * 0.45, max_clock_mhz * 1.1)
    ax_clk.grid(True, alpha=0.3)
    # Sustained clock annotation
    sustained_mask = ip > tdp_w * 0.8
    if sustained_mask.any():
        sustained_clk = np.median(clk[sustained_mask])
        ax_clk.axhline(y=sustained_clk, color='red', ls='--', lw=1.2)
        ax_clk.axhline(y=max_clock_mhz, color='green', ls='--', lw=1.2)
        ax_clk.text(end_ms * 0.6, sustained_clk - max_clock_mhz * 0.04,
                    f'Sustained: {sustained_clk:.0f} MHz',
                    fontsize=10, fontweight='bold', color='darkred',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='darkred', alpha=0.9))
        ax_clk.text(end_ms * 0.6, max_clock_mhz + max_clock_mhz * 0.015,
                    f'Boost: {max_clock_mhz} MHz',
                    fontsize=9, color='darkgreen',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='darkgreen', alpha=0.9))

    # TFLOPS — auto-scale y axis
    all_tf = tf
    tf_min = np.percentile(all_tf, 1) * 0.95
    tf_max = np.percentile(all_tf, 99) * 1.05
    ax_tf.scatter(cum, tf, s=4, c='steelblue', alpha=0.5)
    ax_tf.axhline(y=burst_tf, color='green', ls='--', lw=1, label=f'Burst: {burst_tf:.0f}')
    if burst_end_idx < len(tf):
        ax_tf.axhline(y=sustained_tf, color='red', ls='--', lw=1, label=f'Sustained: {sustained_tf:.0f}')
    ax_tf.set_ylabel('TFLOPS', fontsize=10)
    ax_tf.set_xlabel('Time (ms)', fontsize=10)
    ax_tf.set_ylim(tf_min, tf_max)
    ax_tf.legend(loc='lower right', fontsize=7)
    ax_tf.grid(True, alpha=0.3)
    ax_tf.set_xlim(xmin, xmax)


def make_overlay_plot(fig, axes, title, results_list, tdp_w=1000, max_clock_mhz=1965):
    """Overlay multiple runs on the same axes. results_list: [(label, times, ip, ap, clk), ...]"""
    ax_pow, ax_clk, ax_tf = axes
    colors_line = ['red', 'blue']
    colors_avg = ['darkred', 'darkblue']
    colors_tf = ['steelblue', 'darkorange']

    xmax_global = 0
    all_tflops = []
    for idx, (label, times, ip, ap, clk) in enumerate(results_list):
        tf = FLOPS / (times * 1e-3) / 1e12
        all_tflops.extend(tf)
        cum = np.cumsum(times + 0.07)
        xmax_global = max(xmax_global, cum[-1])

        dcgm_ms = np.arange(len(ip)) * 100.0
        si = next((i for i, p in enumerate(ip) if p > 300), len(ip) // 3)
        dcgm_ms = dcgm_ms - dcgm_ms[si]

        m = (dcgm_ms >= -200) & (dcgm_ms <= cum[-1] + 300)
        if m.any():
            ax_pow.plot(dcgm_ms[m], ip[m], '.-', color=colors_line[idx], lw=1.5, ms=4, label=f'{label} instant', alpha=0.8)
            ax_pow.plot(dcgm_ms[m], ap[m], '.-', color=colors_avg[idx], lw=1, ms=3, label=f'{label} avg', alpha=0.4)
            ax_clk.plot(dcgm_ms[m], clk[m], '.-', color=colors_line[idx], lw=1.5, ms=4, label=label, alpha=0.8)
        ax_tf.scatter(cum, tf, s=4, c=colors_tf[idx], alpha=0.5, label=f'{label} ({np.median(tf):.0f})', zorder=2)

    ax_pow.axhline(y=tdp_w, color='black', ls='--', lw=1.2, label=f'TDP {tdp_w:.0f}W')
    ax_pow.set_ylabel('Power (W)', fontsize=11)
    ax_pow.set_ylim(0, tdp_w * 1.5)
    ax_pow.set_title(title, fontsize=12, fontweight='bold')
    ax_pow.legend(loc='upper right', fontsize=7)
    ax_pow.grid(True, alpha=0.3)

    ax_clk.set_ylabel('Clock (MHz)', fontsize=11)
    ax_clk.set_ylim(max_clock_mhz * 0.45, max_clock_mhz * 1.1)
    ax_clk.legend(loc='lower left', fontsize=7)
    ax_clk.grid(True, alpha=0.3)

    all_tflops = np.array(all_tflops)
    ax_tf.set_ylabel('TFLOPS', fontsize=11)
    ax_tf.set_xlabel('Time (ms)', fontsize=11)
    ax_tf.set_ylim(np.percentile(all_tflops, 1) * 0.95, np.percentile(all_tflops, 99) * 1.05)
    ax_tf.legend(loc='lower right', fontsize=7)
    ax_tf.grid(True, alpha=0.3)
    ax_tf.set_xlim(-200, xmax_global + 300)


def main():
    torch.cuda.set_device(0)
    gpu_name = torch.cuda.get_device_name(0)
    tdp_w, max_clock = get_gpu_info()
    print(f"GPU: {gpu_name}, {M}^3 bf16, TDP={tdp_w:.0f}W, max_clock={max_clock}MHz")

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_kw = dict(tdp_w=tdp_w, max_clock_mhz=max_clock)

    # --- Run all 4 benchmarks ---
    configs = [
        ("do_bench DeepGEMM bf16", run_do_bench_deepgemm),
        ("do_bench torch.mm", lambda: run_do_bench_torchmm()),
        ("bgt event torch.mm", run_bgt_event),
        ("bgt cudagraph torch.mm", run_bgt_cudagraph),
    ]

    results = {}
    for label, fn in configs:
        print(f"Running: {label}...")
        times, ip, ap, clk = run_with_dcgm(label.replace(" ", "_"), fn)
        results[label] = (label, times, ip, ap, clk)
        print(f"  n={len(times)}, median={np.median(times):.4f}ms, "
              f"TFLOPS={FLOPS/(np.median(times)*1e-3)/1e12:.0f}, "
              f"peak_power={max(ip) if len(ip) else 0:.0f}W")

    # --- Plot 1: torch.mm vs DeepGEMM (2 columns, same format) ---
    two = [results["do_bench torch.mm"], results["do_bench DeepGEMM bf16"]]
    xmax_2 = max(np.sum(t + 0.07) for _, t, _, _, _ in two)
    fig1, axes1 = plt.subplots(3, 2, figsize=(20, 12), gridspec_kw={'height_ratios': [2, 1, 1.2]})
    fig1.suptitle(f'torch.mm vs DeepGEMM bf16 {M}\u00b3 on {gpu_name} (do_bench, no cooldown)',
                  fontsize=14, fontweight='bold')
    for col, (label, times, ip, ap, clk) in enumerate(two):
        make_plot(axes1[0, col], axes1[1, col], axes1[2, col], label, times, ip, ap, clk,
                  forced_xmax=xmax_2, **plot_kw)
    plt.tight_layout()
    fig1.savefig('bench_mm_vs_deepgemm.png', dpi=150, bbox_inches='tight')
    print("Saved: bench_mm_vs_deepgemm.png")
    plt.close(fig1)

    # --- Plot 2: 3-way torch.mm comparison (same x axis) ---
    three = [results["do_bench torch.mm"], results["bgt event torch.mm"], results["bgt cudagraph torch.mm"]]
    xmax_3 = max(np.sum(t + 0.07) for _, t, _, _, _ in three)
    fig2, axes2 = plt.subplots(3, 3, figsize=(28, 12), gridspec_kw={'height_ratios': [2, 1, 1.2]})
    fig2.suptitle(f'torch.mm {M}\u00b3 bf16 on {gpu_name} \u2014 do_bench vs bench_gpu_time vs bench_gpu_time+cudagraph',
                  fontsize=14, fontweight='bold')
    for col, (label, times, ip, ap, clk) in enumerate(three):
        make_plot(axes2[0, col], axes2[1, col], axes2[2, col], label, times, ip, ap, clk,
                  forced_xmax=xmax_3, **plot_kw)
    plt.tight_layout()
    fig2.savefig('bench_3way_power_tflops.png', dpi=150, bbox_inches='tight')
    print("Saved: bench_3way_power_tflops.png")
    plt.close(fig2)


if __name__ == "__main__":
    main()
