"""Per-kernel times of native v3 (the current serving build) and sglang's FA3 call on one capture."""
import sys
from collections import defaultdict
from pathlib import Path
import torch
sys.path.insert(0, "/dev/shm/w/ub")
import bench_captured_mixed_20261023 as B  # noqa: E402

case = B.load_case(Path(sys.argv[1]))
launches = B.launches_for(case, policy_v3=True)
for arm in ("native_pdl_geometry_v3", "fa3_full_default", "fa3_dual_split32"):
    run = launches[arm]
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            run()
        torch.cuda.synchronize()
    k = defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            k[e.name[:70]] += e.device_time / 10 / 1e3
    print(arm, " | ".join(f"{a}={b:.4f}ms" for a, b in sorted(k.items(), key=lambda x: -x[1])[:5]), flush=True)
