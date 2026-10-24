import fcntl
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


root = Path("/dev/shm/w")
directory = Path(sys.argv[1]).resolve()
original = root / "b1_kernel_concurrency_overlay_20261014/audit_kernel_concurrency_original.py"
runpy.run_path(str(original), run_name="__main__")
target = root / "b1_kernel_concurrency_20261014/mix128_c24/BASE_2"
if directory == target:
    state_path = root / "b1_kernel_concurrency_20261014/stock_fa3_pdl_midpoint_status.json"
    if state_path.exists():
        raise RuntimeError("FA3 midpoint hook already has an execution record")
    owner = 766826
    assert os.getppid() == owner
    lock_path = root / "a2_kernel.lock"
    assert os.fstat(9).st_ino == lock_path.stat().st_ino
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {"owner": owner, "started": time.time(), "trigger": str(directory)}
    state_path.write_text(json.dumps(status, indent=2) + "\n")
    environment = {**os.environ, "INHERIT_LOCK": "1", "GPU_LOCK_OWNER_PID": str(owner)}
    with (root / "stock_fa3_pdl_midpoint_20261014.log").open("w") as output:
        result = subprocess.run(
            ["bash", str(root / "stock_fa3_pdl_20261014.sh")],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            pass_fds=(9,),
        )
    status.update(returncode=result.returncode, finished=time.time())
    state_path.write_text(json.dumps(status, indent=2) + "\n")
    remaining = subprocess.run(
        ["pgrep", "-af", "[s]tock_fa3_reference.py|[t]est_stock_fa3.py"],
        capture_output=True,
        text=True,
    )
    if remaining.returncode == 0:
        raise RuntimeError("FA3 child is still running; refusing to resume serving")
    print("FA3_MIDPOINT_RETURN", json.dumps(status), flush=True)
