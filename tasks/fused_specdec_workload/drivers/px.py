#!/usr/bin/env python3
# Chunked push through the webshell. The exec argument is capped near 90-100 KB,
# so the tarball goes over base64'd in chunks, appended on the far side, and the
# byte count is checked BEFORE decoding -- a truncated chunk otherwise surfaces
# much later as a corrupt tar header.
import base64, json, subprocess, sys, os, tempfile

SID = "444d5bc154bd9050"
# Proxy vars MUST be stripped. NO_PROXY (uppercase) does not list .bytedance.net or
# .byteintl.net while no_proxy (lowercase) does, and Go's ProxyFromEnvironment prefers the
# uppercase one -- so merlin-cli proxies these hosts, the proxy fails TLS, and Go surfaces
# it as a bare "EOF". Any merlin-cli call made WITHOUT this stripping will appear to fail
# as a network error even when the control plane is perfectly healthy.
ENV = {k: v for k, v in os.environ.items()
       if k not in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")}

def probe_trial():
    """Is the trial still alive? An expired pod makes execute-script return the SAME
    bare {"error": "EOF"} as a network fault, which cost an hour of chasing a
    non-existent control-plane outage. Check this before blaming the network."""
    import subprocess as sp
    p = sp.run(["merlin-cli", "--control-plane", "i18n-bd", "job-v2", "runs", "get",
                "--json", json.dumps({"sid": SID})],
               capture_output=True, text=True, timeout=120, env=ENV)
    t = p.stdout + p.stderr
    if '"status"' not in t:
        return "unknown (API unreachable): " + t.strip()[-200:]
    st = t.split('"status"')[1].split('"')[1]
    return st


def run(script, timeout=200):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"job_run_id": SID, "exec": script, "exec_timeout_seconds": 100}, f)
        path = f.name
    try:
        p = subprocess.run(
            ["merlin-cli", "--control-plane", "i18n-bd", "job-v2", "runs",
             "execute-script", "--from-file", path],
            capture_output=True, text=True, timeout=timeout, env=ENV)
        out = "\n".join(l for l in (p.stdout + p.stderr).splitlines()
                        if "ZTI token zone" not in l)
        return p.returncode, out
    finally:
        os.unlink(path)

def push(local, remote_b64, dest):
    b64 = base64.b64encode(open(local, "rb").read()).decode()
    n = len(b64)
    print(f"pushing {local}: {os.path.getsize(local)} raw, {n} b64", flush=True)
    rc, out = run(f"rm -f {remote_b64}; mkdir -p {dest}")
    if rc: print("prep failed:", out); return False
    CH = 60000
    for i, off in enumerate(range(0, n, CH), 1):
        part = b64[off:off + CH]
        rc, out = run(f"printf %s '{part}' >> {remote_b64}")
        if rc:
            print(f"chunk {i} failed:", out[-500:]); return False
        print(f"  chunk {i} ok ({min(off+CH,n)}/{n})", flush=True)
    rc, out = run(
        f"set -e; got=$(wc -c < {remote_b64}); echo \"on pod: $got expected {n}\"; "
        f"[ \"$got\" = \"{n}\" ] || {{ echo TRUNCATED; exit 1; }}; "
        f"base64 -d {remote_b64} | tar xzf - -C {dest}; echo EXTRACTED")
    print(out)
    return rc == 0 and "EXTRACTED" in out

if __name__ == "__main__":
    ok = push(sys.argv[1], sys.argv[2], sys.argv[3])
    sys.exit(0 if ok else 1)
