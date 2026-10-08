#!/usr/bin/env python3
"""px.py for multi-node trials: run a script on one node (instance) of a merlin job run.

  from px_multi import Cluster
  c = Cluster(sid="...", cp="cn-seed", nodes=["host-a", "host-b", ...])
  c.run(0, "nvidia-smi -L")        # node index -> pod_name substring
  c.run_all("hostname")            # every node, in parallel
  c.push(0, local_tgz, remote_b64, dest)

Proxy env is stripped for every call (merlin-cli otherwise fails with a bare EOF).
"""
import base64, concurrent.futures as cf, json, os, subprocess, tempfile

ENV = {k: v for k, v in os.environ.items()
       if k not in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")}


class Cluster:
    def __init__(self, sid, cp="cn-seed", nodes=None):
        self.sid, self.cp, self.nodes = sid, cp, nodes or []

    def _call(self, node, script, timeout):
        body = {"job_run_id": self.sid, "exec": script, "exec_timeout_seconds": 100}
        if node is not None and self.nodes:
            body["pod_name"] = self.nodes[node]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(body, f)
            path = f.name
        try:
            p = subprocess.run(["merlin-cli", "--control-plane", self.cp, "job-v2", "runs",
                                "execute-script", "--from-file", path],
                               capture_output=True, text=True, timeout=timeout, env=ENV)
            out = "\n".join(l for l in (p.stdout + p.stderr).splitlines()
                            if "ZTI token zone" not in l)
            return p.returncode, out
        finally:
            os.unlink(path)

    def run(self, node, script, timeout=200):
        return self._call(node, script, timeout)

    def run_all(self, script, timeout=200):
        with cf.ThreadPoolExecutor(len(self.nodes) or 1) as ex:
            return list(ex.map(lambda i: self._call(i, script, timeout), range(len(self.nodes))))

    def push(self, node, local, remote_b64, dest, chunk=60000):
        b64 = base64.b64encode(open(local, "rb").read()).decode()
        rc, out = self.run(node, f"rm -f {remote_b64}; mkdir -p {dest}")
        if rc:
            return False
        for off in range(0, len(b64), chunk):
            rc, out = self.run(node, f"printf %s '{b64[off:off + chunk]}' >> {remote_b64}")
            if rc:
                return False
        rc, out = self.run(node, f"[ $(wc -c < {remote_b64}) = {len(b64)} ] && "
                                 f"base64 -d {remote_b64} | tar xzf - -C {dest} && echo EXTRACTED")
        return "EXTRACTED" in out
