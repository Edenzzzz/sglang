import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ANALYZER = Path(__file__).resolve().parents[2] / "drivers/serving/eval_curves.py"


class EvalCurvesTest(unittest.TestCase):
    def write_run(self, root, directory, arm, tenant, scale=1.0):
        path = root / directory / f"{arm}_{tenant}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "duration": 10.0,
                    "completed": 2,
                    "output_lens": [3, 3],
                    "ttfts": [0.1 * scale, 0.3 * scale],
                    "itls": [
                        [0.02 * scale, 0.02 * scale],
                        [0.04 * scale, 0.04 * scale],
                    ],
                }
            )
        )

    def analyze(self, root, *arguments):
        return subprocess.run(
            [sys.executable, str(ANALYZER), "--root", str(root), *map(str, arguments)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_unloaded_tenant_layout_maps_all_context_ranges(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            reference = Path(temporary) / "unloaded"
            scales = {}
            for index, workload in enumerate(("mix32", "mix64", "mix128"), start=1):
                for tenant_index, tenant in enumerate(("chat", "codex"), start=1):
                    scale = index * tenant_index
                    scales[workload, tenant] = scale
                    self.write_run(
                        reference,
                        f"{workload}_unloaded_{tenant}",
                        "MTP_LPT",
                        tenant,
                        scale,
                    )
                    for arm in ("MTP_LPT", "MTPMIX_SPLIT"):
                        self.write_run(root, f"{workload}_r4", arm, tenant, 100.0)
            result = self.analyze(root, "--slo-relative", "5,2", "--ref-root", reference)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = list(csv.DictReader(io.StringIO(result.stdout)))
            self.assertEqual(len(rows), 12)
            for row in rows:
                scale = scales[row["workload"], row["stream"]]
                self.assertAlmostEqual(float(row["slo_ttft_ms"]), 1000.0 * scale)
                self.assertAlmostEqual(float(row["slo_tpot_ms"]), 60.0 * scale)

    def test_underscored_workload_name_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            reference = Path(temporary) / "unloaded"
            self.write_run(root, "rag_agent_128_r4", "MTPMIX_SPLIT", "chat")
            self.write_run(reference, "rag_agent_128_unloaded_chat", "MTP_LPT", "chat")
            result = self.analyze(root, "--slo-relative", "5,2", "--ref-root", reference)
            self.assertEqual(result.returncode, 0, result.stderr)
            row = next(csv.DictReader(io.StringIO(result.stdout)))
            self.assertEqual(row["workload"], "rag_agent_128")

    def test_legacy_single_suffix_reference_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            reference = Path(temporary) / "unloaded"
            self.write_run(root, "mix32_r4", "MTPMIX_SPLIT", "chat")
            self.write_run(reference, "mix32_unloaded", "MTP_LPT", "chat")
            result = self.analyze(root, "--slo-relative", "5,2", "--ref-root", reference)
            self.assertEqual(result.returncode, 0, result.stderr)
            row = next(csv.DictReader(io.StringIO(result.stdout)))
            self.assertEqual(float(row["slo_ttft_ms"]), 1000.0)

    def test_missing_unloaded_reference_is_not_replaced_by_loaded_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            reference = Path(temporary) / "unloaded"
            reference.mkdir()
            self.write_run(root, "mix32_r4", "MTP_LPT", "chat")
            result = self.analyze(root, "--slo-relative", "5,2", "--ref-root", reference)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no unloaded reference", result.stderr)

    def test_absolute_slo_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_run(root, "mix32_r4", "MTPMIX_SPLIT", "chat")
            result = self.analyze(root, "--slo", '{"chat":[2000,50]}')
            self.assertEqual(result.returncode, 0, result.stderr)
            row = next(csv.DictReader(io.StringIO(result.stdout)))
            self.assertEqual(float(row["slo_ttft_ms"]), 2000.0)
            self.assertEqual(float(row["slo_tpot_ms"]), 50.0)


if __name__ == "__main__":
    unittest.main()
