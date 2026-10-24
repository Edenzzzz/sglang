import os
from pathlib import Path
import unittest

from load_native_fp32_partials_20261023 import direct_output_bridge, direct_output_prefill, transform_bridge


class TestDirectSource(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("NATIVE_Q32_SOURCE_ROOT"), "requires native source")
    def test_partial_dtype_unchanged_and_unsplit_rejected(self):
        root = Path(os.environ["NATIVE_Q32_SOURCE_ROOT"])
        result = direct_output_bridge(transform_bridge((root / "native_prefill_bridge.cuh").read_text()))
        self.assertIn("using DTypeO = float;", result)
        self.assertIn("GetPtrFromBaseOffset<float>(float_buffer, plan.v_offset)", result)
        self.assertIn("TORCH_CHECK(plan.split_kv", result)
        self.assertIn("params.final_bf16_output = static_cast<__nv_bfloat16*>", result)
        source = (root / "port07/include/flashinfer/attention/prefill.cuh").read_text()
        changed = direct_output_prefill(source)
        self.assertEqual(changed.count("auto o = params.final_bf16_output;"), 1)
        self.assertEqual(changed.replace("auto o = params.final_bf16_output;", "auto o = params.o;"), source)

    def test_reject_unknown_source(self):
        with self.assertRaises(ValueError):
            direct_output_bridge("unknown")
        with self.assertRaises(ValueError):
            direct_output_prefill("unknown")


@unittest.skipUnless(os.environ.get("NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY"), "requires CUDA direct-output library")
class TestDirectOutput(unittest.TestCase):
    def test_split_graph_outputs_and_consumers_exact(self):
        import torch

        from native_fa3_fa2_pdl_20261023 import NativePdlExecutor
        from test_native_pdl_combined_20261023 import fixture

        torch.manual_seed(20261028)
        for heads, kv_heads, row_query in ((28, 4, 4), (32, 8, 4), (24, 2, 1)):
            case, template = fixture([79, row_query, row_query], [8192, 32768, 8192], heads, kv_heads)
            plans = []
            for direct in (False, True):
                executor = NativePdlExecutor(heads, kv_heads, case.query.device, fp32_partials=True, direct_bf16_output=direct, prefix_splits=4, row_split_size=1024, workspace_bytes=512 << 20)
                plans.append(executor.plan(case.query_lengths, [length - query for query, length in zip(case.query_lengths, case.kv_lengths)], template.shape.page_table, template.shape.cache_lengths, template.shape.cumulative_query))
            reference, direct = plans
            self.assertTrue(direct.shape.direct_row_output)
            self.assertIsNone(direct.shape.float_row_output)
            projection = torch.randn(heads * 128, 32, dtype=torch.bfloat16, device="cuda")
            for warmup in range(3):
                direct.run(case.query, case.key, case.value, 128**-0.5, case.output)
                case.output.flatten(1) @ projection
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                direct.run(case.query, case.key, case.value, 128**-0.5, case.output)
                projected = case.output.flatten(1) @ projection
            for iteration in range(4):
                case.query.normal_()
                case.key.normal_()
                case.value.normal_()
                reference.run(case.query, case.key, case.value, 128**-0.5, case.output, pdl=False)
                expected = case.output.clone()
                case.output.fill_(float("nan"))
                direct.executor.workspace.view(torch.float32).fill_(float("nan"))
                graph.replay()
                snapshot, consumer = case.output.clone(), projected.clone()
                torch.testing.assert_close(snapshot, expected, rtol=0, atol=0)
                torch.testing.assert_close(consumer, snapshot.flatten(1) @ projection, rtol=0, atol=0)
                direct.run(case.query, case.key, case.value, 128**-0.5, case.output, pdl=False)
                torch.testing.assert_close(case.output, expected, rtol=0, atol=0)

    def test_unsplit_plan_uses_existing_float_conversion(self):
        import torch

        from native_fa3_fa2_pdl_20261023 import NativePdlExecutor
        from test_native_pdl_combined_20261023 import fixture

        case, template = fixture([79, 4, 4], [79, 4, 4])
        executor = NativePdlExecutor(28, 4, case.query.device, fp32_partials=True, direct_bf16_output=True, row_split_size=1024, workspace_bytes=512 << 20)
        plan = executor.plan(case.query_lengths, [0, 0, 0], template.shape.page_table, template.shape.cache_lengths, template.shape.cumulative_query)
        self.assertFalse(executor.wrapper._plan_info[14])
        self.assertFalse(plan.shape.direct_row_output)
        self.assertIsNotNone(plan.shape.float_row_output)
        plan.run(case.query, case.key, case.value, 128**-0.5, case.output)
        torch.cuda.synchronize()
        self.assertTrue(bool(torch.isfinite(case.output).all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
