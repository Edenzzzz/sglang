import os
from pathlib import Path
import unittest

from load_native_fp32_partials_20261023 import replace_exact, specialize_merge, transform_bridge


class TestBridgeTransform(unittest.TestCase):
    def test_reject_changed_source(self):
        with self.assertRaises(ValueError):
            transform_bridge("unrecognized bridge")
        with self.assertRaises(ValueError):
            replace_exact("duplicate duplicate", "duplicate", "replacement")

    @unittest.skipUnless(os.environ.get("NATIVE_Q32_SOURCE_ROOT"), "requires original bridge source")
    def test_inputs_unchanged(self):
        original = (Path(os.environ["NATIVE_Q32_SOURCE_ROOT"]) / "native_prefill_bridge.cuh").read_text()
        transformed = transform_bridge(original)
        self.assertIn("using DTypeO = float;", transformed)
        self.assertIn("using DTypeQ = __nv_bfloat16;", transformed)
        self.assertIn("using DTypeKV = __nv_bfloat16;", transformed)
        self.assertNotIn("NativePagedParams", transformed)
        self.assertIn("GetPtrFromBaseOffset<float>(float_buffer, plan.v_offset)", transformed)

    def test_merge_specialization_is_scoped(self):
        original = "unchanged\ncudaError_t VariableLengthMergeStates(...) {\n  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {\n    kernel();\n  });\n  return cudaSuccess;\n}\ntemplate <other>\nunchanged"
        result = specialize_merge(original)
        self.assertTrue(result.startswith("unchanged\n"))
        self.assertTrue(result.endswith("template <other>\nunchanged"))
        self.assertIn("if (head_dim != HEAD_DIM) return cudaErrorInvalidValue;", result)
        self.assertNotIn("DISPATCH_HEAD_DIM", result)


@unittest.skipUnless(os.environ.get("NATIVE_PDL_FA2_FP32_LIBRARY"), "requires FP32 native CUDA library")
class TestNativeFloatPartials(unittest.TestCase):
    def test_original_tile_uses_public_plan(self):
        import torch

        from native_fa3_fa2_pdl_20261023 import NativePdlExecutor
        from fp32_attention_adjudication_20261023 import causal_fp32
        from test_native_pdl_combined_20261023 import fixture

        torch.manual_seed(20261026)
        for queries, lengths, split in (
            ([79, 4, 3], [8192, 2048, 4096], None),
            ([79, 4, 3], [8192, 16384, 8192], 1024),
            ([79, 1, 1], [8192, 16384, 8192], 1024),
        ):
            with self.subTest(queries=queries, split=split):
                case, original = fixture(queries, lengths)
                executor = NativePdlExecutor(28, 4, case.query.device, fp32_partials=True, row_split_size=split, workspace_bytes=512 << 20)
                plan = executor.plan(queries, [length - query for query, length in zip(queries, lengths)], original.shape.page_table, original.shape.cache_lengths, original.shape.cumulative_query)
                if split is None:
                    wrapper = executor.wrapper
                    wrapper.plan(
                        wrapper._qo_indptr_buf.cpu(), wrapper._paged_kv_indptr_buf.cpu(),
                        wrapper._paged_kv_indices_buf, wrapper._paged_kv_last_page_len_buf.cpu(),
                        28, 4, 128, 1, causal=True, q_data_type=torch.bfloat16,
                        kv_data_type=torch.bfloat16, disable_split_kv=True,
                    )
                    self.assertFalse(wrapper._plan_info[14])
                plan.run(case.query, case.key, case.value, 128**-0.5, case.output, pdl=False, query_tile=0)
                actual = plan.shape.float_row_output.clone()
                public = executor.wrapper.run(case.query[plan.shape.prefix_tokens:], (case.key, case.value))
                expected = causal_fp32(case.query, case.key, case.value, plan.shape.page_table, queries, lengths, 128**-0.5)[plan.shape.prefix_tokens:]
                torch.cuda.synchronize()
                self.assertEqual(actual.dtype, torch.float32)
                relative = (actual - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-6)
                self.assertLess(float(relative.max()), 0.02)
                if split is None:
                    torch.testing.assert_close(actual.bfloat16(), public, rtol=0, atol=0)

    def test_split_q32_graph_consumer_and_changed_inputs(self):
        import torch

        from fp32_attention_adjudication_20261023 import causal_fp32
        from native_fa3_fa2_pdl_20261023 import NativePdlExecutor
        from test_native_pdl_combined_20261023 import fixture

        torch.manual_seed(20261027)
        torch.backends.cuda.matmul.allow_tf32 = False
        case, original = fixture([79, 4, 3], [8192, 32768, 8192])
        executor = NativePdlExecutor(28, 4, case.query.device, fp32_partials=True, prefix_splits=4, row_split_size=1024, workspace_bytes=512 << 20)
        plan = executor.plan(case.query_lengths, [length - query for query, length in zip(case.query_lengths, case.kv_lengths)], original.shape.page_table, original.shape.cache_lengths, original.shape.cumulative_query)
        self.assertEqual(plan.shape.native_tile, 32)
        projection = torch.randn(28 * 128, 32, dtype=torch.bfloat16, device="cuda")
        for warmup in range(3):
            plan.run(case.query, case.key, case.value, 128**-0.5, case.output)
            case.output.flatten(1) @ projection
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan.run(case.query, case.key, case.value, 128**-0.5, case.output)
            projected = case.output.flatten(1) @ projection
        for iteration in range(4):
            case.query.normal_()
            case.key.normal_()
            case.value.normal_()
            case.output.fill_(float("nan"))
            executor.workspace.view(torch.float32).fill_(float("nan"))
            graph.replay()
            snapshot, consumed = case.output.clone(), projected.clone()
            expected = causal_fp32(case.query, case.key, case.value, plan.shape.page_table, case.query_lengths, case.kv_lengths, 128**-0.5)
            relative = (snapshot.float() - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-6)
            self.assertLess(float(relative.max()), 0.02)
            torch.testing.assert_close(consumed, snapshot.flatten(1) @ projection, rtol=0, atol=0)
            plan.run(case.query, case.key, case.value, 128**-0.5, case.output, pdl=False)
            torch.testing.assert_close(case.output, snapshot, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
