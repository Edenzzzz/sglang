import unittest

import torch

from fp32_attention_adjudication_20261023 import causal_fp32
from native_fa3_fa2_pdl_20261023 import NativePdlExecutor
from test_native_pdl_combined_20261023 import fixture


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestNativeQ64Rows(unittest.TestCase):
    def test_glm_verify_and_qwen_eight_token_rows(self):
        torch.manual_seed(20261029)
        torch.backends.cuda.matmul.allow_tf32 = False
        for heads, kv_heads, row_query in ((24, 2, 4), (28, 4, 8), (32, 8, 16)):
            for prefix_splits in (1, 8):
                with self.subTest(heads=heads, row_query=row_query, prefix_splits=prefix_splits):
                    case, template = fixture([79, row_query, row_query], [8192, 131072, 16384], heads, kv_heads, packed_query_limit=64)
                    executor = NativePdlExecutor(heads, kv_heads, case.query.device, fp32_partials=True, direct_bf16_output=True, prefix_splits=prefix_splits, row_split_size=4096, workspace_bytes=512 << 20, packed_query_limit=64)
                    plan = executor.plan(case.query_lengths, [length - query for query, length in zip(case.query_lengths, case.kv_lengths)], template.shape.page_table, template.shape.cache_lengths, template.shape.cumulative_query)
                    self.assertEqual(plan.shape.native_tile, 0)
                    self.assertEqual(executor.wrapper._plan_info[3], 64)
                    projection = torch.randn(heads * 128, 32, dtype=torch.bfloat16, device="cuda")
                    for warmup in range(3):
                        plan.run(case.query, case.key, case.value, 128**-0.5, case.output)
                        case.output.flatten(1) @ projection
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        plan.run(case.query, case.key, case.value, 128**-0.5, case.output)
                        projected = case.output.flatten(1) @ projection
                    for iteration in range(3):
                        case.query.normal_()
                        case.key.normal_()
                        case.value.normal_()
                        executor.workspace.view(torch.float32).fill_(float("nan"))
                        case.output.fill_(float("nan"))
                        graph.replay()
                        snapshot, consumer = case.output.clone(), projected.clone()
                        expected = causal_fp32(case.query, case.key, case.value, plan.shape.page_table, case.query_lengths, case.kv_lengths, 128**-0.5)
                        relative = (snapshot.float() - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-6)
                        self.assertLess(float(relative.max()), 0.02)
                        torch.testing.assert_close(consumer, snapshot.flatten(1) @ projection, rtol=0, atol=0)
                        plan.run(case.query, case.key, case.value, 128**-0.5, case.output, pdl=False)
                        torch.testing.assert_close(case.output, snapshot, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
