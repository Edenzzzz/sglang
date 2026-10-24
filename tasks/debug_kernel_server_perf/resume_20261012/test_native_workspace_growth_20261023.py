import itertools
import json
import os
from types import SimpleNamespace
import unittest

from native_fa3_fa2_pdl_20261023 import NativePdlExecutor, float_workspace_overflow, grown_workspace_size


class TestGrowthPolicy(unittest.TestCase):
    def test_rounding_and_ceiling(self):
        self.assertEqual(grown_workspace_size(512 << 20, 710148096, 4 << 30), 704 << 20)
        self.assertEqual(grown_workspace_size(100, 150, 200), 200)
        with self.assertRaisesRegex(ValueError, "ceiling"):
            grown_workspace_size(512 << 20, 710148096, 640 << 20)
        with self.assertRaises(ValueError):
            grown_workspace_size(512, 512, 1024)

    def test_unrelated_errors_cannot_trigger_a_retry(self):
        self.assertTrue(float_workspace_overflow(RuntimeError("Error in function 'aligned_alloc' at allocator.h: Buffer overflow when allocating memory for batch_prefill_tmp_v")))
        self.assertFalse(float_workspace_overflow(RuntimeError("CUDA out of memory")))
        self.assertFalse(float_workspace_overflow(RuntimeError("Error in function 'aligned_alloc': Buffer overflow when allocating memory for batch_prefill_request_indices")))
        executor = NativePdlExecutor.__new__(NativePdlExecutor)
        calls = []

        def fail(*arguments, **options):
            calls.append(True)
            raise RuntimeError("invalid page metadata")

        executor.wrapper = SimpleNamespace(plan=fail)
        with self.assertRaisesRegex(RuntimeError, "invalid page metadata"):
            executor._plan_rows((), {})
        self.assertEqual(len(calls), 1)


@unittest.skipUnless(os.environ.get("NATIVE_WORKSPACE_GPU_TEST") == "1", "requires CUDA workspace stress gate")
class TestWorkspaceGrowth(unittest.TestCase):
    def test_large_shared_prefix_plans_and_graph_outputs(self):
        import torch

        from fp32_attention_adjudication_20261023 import causal_fp32

        torch.manual_seed(20261031)
        torch.backends.cuda.matmul.allow_tf32 = False
        for rows, context in ((24, 65536), (47, 524288)):
            with self.subTest(rows=rows, context=context):
                queries = [128] + [4] * rows
                lengths = [128] + [context] * rows
                prefixes = [length - query for length, query in zip(lengths, queries)]
                shared = context - 4
                physical_pages = shared + rows * 4 + 128
                pages = torch.arange(context, device="cuda", dtype=torch.int32).expand(rows + 1, -1).clone()
                pages[0, :128] = torch.arange(shared + rows * 4, physical_pages, device="cuda", dtype=torch.int32)
                for request in range(rows):
                    pages[request + 1, shared:] = torch.arange(shared + request * 4, shared + (request + 1) * 4, device="cuda", dtype=torch.int32)
                cache_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
                cumulative = torch.tensor([0, *itertools.accumulate(queries)], device="cuda", dtype=torch.int32)
                query = torch.randn(sum(queries), 28, 128, device="cuda", dtype=torch.bfloat16)
                key = torch.randn(physical_pages, 1, 4, 128, device="cuda", dtype=torch.bfloat16)
                value = torch.randn_like(key)
                output = torch.empty_like(query)
                executor = NativePdlExecutor(28, 4, query.device, adaptive_splits=True, policy_version="geometry_v3", fp32_partials=True, direct_bf16_output=True, workspace_bytes=512 << 20, max_workspace_bytes=4 << 30)
                plan = executor.plan(queries, prefixes, pages, cache_lengths, cumulative)
                self.assertEqual(executor.workspace_grows, 1)
                self.assertGreater(executor.workspace.numel(), 512 << 20)
                reference_executor = NativePdlExecutor(28, 4, query.device, adaptive_splits=True, policy_version="geometry_v3", fp32_partials=True, direct_bf16_output=True, workspace_bytes=executor.workspace.numel())
                reference_plan = reference_executor.plan(queries, prefixes, pages, cache_lengths, cumulative)
                self.assertEqual(list(executor.wrapper._plan_info), list(reference_executor.wrapper._plan_info))
                expected = causal_fp32(query, key, value, pages, queries, lengths, 128**-0.5)
                for warmup in range(3):
                    plan.run(query, key, value, 128**-0.5, output)
                torch.cuda.synchronize()
                projection = torch.randn(28 * 128, 16, device="cuda", dtype=torch.bfloat16)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    plan.run(query, key, value, 128**-0.5, output)
                    consumed = output.flatten(1) @ projection
                for iteration in range(3):
                    if iteration:
                        query.normal_()
                        key.normal_()
                        value.normal_()
                        expected = causal_fp32(query, key, value, pages, queries, lengths, 128**-0.5)
                    output.fill_(float("nan"))
                    executor.workspace.view(torch.float32).fill_(float("nan"))
                    graph.replay()
                    snapshot, projected = output.clone(), consumed.clone()
                    relative = (snapshot.float() - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-6)
                    self.assertLess(float(relative.max()), 0.02)
                    torch.testing.assert_close(projected, snapshot.flatten(1) @ projection, rtol=0, atol=0)
                    reference_plan.run(query, key, value, 128**-0.5, output, pdl=False)
                    torch.testing.assert_close(output, snapshot, rtol=0, atol=0)
                previous_capacity = executor.workspace.numel()
                smaller = executor.plan([128, 4], [0, 1020], pages[:2], torch.tensor([128, 1024], device="cuda", dtype=torch.int32), torch.tensor([0, 128, 132], device="cuda", dtype=torch.int32))
                smaller.run(query[:132], key, value, 128**-0.5, output[:132])
                torch.cuda.synchronize()
                smaller_expected = causal_fp32(query[:132], key, value, pages[:2], [128, 4], [128, 1024], 128**-0.5)
                smaller_relative = (output[:132].float() - smaller_expected).norm(dim=-1) / smaller_expected.norm(dim=-1).clamp_min(1e-6)
                self.assertLess(float(smaller_relative.max()), 0.02)
                self.assertEqual(executor.workspace.numel(), previous_capacity)
                self.assertEqual(executor.workspace_grows, 1)
                print("WORKSPACE_GROWTH_GATE", json.dumps(dict(rows=rows, context=context, workspace_bytes=previous_capacity, serial_fixed_workspace_exact=True, scope="Synthetic shared-prefix capacity/correctness gate, not a serving performance claim")), flush=True)
                del executor, reference_executor, plan, reference_plan, smaller, graph, query, key, value, output, expected, smaller_expected, smaller_relative, snapshot, projected, consumed, projection, pages, cache_lengths, cumulative
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main(verbosity=2)
