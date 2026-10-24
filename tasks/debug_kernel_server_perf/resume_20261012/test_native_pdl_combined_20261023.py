import itertools
import json
import math
from types import SimpleNamespace
import unittest

import torch

from native_fa3_fa2_pdl_20261023 import NativePdlExecutor
from test_native_q32 import assert_numerical, reference


def fixture(queries, lengths, heads=28, kv_heads=4, sm_budget=None, prefix_splits=1, row_split_size=None, packed_query_limit=32):
    physical_ids = torch.randperm(sum(lengths), dtype=torch.int64, device="cuda").to(torch.int32)
    query = torch.randn(sum(queries), heads, 128, dtype=torch.bfloat16, device="cuda")
    key = torch.randn(sum(lengths), 1, kv_heads, 128, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    case = SimpleNamespace(
        query=query, key=key, value=value, output=torch.empty_like(query),
        heads=heads, kv_heads=kv_heads, query_lengths=queries, kv_lengths=lengths,
        page_counts=lengths, page_indices=physical_ids,
    )
    pages = torch.zeros((len(lengths), max(lengths)), dtype=torch.int32, device="cuda")
    offset = 0
    for index, length in enumerate(lengths):
        pages[index, :length] = case.page_indices[offset:offset + length]
        offset += length
    cache_lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    cumulative_query = torch.tensor([0, *itertools.accumulate(queries)], dtype=torch.int32, device="cuda")
    executor = NativePdlExecutor(
        heads, kv_heads, torch.device("cuda"), sm_budget=sm_budget,
        prefix_splits=prefix_splits, row_split_size=row_split_size,
        packed_query_limit=packed_query_limit,
    )
    plan = executor.plan(queries, [length - query for length, query in zip(lengths, queries)], pages, cache_lengths, cumulative_query)
    if plan is None:
        raise AssertionError("Fixture did not activate native PDL")
    return case, plan


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestNativePdlCombined(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = False

    def test_varlen_and_sm_budgets(self):
        torch.manual_seed(20261023)
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        for heads, kv_heads, tail in ((28, 4, 4), (32, 8, 4), (24, 2, 1)):
            for margin in (0, 16, 32):
                with self.subTest(heads=heads, margin=margin):
                    case, plan = fixture([128, 17, 1, tail], [128, 513, 65, 8192], heads, kv_heads, sms - margin)
                    expected = reference(case)
                    for pdl in (False, True):
                        case.output.fill_(float("nan"))
                        plan.run(case.query, case.key, case.value, 1 / math.sqrt(128), case.output, pdl=pdl)
                        torch.cuda.synchronize()
                        assert_numerical(case.output, expected)

    def test_changed_inputs_graph_and_full_output_consumer(self):
        torch.manual_seed(271828)
        case, plan = fixture([512, 4, 3], [4096, 65536, 8192])
        projection = torch.randn(case.heads * 128, 64, dtype=torch.bfloat16, device="cuda")
        for warmup in range(3):
            plan.run(case.query, case.key, case.value, 1 / math.sqrt(128), case.output)
            case.output.flatten(1) @ projection
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan.run(case.query, case.key, case.value, 1 / math.sqrt(128), case.output)
            projected = case.output.flatten(1) @ projection
        for iteration in range(6):
            case.query.normal_()
            case.key.normal_()
            case.value.normal_()
            case.output.fill_(float("nan"))
            plan.executor.workspace.view(torch.float32).fill_(float("nan"))
            graph.replay()
            snapshot, consumer = case.output.clone(), projected.clone()
            torch.cuda.synchronize()
            assert_numerical(snapshot, reference(case))
            torch.testing.assert_close(consumer, snapshot.flatten(1) @ projection, rtol=0, atol=0)
            plan.run(case.query, case.key, case.value, 1 / math.sqrt(128), case.output, pdl=False)
            torch.cuda.synchronize()
            torch.testing.assert_close(case.output, snapshot, rtol=0, atol=0)

    def test_long_context_timeline(self):
        torch.manual_seed(314159)
        for chunk, prefix, tail in ((128, 0, 1), (128, 0, 4), (512, 32768, 4)):
            case, plan = fixture([chunk, tail, tail], [prefix + chunk, 393216, 393216])
            expected = reference(case)
            timeline = torch.tensor([2**63 - 1, 0, 2**63 - 1, 2**63 - 1], dtype=torch.int64, device="cuda")
            plan.run(case.query, case.key, case.value, 1 / math.sqrt(128), case.output)
            torch.cuda.synchronize()
            case.output.fill_(float("nan"))
            plan.run(case.query, case.key, case.value, 1 / math.sqrt(128), case.output, timeline=timeline)
            torch.cuda.synchronize()
            worst = assert_numerical(case.output, expected)
            stamps = timeline.tolist()
            self.assertTrue(all(0 < stamp < 2**63 - 1 for stamp in stamps), stamps)
            self.assertLess(stamps[0], stamps[1])
            self.assertLess(stamps[2], stamps[3])
            print("NATIVE_PDL_TIMELINE", json.dumps(dict(
                prefill_query=chunk, prefill_prefix=prefix, row_query=tail, stamps_ns=stamps,
                fa2_starts_before_fa3_end=stamps[2] < stamps[1],
                worst_head_relative_error=worst,
            )), flush=True)

    def test_split_plans_and_consumer(self):
        torch.manual_seed(20261024)
        for prefix_splits in (1, 4, 8):
            for row_split_size in (None, 4096):
                case, plan = fixture(
                    [79, 1, 1, 1, 1], [60352, 48, 50065, 32040, 35770],
                    prefix_splits=prefix_splits, row_split_size=row_split_size,
                )
                expected = reference(case)
                projection = torch.randn(case.heads * 128, 32, dtype=torch.bfloat16, device="cuda")
                plan.run(case.query, case.key, case.value, 128**-0.5, case.output)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    plan.run(case.query, case.key, case.value, 128**-0.5, case.output)
                    projected = case.output.flatten(1) @ projection
                for iteration in range(3):
                    case.output.fill_(float("nan"))
                    plan.shape.partial_output.fill_(float("nan"))
                    graph.replay()
                    snapshot, consumer = case.output.clone(), projected.clone()
                    torch.cuda.synchronize()
                    assert_numerical(snapshot, expected)
                    torch.testing.assert_close(consumer, snapshot.flatten(1) @ projection, rtol=0, atol=0)


    def test_geometry_policy_replanning(self):
        torch.manual_seed(20261025)
        executor = NativePdlExecutor(28, 4, torch.device("cuda"), adaptive_splits=True, workspace_bytes=512 << 20)
        for queries, lengths in (
            ([79, 4, 4], [60352, 65536, 2048]),
            ([2048, 4, 4], [6144, 393216, 2048]),
            ([128, 1, 1], [128, 16384, 4096]),
        ):
            case, template = fixture(queries, lengths)
            plan = executor.plan(
                queries, [length - query for length, query in zip(lengths, queries)],
                template.shape.page_table, template.shape.cache_lengths, template.shape.cumulative_query,
            )
            expected = reference(case)
            plan.run(case.query, case.key, case.value, 128**-0.5, case.output, pdl=False)
            serial = case.output.clone()
            case.output.fill_(float("nan"))
            plan.run(case.query, case.key, case.value, 128**-0.5, case.output)
            torch.cuda.synchronize()
            assert_numerical(case.output, expected)
            torch.testing.assert_close(case.output, serial, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
