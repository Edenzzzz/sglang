import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from native_pdl_model_shadow_20261023 import check_committed_sample, compare_logits, decode_acceptance, prefill_trace, probability_metrics, reference_forwards


class TestGreedyComparison(unittest.TestCase):
    def compare(self, reference, actual):
        batch = SimpleNamespace(mixed_verify_tokens=1, extend_seq_lens_cpu=[1], extend_prefix_lens_cpu=[10])
        result = SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=actual))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "shadow.jsonl"
            with patch.dict(os.environ, {"NATIVE_PDL_MODEL_SHADOW": str(output)}):
                compare_logits([reference, reference.clone(), actual.clone()], result, batch, 0)
            return json.loads(output.read_text())

    def test_topk_tie_is_not_a_greedy_flip(self):
        reference = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        actual = torch.tensor([[1.1, 1.0, 0.0, 0.0]])
        report = self.compare(reference, actual)
        self.assertEqual(report["argmax_flips"], 0)
        self.assertEqual(report["flipped_rows"], [])

    def test_real_greedy_flip(self):
        reference = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        actual = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        report = self.compare(reference, actual)
        self.assertEqual(report["argmax_flips"], 1)
        self.assertEqual(report["flipped_rows"][0]["reference_argmax"], 0)
        self.assertEqual(report["flipped_rows"][0]["candidate_argmax"], 1)

    def test_probability_metrics_ignore_constant_logit_shift(self):
        reference = torch.tensor([[1.0, 2.0, 3.0, 0.0]])
        metrics = probability_metrics(reference, reference + 10)
        self.assertAlmostEqual(metrics["max_kl_ref_to_candidate"], 0.0)
        self.assertAlmostEqual(metrics["max_total_variation"], 0.0)
        self.assertAlmostEqual(metrics["worst_centered_relative"], 0.0)

    def test_probability_metrics_detect_changed_distribution(self):
        metrics = probability_metrics(torch.tensor([[5.0, 0.0]]), torch.tensor([[0.0, 5.0]]))
        self.assertGreater(metrics["max_kl_ref_to_candidate"], 4)
        self.assertGreater(metrics["max_total_variation"], 0.9)

    def test_accepted_positions_exclude_padding(self):
        trace = decode_acceptance([10, 11, 12, 13, 20, 21, 22, 23], [[0, 1, -1, -1], [4, -1, -1, -1]], [2, 1])
        self.assertEqual(trace["tokens"], [[10, 11], [20]])
        self.assertEqual(trace["indices"], [[0, 1], [4]])
        with self.assertRaises(ValueError):
            decode_acceptance([10], [[-1]], [1])

    def test_prefill_probe_isolates_logits_and_sampling_snapshot(self):
        original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        prototype = SimpleNamespace(next_token_logits=original, hidden_states=original, auxiliary_device_output=None)
        info = SimpleNamespace(is_all_greedy=True, penalizer_orchestrator=None, grammar_mask="unchanged")
        calls = []

        def preprocess(output, snapshot):
            snapshot.grammar_mask = None
            output.next_token_logits[:, 0] += 5
            calls.append("preprocess")

        def sample(output, snapshot, *arguments):
            calls.append("sample")
            return output.next_token_logits.argmax(dim=-1)

        runner = SimpleNamespace(sampling_observer=None, _preprocess_logits=preprocess, sampler=sample)
        batch = SimpleNamespace(return_logprob=False, top_logprobs_nums=None, token_ids_logprobs=None, seq_lens=torch.tensor([10, 20]))
        result = prefill_trace(original, prototype, batch, 1, SimpleNamespace(model_runner=runner), info)
        self.assertEqual(result, [0])
        self.assertEqual(calls, ["preprocess", "sample"])
        self.assertEqual(info.grammar_mask, "unchanged")
        torch.testing.assert_close(original, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_committed_verify_and_prefill_are_checked(self):
        report = dict(
            greedy_sampling={"native": {"tokens": [[11]], "indices": [[0]], "accept_lens": [1]}},
            prefill_sampling={"native": [21]},
        )
        actual = SimpleNamespace(next_token_ids=torch.tensor([11, 12]), accept_index=torch.tensor([[0, -1]]), accept_lens=torch.tensor([1]))
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"NATIVE_PDL_MODEL_SHADOW": str(Path(directory) / "results.jsonl")}):
                check_committed_sample(report, actual, torch.tensor([21]))
                self.assertTrue(report["committed_sample_exact"])
                self.assertTrue(report["committed_prefill_exact"])
                with self.assertRaises(RuntimeError):
                    check_committed_sample(report, actual, torch.tensor([22]))

    def test_bf16_control_is_isolated_and_flags_restored(self):
        calls = []

        def forward(**arguments):
            calls.append((os.environ.get("SGLANG_NATIVE_PDL"), os.environ.get("SGLANG_NATIVE_PDL_FP32_PARTIALS")))
            return SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=torch.tensor([[float(len(calls))]])))

        environment = {
            "NATIVE_PDL_MODEL_SHADOW": "enabled", "SGLANG_NATIVE_PDL": "1",
            "SGLANG_NATIVE_PDL_FP32_PARTIALS": "1", "NATIVE_PDL_BF16_PARTIAL_CONTROL": "1",
            "NATIVE_PDL_ORIGINAL_ROW_CONTROL": "0",
        }
        batch = SimpleNamespace(forward_metadata_ready=False, mixed_verify_tokens=4)
        worker = SimpleNamespace(forward_batch_generation=forward)
        with patch.dict(os.environ, environment), patch("torch.cuda.is_current_stream_capturing", return_value=False):
            references = reference_forwards(worker, batch, None)
            self.assertEqual(os.environ["SGLANG_NATIVE_PDL_FP32_PARTIALS"], "1")
            self.assertEqual(os.environ["SGLANG_NATIVE_PDL"], "1")
        self.assertEqual(calls, [("0", "1"), ("0", "1"), ("1", "0"), ("1", "1")])
        self.assertIsNone(references[3])
        self.assertEqual(float(references[4].item()), 3)
        self.assertEqual(float(references[2].item()), 4)

    def test_separate_conversion_control_restores_direct_candidate(self):
        calls = []

        def forward(**arguments):
            calls.append((os.environ.get("SGLANG_NATIVE_PDL"), os.environ.get("SGLANG_NATIVE_PDL_DIRECT_BF16_OUTPUT")))
            return SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=torch.zeros(1, 2)))

        environment = {
            "NATIVE_PDL_MODEL_SHADOW": "enabled", "SGLANG_NATIVE_PDL": "1",
            "SGLANG_NATIVE_PDL_FP32_PARTIALS": "1", "SGLANG_NATIVE_PDL_DIRECT_BF16_OUTPUT": "1",
            "NATIVE_PDL_BF16_PARTIAL_CONTROL": "0", "NATIVE_PDL_ORIGINAL_ROW_CONTROL": "0",
            "NATIVE_PDL_SHADOW_SERIAL_SEPARATE_CONVERSION": "1",
        }
        batch = SimpleNamespace(forward_metadata_ready=False, mixed_verify_tokens=4)
        with patch.dict(os.environ, environment), patch("torch.cuda.is_current_stream_capturing", return_value=False):
            references = reference_forwards(SimpleNamespace(forward_batch_generation=forward), batch, None)
            self.assertEqual(os.environ["SGLANG_NATIVE_PDL_DIRECT_BF16_OUTPUT"], "1")
        self.assertEqual(len(references), 3)
        self.assertEqual(calls, [("0", "1"), ("0", "1"), ("1", "0")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
