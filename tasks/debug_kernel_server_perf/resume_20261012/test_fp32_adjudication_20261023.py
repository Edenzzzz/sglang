import math
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fp32_attention_adjudication_20261023 import causal_fp32, golden_and_restore


class TestFp32Reference(unittest.TestCase):
    def test_matches_independent_float64_gqa_reference(self):
        torch.manual_seed(20261023)
        query = torch.randn(5, 6, 8).bfloat16()
        key = torch.randn(14, 1, 2, 8).bfloat16()
        value = torch.randn(28, 1, 2, 8).bfloat16()[::2]
        pages = torch.tensor([[5, 2, 9, 4, 12, 0], [5, 8, 10, 1, 13, 7]])
        queries, lengths = [3, 2], [5, 6]
        expected = torch.empty_like(query, dtype=torch.float64)
        offset = 0
        for request, (query_length, length) in enumerate(zip(queries, lengths)):
            selected_key = key[pages[request, :length], 0].double().repeat_interleave(3, dim=1).transpose(0, 1)
            selected_value = value[pages[request, :length], 0].double().repeat_interleave(3, dim=1).transpose(0, 1)
            selected_query = query[offset:offset + query_length].double().transpose(0, 1)
            logits = selected_query @ selected_key.transpose(1, 2) / math.sqrt(8)
            for position in range(query_length):
                logits[:, position, length - query_length + position + 1:] = -float("inf")
            expected[offset:offset + query_length] = (logits.softmax(-1) @ selected_value).transpose(0, 1)
            offset += query_length
        for chunk in (1, 2, 32):
            actual = causal_fp32(query, key, value, pages, queries, lengths, 8**-0.5, query_chunk=chunk)
            torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-7)
        if torch.cuda.is_available():
            previous = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                actual = causal_fp32(query.cuda(), key.cuda(), value.cuda(), pages.cuda(), queries, lengths, 8**-0.5, query_chunk=2)
                torch.testing.assert_close(actual.cpu().double(), expected, rtol=2e-6, atol=2e-7)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous

    def test_rejects_inconsistent_lengths(self):
        query = torch.zeros(2, 2, 8)
        key = torch.zeros(4, 1, 1, 8)
        pages = torch.tensor([[0, 1, 2, 3]])
        for queries, lengths in (([1], [4]), ([2], [1]), ([], [])):
            with self.assertRaises(ValueError):
                causal_fp32(query, key, key, pages, queries, lengths, 1.0)

    def test_reference_repeats_and_native_state_are_verified(self):
        logits = torch.tensor([[1.0, 2.0]])
        hidden = torch.tensor([[3.0, 4.0]])
        candidate = SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=logits, hidden_states=hidden))
        calls = []

        def restore(**arguments):
            self.assertNotEqual(os.environ.get("SGLANG_NATIVE_PDL_FP32"), "1")
            calls.append("restored")
            return SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=logits.clone(), hidden_states=hidden.clone()))

        worker = SimpleNamespace(forward_batch_generation=restore)
        reference = (logits + 0.01, hidden + 0.01, [0, 1])
        with patch.dict(os.environ, {"SGLANG_NATIVE_PDL": "1", "SGLANG_NATIVE_PDL_FP32": "0"}):
            with patch("fp32_attention_adjudication_20261023.target_reference", side_effect=[reference, reference]):
                golden, status = golden_and_restore(worker, SimpleNamespace(), candidate)
            self.assertEqual(os.environ["SGLANG_NATIVE_PDL_FP32"], "0")
        torch.testing.assert_close(golden, reference[0])
        self.assertEqual(calls, ["restored"])
        self.assertTrue(status["native_hidden_restored_exact"])

    def test_rejects_native_restoration_drift(self):
        logits = torch.tensor([[1.0, 2.0]])
        hidden = torch.tensor([[3.0, 4.0]])
        candidate = SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=logits, hidden_states=hidden))
        restored = SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=logits + 0.1, hidden_states=hidden))
        worker = SimpleNamespace(forward_batch_generation=lambda **arguments: restored)
        reference = (logits.clone(), hidden.clone(), [0, 1])
        with patch.dict(os.environ, {"SGLANG_NATIVE_PDL": "1"}):
            with patch("fp32_attention_adjudication_20261023.target_reference", side_effect=[reference, reference]):
                with self.assertRaisesRegex(RuntimeError, "logits changed"):
                    golden_and_restore(worker, SimpleNamespace(), candidate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
