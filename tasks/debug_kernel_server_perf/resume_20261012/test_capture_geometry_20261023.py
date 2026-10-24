import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from capture_mixed_attention_20261023 import capture_mixed_attention, verify_tail_rows
from native_fa3_fa2_pdl_20261023 import partition_lengths


class TestCaptureGeometry(unittest.TestCase):
    def test_small_prefill_is_not_a_verify_row(self):
        queries = [511, 3, 4, 4]
        shape = partition_lengths(queries, [0, 10, 32768, 65536], 28, 4)
        self.assertEqual(shape.prefix_rows, 1)
        self.assertEqual(verify_tail_rows(queries, 8), 2)
        self.assertEqual(len(queries) - shape.prefix_rows - verify_tail_rows(queries, 8), 1)

    def test_regular_verify_tail(self):
        self.assertEqual(verify_tail_rows([2047, 4, 4], 8), 2)

    def test_bad_tail(self):
        for queries, tokens in (([511, 3, 4], 8), ([511, 4], 3), ([4, 4], 8)):
            with self.assertRaises(ValueError):
                verify_tail_rows(queries, tokens)

    def test_other_tp_ranks_cannot_write_the_same_capture(self):
        with patch.dict(os.environ, {"SGLANG_CAPTURE_MIXED_ATTN": "/unused", "CAPTURE_LAYER": "14", "CAPTURE_RANK": "0"}), patch("torch.distributed.is_initialized", return_value=True), patch("torch.distributed.get_rank", return_value=1):
            capture_mixed_attention(None, None, None, None, None, SimpleNamespace(layer_id=14), None)

    def test_glm_verify_tail_with_q64_row_class(self):
        queries = [2047, 4, 4]
        shape = partition_lengths(queries, [0, 32768, 65536], 24, 2, 64)
        self.assertEqual(shape.prefix_rows, 1)
        self.assertEqual(verify_tail_rows(queries, 8), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
