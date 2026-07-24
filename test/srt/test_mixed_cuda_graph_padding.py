import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner


class TestMixedCudaGraphPadding(unittest.TestCase):
    def setUp(self):
        self.runner = CudaGraphRunner.__new__(CudaGraphRunner)
        self.runner.mixed_graphs = {1: object()}
        self.runner.disable_padding = False
        self.runner.mixed_capture_bs = [2048, 2304, 4096]
        self.runner.mixed_max_bs = 4096
        self.runner._mixed_cg_max_padding_tokens = 4
        self.runner._mixed_cg_padding_pct = 0

    @staticmethod
    def _batch(num_tokens):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_mixed=lambda: True),
            input_ids=torch.empty(num_tokens),
        )

    def test_gate_uses_actual_spare_replay_shape(self):
        # An exact-size replay advances to the next graph for a safety slot.
        # The gate must account for all 256 padded full-model tokens.
        self.assertFalse(self.runner.can_run(self._batch(2048)))

    def test_small_padding_is_allowed(self):
        self.assertTrue(self.runner.can_run(self._batch(2045)))

    def test_largest_exact_graph_falls_back(self):
        # There is no larger safe graph for the required spare slot.
        self.assertFalse(self.runner.can_run(self._batch(4096)))


if __name__ == "__main__":
    unittest.main()
