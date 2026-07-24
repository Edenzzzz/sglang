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
        self.runner._mixed_cg_max_padding_tokens = 0
        self.runner._mixed_cg_padding_pct = 0

    @staticmethod
    def _batch(num_tokens):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_mixed=lambda: True),
            input_ids=torch.empty(num_tokens),
        )

    def test_exact_shape_falls_back_without_padding_budget(self):
        self.assertFalse(self.runner.can_run(self._batch(2048)))

    def test_padding_is_disabled_by_default(self):
        self.assertFalse(self.runner.can_run(self._batch(2045)))

    def test_largest_exact_graph_falls_back(self):
        self.assertFalse(self.runner.can_run(self._batch(4096)))

    def test_padding_remains_explicitly_opt_in(self):
        self.runner._mixed_cg_max_padding_tokens = 256
        self.assertTrue(self.runner.can_run(self._batch(2048)))


if __name__ == "__main__":
    unittest.main()
