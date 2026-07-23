import time
import unittest
from unittest.mock import patch

import numpy as np

from sglang.srt.managers import edf_scheduler


class _Req:
    def __init__(self, elapsed_ms: float):
        self._edf_last_step_time = time.monotonic() - elapsed_ms / 1000


class _PriorityReq:
    def __init__(self, priority=None):
        self.priority = priority
        self.last_tic = time.monotonic() - 1


class TestEDFChunkControl(unittest.TestCase):
    def _compute(self, reqs, predictor, default=8192):
        with patch.object(edf_scheduler, "EDF_ENABLED", True):
            return edf_scheduler.edf_compute_max_chunk_size(
                reqs, default, predictor, decode_avg_kv=2048
            )

    def test_uses_decode_only_when_decode_is_already_late(self):
        chunk = self._compute([_Req(edf_scheduler.ITL_SLO_MS + 1)], lambda *_: 1)
        self.assertEqual(chunk, 0)

    def test_uses_decode_only_when_minimum_chunk_does_not_fit(self):
        # Pure decode fits, but every non-empty prefill chunk exceeds the slack.
        def predictor(n_p, *_):
            return 5 if n_p == 0 else 200

        chunk = self._compute([_Req(0)], predictor)
        self.assertEqual(chunk, 0)

    def test_returns_feasible_chunk_below_legacy_minimum(self):
        chunk = self._compute([_Req(0)], lambda n_p, *_: 50 + n_p / 10)
        self.assertGreaterEqual(chunk, 490)
        self.assertLess(chunk, 512)

    def test_returns_largest_feasible_chunk(self):
        chunk = self._compute([_Req(0)], lambda n_p, *_: 5 + n_p / 100)
        self.assertEqual(chunk, 8192)

    def test_passes_continued_prefill_prefix_to_model(self):
        seen_inputs = []

        def predictor(n_p, r_p, *_):
            seen_inputs.append((n_p, r_p))
            return 5 + n_p / 100

        with patch.object(edf_scheduler, "EDF_ENABLED", True):
            edf_scheduler.edf_compute_max_chunk_size(
                [_Req(0)],
                8192,
                predictor,
                decode_avg_kv=2048,
                prefill_prefix_len=2048,
            )

        self.assertTrue(seen_inputs)
        self.assertTrue(all(r_p == 2048 for n_p, r_p in seen_inputs if n_p > 0))

    def test_clamps_prefill_prefix_to_profiled_domain(self):
        seen_inputs = []

        def predictor(n_p, r_p, *_):
            seen_inputs.append((n_p, r_p))
            return 5 + n_p / 100

        with (
            patch.object(edf_scheduler, "EDF_ENABLED", True),
            patch.object(edf_scheduler, "MAX_PROFILED_PREFILL_PREFIX", 4096),
        ):
            edf_scheduler.edf_compute_max_chunk_size(
                [_Req(0)],
                8192,
                predictor,
                decode_avg_kv=2048,
                prefill_prefix_len=16384,
            )

        self.assertTrue(all(r_p == 4096 for n_p, r_p in seen_inputs if n_p > 0))

    def test_projects_only_extreme_prefix_attention_work(self):
        with (
            patch.object(edf_scheduler, "MAX_PROFILED_PREFILL_PREFIX", 4096),
            patch.object(edf_scheduler, "LONG_PREFIX_PROJECTION_THRESHOLD", 12288),
        ):
            ordinary_n, ordinary_r = edf_scheduler.project_extreme_prefix_work(
                2048, 12000
            )
            extreme_n, extreme_r = edf_scheduler.project_extreme_prefix_work(
                2048, 16384
            )

        self.assertEqual((ordinary_n, ordinary_r), (2048, 4096))
        self.assertEqual(extreme_r, 4096)
        original_work = 2048 * 16384 + 2048**2 / 2
        projected_work = extreme_n * extreme_r + extreme_n**2 / 2
        self.assertAlmostEqual(projected_work, original_work, places=5)

    def test_numpy_predictor_matches_checkpoint_model(self):
        predictor = edf_scheduler.LatencyPredictor(
            "tasks/perf_model/lpt_model.pt"
        )
        inputs = [
            (0, 0, 100, 1),
            (4096, 4096, 10000, 200),
            (1173, 3000, 4000, 80),
        ]

        import torch

        for values in inputs:
            x = np.asarray([*values, 0.0], dtype=np.float32)
            x_norm = (x - predictor.X_mean) / predictor.X_std
            with torch.no_grad():
                expected = float(
                    torch.expm1(
                        predictor.model(torch.from_numpy(x_norm).unsqueeze(0))
                    ).item()
                )
            self.assertAlmostEqual(predictor.predict(*values), expected, places=3)

    def test_retracted_request_preserves_earlier_deadline(self):
        req = _PriorityReq(priority=123.0)
        with patch.object(edf_scheduler, "EDF_ENABLED", True):
            edf_scheduler.edf_set_retracted_priority(req)
        self.assertEqual(req.priority, 123.0)

    def test_retracted_request_without_priority_uses_itl_deadline(self):
        req = _PriorityReq()
        with patch.object(edf_scheduler, "EDF_ENABLED", True):
            edf_scheduler.edf_set_retracted_priority(req)
        self.assertAlmostEqual(
            req.priority,
            req.last_tic + edf_scheduler.ITL_SLO_MS / 1000,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
