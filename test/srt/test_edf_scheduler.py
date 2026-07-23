import time
import unittest
from unittest.mock import patch

from sglang.srt.managers import edf_scheduler


class _Req:
    def __init__(self, elapsed_ms: float):
        self._edf_last_step_time = time.monotonic() - elapsed_ms / 1000


class TestEDFChunkControl(unittest.TestCase):
    def _compute(self, reqs, predictor, default=8192):
        with patch.object(edf_scheduler, "EDF_ENABLED", True):
            return edf_scheduler.edf_compute_max_chunk_size(
                reqs, default, predictor, decode_avg_kv=2048
            )

    def test_uses_decode_only_when_decode_is_already_late(self):
        with patch.object(edf_scheduler, "ALLOW_DECODE_ONLY", True):
            chunk = self._compute([_Req(edf_scheduler.ITL_SLO_MS + 1)], lambda *_: 1)
        self.assertEqual(chunk, 0)

    def test_uses_decode_only_when_minimum_chunk_does_not_fit(self):
        # Pure decode fits, but every non-empty prefill chunk exceeds the slack.
        def predictor(n_p, *_):
            return 5 if n_p == 0 else 200

        with patch.object(edf_scheduler, "ALLOW_DECODE_ONLY", True):
            chunk = self._compute([_Req(0)], predictor)
        self.assertEqual(chunk, 0)

    def test_legacy_minimum_chunk_is_available_for_ablation(self):
        with patch.object(edf_scheduler, "ALLOW_DECODE_ONLY", False):
            chunk = self._compute([_Req(edf_scheduler.ITL_SLO_MS + 1)], lambda *_: 1)
        self.assertEqual(chunk, edf_scheduler.MIN_PREFILL_CHUNK_SIZE)

    def test_returns_largest_feasible_chunk(self):
        with patch.object(edf_scheduler, "ALLOW_DECODE_ONLY", True):
            chunk = self._compute([_Req(0)], lambda n_p, *_: 5 + n_p / 100)
        self.assertEqual(chunk, 8192)


if __name__ == "__main__":
    unittest.main()
