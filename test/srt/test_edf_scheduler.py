import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sglang.srt.managers import edf_scheduler


class _Req:
    def __init__(self, elapsed_ms: float, max_new_tokens=None):
        self._edf_last_step_time = time.monotonic() - elapsed_ms / 1000
        if max_new_tokens is not None:
            self.sampling_params = SimpleNamespace(max_new_tokens=max_new_tokens)


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

    def test_kv_watermark_only_blocks_at_high_usage(self):
        with patch.object(edf_scheduler, "KV_ADMISSION_WATERMARK", 0.95):
            self.assertTrue(edf_scheduler.edf_allow_prefill_at_kv_usage(0.949))
            self.assertFalse(edf_scheduler.edf_allow_prefill_at_kv_usage(0.95))

    def test_adaptive_plan_overhead_scales_with_output_budget(self):
        with patch.object(edf_scheduler, "ADAPTIVE_PLAN_OVERHEAD", True):
            self.assertEqual(edf_scheduler.edf_plan_overhead_ms([_Req(0, 1024)], 0), 5)
            self.assertEqual(
                edf_scheduler.edf_plan_overhead_ms([_Req(0, 2048)], 0), 7.5
            )
            self.assertEqual(
                edf_scheduler.edf_plan_overhead_ms([_Req(0, 4096)], 0), 10
            )

    def test_adaptive_plan_overhead_protects_extreme_or_unknown_work(self):
        with (
            patch.object(edf_scheduler, "ADAPTIVE_PLAN_OVERHEAD", True),
            patch.object(
                edf_scheduler, "LONG_PREFIX_PROJECTION_THRESHOLD", 14336
            ),
        ):
            self.assertEqual(
                edf_scheduler.edf_plan_overhead_ms([_Req(0, 1024)], 14336), 10
            )
            self.assertEqual(edf_scheduler.edf_plan_overhead_ms([_Req(0)], 0), 10)

    def test_fixed_plan_overhead_remains_compatible(self):
        with (
            patch.object(edf_scheduler, "ADAPTIVE_PLAN_OVERHEAD", False),
            patch.object(edf_scheduler, "PLAN_OVERHEAD_MS", 6.25),
        ):
            self.assertEqual(
                edf_scheduler.edf_plan_overhead_ms([_Req(0, 4096)], 20000), 6.25
            )

    def test_forward_snapshot_uses_admitted_chunk_and_average_lengths(self):
        seen = []

        def predictor(n_p, r_p, r_d, decode_bs):
            seen.append((n_p, r_p, r_d, decode_bs))
            return 42.5

        predicted = edf_scheduler.predict_forward_snapshot_ms(
            predictor,
            num_prefill_tokens=12000,
            prefill_prefix_tokens=6000,
            prefill_request_count=3,
            decode_kv_tokens=24000,
            decode_batch_size=8,
            chunked_prefill_size=4096,
        )

        self.assertEqual(predicted, 42.5)
        self.assertEqual(seen, [(4096.0, 2000.0, 3000.0, 8)])

    def test_forward_snapshot_without_checkpoint_is_explicitly_missing(self):
        self.assertIsNone(
            edf_scheduler.predict_forward_snapshot_ms(
                None,
                num_prefill_tokens=100,
                prefill_prefix_tokens=0,
                prefill_request_count=1,
                decode_kv_tokens=0,
                decode_batch_size=0,
                chunked_prefill_size=4096,
            )
        )

    def test_forward_snapshot_can_report_pure_decode(self):
        seen = []

        def predictor(*features):
            seen.append(features)
            return 17

        predicted = edf_scheduler.predict_forward_snapshot_ms(
            predictor,
            num_prefill_tokens=0,
            prefill_prefix_tokens=0,
            prefill_request_count=0,
            decode_kv_tokens=9000,
            decode_batch_size=3,
            chunked_prefill_size=4096,
        )

        self.assertEqual(predicted, 17.0)
        self.assertEqual(seen, [(0.0, 0.0, 3000.0, 3)])

    def test_forward_snapshot_is_zero_when_server_is_idle(self):
        predicted = edf_scheduler.predict_forward_snapshot_ms(
            lambda *_: 99,
            num_prefill_tokens=0,
            prefill_prefix_tokens=0,
            prefill_request_count=0,
            decode_kv_tokens=0,
            decode_batch_size=0,
            chunked_prefill_size=4096,
        )
        self.assertEqual(predicted, 0.0)

    def test_explicit_checkpoint_loads_without_enabling_edf(self):
        sentinel = object()
        with (
            patch.object(edf_scheduler, "EDF_ENABLED", False),
            patch.object(edf_scheduler, "LATENCY_MODEL_PATH", "/model.pt"),
            patch.object(edf_scheduler.os.path, "exists", return_value=True),
            patch.object(
                edf_scheduler, "LatencyPredictor", return_value=sentinel
            ) as constructor,
        ):
            predictor = edf_scheduler.try_load_latency_predictor()

        self.assertIs(predictor, sentinel)
        constructor.assert_called_once_with("/model.pt")


if __name__ == "__main__":
    unittest.main()
