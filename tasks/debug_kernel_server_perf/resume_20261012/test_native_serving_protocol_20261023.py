import copy
import unittest

from native_serving_protocol_20261023 import assert_same_requests, measured_step_trace, summarize_stream


class TestServingProtocol(unittest.TestCase):
    def data(self):
        return dict(
            completed=2, duration=2, errors=["", ""], request_keys=["first", "second"],
            requested_output_lens=[4, 2], output_lens=[4, 2], ttfts=[0.2, 0.1],
            token_events=[[[1, 0.2], [4, 0.23]], [[2, 0.1]]],
            spec_verify_ct=[2, 1], spec_num_correct_drafts=[2, 1],
        )

    def test_speculative_bursts_keep_token_denominators(self):
        result = summarize_stream(self.data(), 2)
        self.assertAlmostEqual(result["output_tok_s"], 3)
        self.assertAlmostEqual(result["mean_request_tpot_ms"], 5)
        self.assertAlmostEqual(result["token_weighted_mean_itl_ms"], 7.5)
        self.assertAlmostEqual(result["tokens_per_verify"], 2)

    def test_partial_and_failed_requests_are_rejected(self):
        for field, replacement in (("completed", 1), ("output_lens", [3, 2]), ("errors", ["failure", ""]), ("token_events", [[[1, 0.2]], [[2, 0.1]]])):
            data = self.data()
            data[field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize_stream(data, 2)

    def test_request_identity_is_not_just_equal_counts(self):
        stream = summarize_stream(self.data(), 2)
        first = dict(tenants=dict(agent=stream, chat=stream))
        second = copy.deepcopy(first)
        assert_same_requests(first, second)
        second["tenants"]["agent"]["requests_detail"][0]["request_key"] = "different"
        with self.assertRaises(ValueError):
            assert_same_requests(first, second)

    def test_trace_excludes_warmup_and_flush_but_keeps_long_steps(self):
        source = "speculative_num_steps=3 speculative_num_draft_tokens=4\nSTEPTRACE_FEATURES source=request_cpu_v2\n"
        source += "STEPTRACE EXTEND t=1.000 gpu=10.000\nSTEPTRACE EXTEND t=2.000 gpu=350.000\nSTEPTRACE EXTEND t=3.000 gpu=1.000\n"
        result, count = measured_step_trace(source, 1.5, 2.8)
        self.assertEqual(count, 1)
        self.assertIn("gpu=350.000", result)
        self.assertIn("speculative_num_steps=3", result)
        self.assertNotIn("t=1.000", result)
        self.assertNotIn("t=3.000", result)
        with self.assertRaises(ValueError):
            measured_step_trace(source.replace("request_cpu_v2", "legacy"), 1.5, 2.8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
