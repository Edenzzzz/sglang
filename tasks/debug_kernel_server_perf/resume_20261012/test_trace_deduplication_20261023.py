import unittest

from deduplicate_trace_rounds_20261023 import unique_rounds


class TestTraceDeduplication(unittest.TestCase):
    def record(self, **changes):
        return dict(provider="claude", session_id="session", round_index=4, input_tokens_total=100, output_tokens=7, **changes)

    def test_exact_duplicates_only_and_order_preserved(self):
        first = self.record()
        second = dict(first, round_index=5, input_tokens_total=107)
        result, count = unique_rounds([first, dict(first), second, dict(second)])
        self.assertEqual(result, [first, second])
        self.assertEqual(count, 2)

    def test_identity_collision_is_not_silently_dropped(self):
        first = self.record()
        with self.assertRaises(ValueError):
            unique_rounds([first, dict(first, output_tokens=8)])
        with self.assertRaises(ValueError):
            unique_rounds([first, dict(first, timing_events=[{"timestamp": "different"}])])

    def test_provider_namespaces_are_independent(self):
        first = self.record()
        second = dict(first, provider="codex")
        result, count = unique_rounds([first, second])
        self.assertEqual(result, [first, second])
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
