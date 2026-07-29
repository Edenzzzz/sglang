import gzip
import json
import random
import tempfile
import unittest

from sglang.bench_serving import sample_tracelab_claude_requests


class _Tokenizer:
    def get_vocab(self):
        return {str(index): index for index in range(32)}


class TestTraceLabReplay(unittest.TestCase):
    def test_preserves_real_session_order_and_delays(self):
        rows = []
        for session_index in range(2):
            for round_index in range(3):
                second = round_index * 3
                rows.append(
                    {
                        "provider": "claude",
                        "session_id": f"session-{session_index}",
                        "round_index": round_index,
                        "input_tokens_total": 100 + round_index * 10,
                        "prefix_tokens": 90 + round_index * 10,
                        "newly_append_tokens": 10,
                        "output_tokens": 4,
                        "timing_events": [
                            {
                                "timestamp": (
                                    f"2026-01-01T00:00:0{second}.000Z"
                                )
                            },
                            {
                                "timestamp": (
                                    f"2026-01-01T00:00:0{second + 1}.000Z"
                                )
                            },
                        ],
                    }
                )

        with tempfile.NamedTemporaryFile(suffix=".jsonl.gz") as trace_file:
            with gzip.open(trace_file.name, "wt") as output:
                for row in rows:
                    output.write(json.dumps(row) + "\n")

            random.seed(1)
            replay = sample_tracelab_claude_requests(
                trace_file.name,
                num_requests=4,
                num_groups=2,
                min_context_len=1,
                max_context_len=1000,
                min_output_len=1,
                tokenizer=_Tokenizer(),
                return_text=False,
            )

        warmups = [row for row in replay if row.warmup_only]
        requests = [row for row in replay if not row.warmup_only]
        self.assertEqual(len(warmups), 2)
        self.assertEqual(len(requests), 4)
        for session_id in {row.session_id for row in requests}:
            warmup = next(
                row for row in warmups if row.session_id == session_id
            )
            session_rows = [
                row for row in requests if row.session_id == session_id
            ]
            self.assertEqual(len(session_rows), 2)
            self.assertEqual(warmup.prompt_len, len(warmup.prompt))
            self.assertEqual(warmup.prompt_len, session_rows[0].prompt_len - 10)
            self.assertNotEqual(
                warmup.prompt,
                session_rows[0].prompt[: warmup.prompt_len],
            )
            self.assertEqual(
                session_rows[1].round_index,
                session_rows[0].round_index + 1,
            )
            self.assertEqual(session_rows[1].inter_round_delay_s, 2.0)


if __name__ == "__main__":
    unittest.main()
