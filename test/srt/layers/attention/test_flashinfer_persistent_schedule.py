import unittest

from sglang.srt.layers.attention.flashinfer_backend import (
    select_persistent_flipped_schedule,
)


class TestPersistentScheduleSelection(unittest.TestCase):
    def test_short_context_uses_flipped_default(self):
        self.assertIsNone(select_persistent_flipped_schedule(32 * 8192, 32))

    def test_long_context_uses_nonflipped_schedule(self):
        self.assertFalse(select_persistent_flipped_schedule(40 * 80000, 40))

    def test_boundary_is_nonflipped(self):
        self.assertFalse(select_persistent_flipped_schedule(4 * 32768, 4))

    def test_explicit_override_is_preserved(self):
        self.assertFalse(select_persistent_flipped_schedule(8 * 1024, 8, True))


if __name__ == "__main__":
    unittest.main()
