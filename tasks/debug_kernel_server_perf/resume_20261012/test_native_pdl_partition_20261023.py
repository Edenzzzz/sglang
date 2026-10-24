import unittest

from native_fa3_fa2_pdl_20261023 import geometry_split_policy, geometry_split_policy_v2, geometry_split_policy_v3, partition_lengths


class TestPartition(unittest.TestCase):
    def test_verify_and_decode(self):
        for tail in (1, 4):
            shape = partition_lengths([128, 17, tail, tail], [0, 4096, 393216, 2048], 28, 4)
            self.assertEqual(shape.prefix_rows, 2)
            self.assertEqual(shape.prefix_tokens, 145)
            self.assertEqual(shape.maximum_row_query, tail)
            self.assertEqual(shape.kv_lengths, [128, 4113, 393216 + tail, 2048 + tail])

    def test_head_ratio_boundary(self):
        self.assertIsNotNone(partition_lengths([128, 4], [0, 8192], 32, 8))
        self.assertIsNone(partition_lengths([128, 4], [0, 8192], 24, 2))
        self.assertIsNotNone(partition_lengths([128, 1], [0, 8192], 24, 2))

    def test_opt_in_q64_row_class(self):
        self.assertIsNotNone(partition_lengths([128, 4], [0, 8192], 24, 2, 64))
        self.assertIsNotNone(partition_lengths([128, 8], [0, 8192], 28, 4, 64))
        self.assertIsNone(partition_lengths([128, 6], [0, 8192], 24, 2, 64))
        self.assertIsNone(partition_lengths([128, 10], [0, 8192], 28, 4, 64))
        with self.assertRaises(ValueError):
            partition_lengths([128, 4], [0, 8192], 24, 2, 48)

    def test_noncontiguous_or_unmixed(self):
        for lengths in ([4, 4], [128, 512], [128, 4, 512], [4, 128]):
            self.assertIsNone(partition_lengths(lengths, [0] * len(lengths), 28, 4))

    def test_bad_host_metadata(self):
        for queries, prefixes in (([], []), ([4], []), ([0], [0]), ([True], [0]), ([4], [-1]), ([4], [0.0])):
            with self.assertRaises(ValueError):
                partition_lengths(queries, prefixes, 28, 4)
        with self.assertRaises(ValueError):
            partition_lengths([128, 4], [0, 2**31], 28, 4)
        with self.assertRaises(ValueError):
            partition_lengths([128, 4], [0, 2048], 29, 4)


    def test_geometry_policy(self):
        cases = (
            ([79, 4], [60273, 59996], 8, 1024),
            ([84, 4], [277, 65532], 1, 2048),
            ([2048, 4], [32768, 131068], 1, 4096),
            ([128, 4], [0, 393212], 1, 8192),
            ([128, 4], [0, 786428], 1, 8192),
        )
        for queries, prefixes, expected_prefix, expected_rows in cases:
            shape = partition_lengths(queries, prefixes, 28, 4)
            selected = geometry_split_policy(shape, 28, 78)
            self.assertEqual(selected["prefix_splits"], expected_prefix)
            self.assertEqual(selected["row_split_size"], expected_rows)


    def test_dominant_cached_row_is_not_hidden_by_fresh_queries(self):
        queries = [133, 15, 116, 65, 27, 1282, 63, 42, 84, 220, 4]
        lengths = [394240, 38, 139, 88, 50, 1305, 86, 65, 107, 243, 394244]
        shape = partition_lengths(queries, [length - query for length, query in zip(lengths, queries)], 28, 4)
        self.assertEqual(geometry_split_policy(shape, 28, 78)["prefix_splits"], 1)
        selected = geometry_split_policy_v2(shape, 28, 78)
        self.assertEqual(selected["prefix_splits"], 16)
        self.assertEqual(selected["dominant_prefill_query"], 133)
        self.assertEqual(selected["dominant_prefill_kv"], 394240)

    def test_large_compute_dominant_prefill_stays_unsplit(self):
        shape = partition_lengths([2046, 4], [6116, 34826], 28, 4)
        self.assertEqual(geometry_split_policy_v2(shape, 28, 78)["prefix_splits"], 1)

    def test_packed_gqa_tiles_follow_native_fa3_layout(self):
        for query, prefix, expected_tiles in ((79, 60273, 20), (71, 39800, 16)):
            shape = partition_lengths([query, 4], [prefix, 65532], 28, 4)
            selected = geometry_split_policy_v3(shape, 28, 4, 78)
            self.assertEqual(selected["packed_prefill_tiles"], expected_tiles)
            self.assertEqual(selected["prefix_splits"], 32)
            self.assertEqual(selected["row_split_size"], geometry_split_policy_v2(shape, 28, 78)["row_split_size"])

    def test_packed_tile_count_handles_ragged_queries_and_gqa_one(self):
        shape = partition_lengths([33, 65, 4], [8192, 8192, 8192], 32, 8)
        self.assertEqual(geometry_split_policy_v3(shape, 32, 8, 78)["packed_prefill_tiles"], 40)
        shape = partition_lengths([79, 1], [60273, 8192], 28, 28)
        self.assertEqual(geometry_split_policy_v3(shape, 28, 28, 78)["prefix_splits"], geometry_split_policy_v2(shape, 28, 78)["prefix_splits"])

    def test_packed_policy_preserves_large_query_guard(self):
        shape = partition_lengths([2046, 4], [6116, 34826], 28, 4)
        self.assertEqual(geometry_split_policy_v3(shape, 28, 4, 78)["prefix_splits"], 1)
        with self.assertRaises(ValueError):
            geometry_split_policy_v3(shape, 28, 3, 78)


if __name__ == "__main__":
    unittest.main(verbosity=2)
