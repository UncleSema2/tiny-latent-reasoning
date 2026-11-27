import unittest
from unittest.mock import patch

from methods.lut.teacher import epoch_batches


class EpochTests(unittest.TestCase):
    def test_rank_coverage_partial_batch_and_resume(self):
        rows = list(range(11))
        with patch(
            "methods.lut.teacher.prepare_example",
            side_effect=lambda _, row, *args: (row, row),
        ):

            def shard(rank, completed):
                return [
                    (row, weight)
                    for row, _, weight in epoch_batches(
                        rows, rank, 4, completed, None, None
                    )
                ]

            shards = [shard(rank, 0) for rank in range(4)]
            seen = [
                row
                for batch in zip(*shards, strict=True)
                for row, weight in batch
                if weight
            ]
            self.assertEqual(seen, rows)
            for batch in zip(*shards, strict=True):
                self.assertAlmostEqual(sum(weight for _, weight in batch), 4)
            for rank in range(4):
                self.assertEqual(shard(rank, 1), shards[rank][1:])
