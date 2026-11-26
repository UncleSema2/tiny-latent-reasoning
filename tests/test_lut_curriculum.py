import unittest

from methods.lut.curriculum import build_curriculum


def score(sample_id, subset, teacher, control):
    return {
        "id": sample_id,
        "subset": subset,
        "teacher_logprobs": teacher,
        "control_logprobs": control,
    }


class LUTCurriculumTests(unittest.TestCase):
    def test_answer_mean_strict_threshold_and_subset_split(self):
        rows = [
            score("threshold", "a", [-1.0], [-0.5]),
            score("low-a", "a", [-1.0, -1.5], [-1.0, -1.0]),
            score("high-a", "a", [-0.5], [-1.0]),
            score("low-b", "b", [-1.0], [-3.0]),
            score("high-b", "b", [-1.0], [-4.0]),
        ]
        result = build_curriculum(rows, seed=19)
        self.assertEqual(result["input_count"], 5)
        self.assertEqual(result["filtered_count"], 1)
        self.assertCountEqual(result["low"], ["low-a", "low-b"])
        self.assertCountEqual(result["high"], ["high-a", "high-b"])

    def test_ties_and_odd_subset_are_reproducible_across_shard_orders(self):
        rows = [score(str(i), "a", [-1], [-1]) for i in range(7)]
        result = build_curriculum(rows, seed=3)
        self.assertCountEqual(result["low"], ["0", "1", "2"])
        self.assertCountEqual(result["high"], ["3", "4", "5", "6"])
        self.assertEqual(result, build_curriculum(reversed(rows), seed=3))
        self.assertNotEqual(result["high"], build_curriculum(rows, seed=4)["high"])

    def test_invalid_scores_and_duplicate_ids(self):
        for teacher, control in [
            ([], []),
            ([0], [0, 0]),
            ([float("nan")], [0]),
            ([True], [0]),
        ]:
            with self.subTest(teacher=teacher), self.assertRaises(ValueError):
                build_curriculum([score("x", "ocr", teacher, control)], seed=0)
        row = score("duplicate", "ocr", [-2], [0])
        with self.assertRaisesRegex(ValueError, "duplicate sample id"):
            build_curriculum([row, row], seed=0)
