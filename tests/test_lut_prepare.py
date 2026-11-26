import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from methods.lut.prepare import BOX_REQUEST, prepare


class VisualCoTSampleTests(unittest.TestCase):
    def test_projection_repeatability_and_missing_media(self):
        row = {
            "split": "train",
            "dataset": "gqa",
            "question_id": 12,
            "image": ["cot/gqa/a.jpg", "cot/gqa/a.jpg###[1,2,3,4]"],
            "conversations": [
                {"from": "human", "value": f"<image>\nWhat color is it? {BOX_REQUEST}"},
                {"from": "gpt", "value": "[0.1, 0.2, 0.3, 0.4]"},
                {"from": "human", "value": "<image>"},
                {"from": "gpt", "value": "Blue."},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gqa").mkdir()
            image = root / "gqa/a.jpg"
            image.touch()
            dest = root / "sample.jsonl"
            rows = [dict(row, question_id=i) for i in range(4)]
            report = prepare(rows, root, dest, 3, 5)
            data = dest.read_bytes()
            self.assertEqual(report["sample_sha256"], hashlib.sha256(data).hexdigest())
            result = [json.loads(line) for line in data.splitlines()]
            self.assertEqual(len({r["id"] for r in result}), 3)
            for r in result:
                self.assertEqual(
                    r,
                    {
                        "id": f"viscot:{r['question_id']}",
                        "question_id": r["question_id"],
                        "subset": "gqa",
                        "image": "gqa/a.jpg",
                        "question": "What color is it?",
                        "answer": "Blue.",
                    },
                )
            self.assertEqual(prepare(rows, root, dest, 3, 5), report)
            image.unlink()
            with self.assertRaises(FileNotFoundError):
                prepare(rows, root, dest, 4, 6)
            self.assertEqual(dest.read_bytes(), data)
