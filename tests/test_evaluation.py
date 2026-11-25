import tomllib
import unittest
from pathlib import Path

from lmms_eval.tasks import TaskManager

from tasks.visualpuzzles.utils import score

ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_paper_task_ids(self):
        suites = tomllib.loads((ROOT / "recipes/benchmarks.toml").read_text())
        available = TaskManager(include_path=str(ROOT / "tasks")).all_tasks
        for paper, recipe in suites.items():
            for size, tasks in recipe.items():
                if size.startswith("tasks_"):
                    with self.subTest(paper=paper, size=size):
                        self.assertFalse(set(tasks).difference(available))

    def test_visualpuzzles_scoring(self):
        doc = {"options": None, "answer": "A"}
        for response, expected in [("A", 1), ("B", 0), ("", 0), ("Answer: (A)", 1)]:
            with self.subTest(response=response):
                self.assertEqual(score(doc, [response])["accuracy"], expected)
