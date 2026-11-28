import json
import sys
import tomllib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.baseline import ROOT, main


class BaselineTests(unittest.TestCase):
    def test_lut_core_uses_student_decoder_and_preserves_answer_budget(self):
        with TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "config.json").write_text("{}")
            tokens = [
                {"content": "<latent>", "id": 151665, "special": True},
                {"content": "</latent>", "id": 151666, "special": True},
            ]
            (model / "tokenizer.json").write_text(json.dumps({"added_tokens": tokens}))
            argv = [
                "baseline",
                "lut",
                "--lut-core",
                "--lut-steps",
                "8",
                "--checkpoint",
                tmp,
                "--output",
                tmp,
            ]

            def inspect(command, *, check, env):
                self.assertTrue(check)
                self.assertEqual(env["SGLANG_EXTERNAL_MODEL_PACKAGE"], "methods.lut")
                self.assertIn("max_new_tokens=42,temperature=0", command)
                directory = Path(command[command.index("--tasks") + 1])
                self.assertEqual(
                    {p.stem for p in directory.glob("*.yaml")},
                    {
                        "vstar_bench",
                        "tlr_hrbench4k_local",
                        "tlr_hrbench8k_local",
                        "mmerealworld_lite",
                    },
                )

            with (
                patch.object(sys, "argv", argv),
                patch("scripts.baseline.subprocess.run", side_effect=inspect) as run,
            ):
                main()
                run.assert_called_once()

    def test_paper_commands(self):
        recipes = tomllib.loads((ROOT / "recipes/benchmarks.toml").read_text())
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text("{}")
            for paper, recipe in recipes.items():
                for variant, tasks in recipe.items():
                    if variant.startswith("tasks_"):
                        size = variant.removeprefix("tasks_")
                        with self.subTest(paper=paper, size=size):
                            self.check_command(paper, size, tasks, tmp)

    def check_command(self, paper, size, tasks, tmp):
        def inspect(command, *, check):
            self.assertTrue(check)
            self.assertIn("--fail_on_error", command)
            directory = Path(command[command.index("--tasks") + 1])
            self.assertEqual({p.stem for p in directory.glob("*.yaml")}, set(tasks))
            for task in tasks:
                config = json.loads((directory / f"{task}.yaml").read_text())
                self.assertEqual(config["task"], task)
                self.assertRegex(
                    config["dataset_kwargs"]["revision"], r"^[0-9a-f]{40}$"
                )
            args = command[command.index("--model_args") + 1]
            if paper in {"lvr", "hylar"}:
                config = json.loads(args.split("mm_process_config=")[1])
                maximum = 5120 if paper == "lvr" else 16384
                self.assertEqual(
                    config["image"]["size"],
                    {
                        "shortest_edge": 128 * 28**2,
                        "longest_edge": maximum * 28**2,
                    },
                )

        argv = ["baseline", paper, "--size", size, "--checkpoint", tmp, "--output", tmp]
        with (
            patch.object(sys, "argv", argv),
            patch("scripts.baseline.subprocess.run", side_effect=inspect) as run,
        ):
            main()
            run.assert_called_once()
