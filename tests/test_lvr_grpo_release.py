import copy
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from methods.lvr import grpo_release
from methods.lvr.grpo_calibration import make_scheduler, validate_resume
from methods.lvr.grpo_pilot import Action


class ReleasedRecipeTests(unittest.TestCase):
    def test_released_mask_includes_closing_marker_and_excludes_opening(self):
        ids = [1, 20, 31, 31, 21, 2, 22, 20, 3, 21]
        path = [Action(token, None, 0.0) for token in ids]
        self.assertEqual(
            grpo_release.completion_mask(path, 20, 21, 22),
            [True, False, False, False, True, True, True, False, False, False],
        )
        path = [Action(token, None, 0.0) for token in [20, 22, 21, 1]]
        self.assertEqual(grpo_release.completion_mask(path, 20, 21, 22), [False] * 4)
        masks = [[True] * n for n in (3, 1, 4, 0, 2, 2, 2, 1)]
        self.assertEqual(grpo_release.token_denominators(masks), [8] * 4 + [7] * 4)

    def test_full_pool_keeps_answers_outside_calibration_types(self):
        rows = [
            {
                "id": i,
                "image": f"{i}.png",
                "conversations": [
                    {"from": "human", "value": "Question"},
                    {"from": "gpt", "value": f"<answer>{answer}</answer>"},
                ],
            }
            for i, answer in enumerate(
                ["A", "3", "x=2", "red", "left", "a,b", "yes", "E"]
            )
        ]
        raw = json.dumps(rows).encode()
        digest = hashlib.sha256(raw).hexdigest()
        with TemporaryDirectory() as directory:
            source = Path(directory) / "virl.json"
            source.write_bytes(raw)
            plan = grpo_release.batches(source, 2, digest)
            self.assertEqual(grpo_release.batches(source, 1, digest), plan[:1])
            self.assertCountEqual([row for batch in plan for row in batch], rows)

    def test_cosine_resume_matches_optimizer_and_rejects_a_new_horizon(self):
        def setup():
            p = torch.nn.Parameter(torch.ones(1))
            optimizer = torch.optim.AdamW([p], lr=5e-6)
            return p, optimizer, make_scheduler(optimizer, 1500)

        def advance(p, optimizer, scheduler, steps):
            used = []
            for _ in range(steps):
                used.append(optimizer.param_groups[0]["lr"])
                p.grad = torch.ones_like(p)
                optimizer.step()
                scheduler.step()
            return used

        p, opt, scheduler = setup()
        warmup = advance(p, opt, scheduler, 40)
        self.assertEqual(warmup[0], 0.0)
        self.assertAlmostEqual(warmup[39], 5e-6 * 39 / 45)
        clone, other_opt, other_sched = setup()
        clone.data.copy_(p.data)
        other_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        other_sched.load_state_dict(scheduler.state_dict())
        want = advance(p, opt, scheduler, 35)
        self.assertEqual(want, advance(clone, other_opt, other_sched, 35))
        torch.testing.assert_close(p, clone, rtol=0, atol=0)
        self.assertEqual(want[5], 5e-6)
        recipe = {"recipe": 2, "schedule_steps": 1500, "groups": [[1], [2]]}
        with self.assertRaises(ValueError):
            validate_resume(
                recipe, {**recipe, "schedule_steps": 1501, "groups": [[1], [2], [3]]}, 1
            )

    def test_symbolic_and_string_rewards(self):
        self.assertEqual(
            grpo_release.rewards(
                r"<answer>$\frac{1}{2}$</answer>", "<answer>$0.5$</answer>"
            ),
            (1, 0),
        )
        self.assertEqual(
            grpo_release.rewards(
                "<|lvr_start|><|lvr|><|lvr_end|> <answer>red</answer>",
                "<answer>red</answer>",
            ),
            (1, 1),
        )
        self.assertEqual(
            grpo_release.rewards("<answer>green</answer>", "<answer>red</answer>"),
            (0, 0),
        )
