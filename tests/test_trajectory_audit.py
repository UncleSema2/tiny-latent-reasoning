import unittest

import torch
import torch.nn.functional as F
from transformers import (
    BatchFeature,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from methods.lut.audit_trajectory import native_score, random_direction
from methods.lut.teacher import Teacher
from methods.lut.validate_teacher import judge_scores


class TrajectoryReplayTests(unittest.TestCase):
    @torch.inference_mode()
    def test_masked_and_ordinary_forwards_score_the_same_answer_positions(self):
        torch.manual_seed(17)
        config = Qwen2_5_VLConfig(
            text_config={
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_hidden_layers": 2,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 64,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000,
                    "mrope_section": [4, 6, 6],
                },
            },
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "num_heads": 4,
                "intermediate_size": 64,
                "out_hidden_size": 32,
                "fullatt_block_indexes": [0],
            },
            image_token_id=55,
            vision_start_token_id=54,
            vision_end_token_id=56,
            attn_implementation="sdpa",
        )
        model = Qwen2_5_VLForConditionalGeneration(config).eval()
        bounds = [57, 58]
        teacher = Teacher(model, bounds)
        prompt = BatchFeature(
            {
                "input_ids": torch.tensor([[2, 54, 55, 56, 7, 8]]),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
                "pixel_values": torch.randn(4, 3 * 2 * 14 * 14),
            }
        )
        visual = model.get_image_features(prompt.pixel_values, prompt.image_grid_thw)
        for answer in (torch.tensor([[13]]), torch.tensor([[13, 17, 23]])):
            with self.subTest(length=answer.numel()):
                logits, z = teacher(prompt, answer, return_trajectory=True)
                cached_logits, cached_z, layers = teacher(
                    prompt, answer, collect_layers=True
                )
                torch.testing.assert_close(cached_logits, logits)
                torch.testing.assert_close(cached_z, z)
                self.assertEqual(tuple(layers.shape), (2, 8, 32))
                torch.testing.assert_close(layers[-1, :-1], z[0, 1:])
                student = Teacher(model, bounds, bottleneck=False)
                student_logits, student_z, _ = student(
                    prompt, answer, collect_layers=True
                )
                self.assertAlmostEqual(
                    native_score(model, prompt, answer, visual, bounds, student_z),
                    -F.cross_entropy(student_logits[0], answer[0]).item(),
                    places=5,
                )
                self.assertAlmostEqual(
                    native_score(
                        model, prompt, answer, visual, bounds, z, bottleneck=True
                    ),
                    -F.cross_entropy(logits[0], answer[0]).item(),
                    places=5,
                )
                ref = judge_scores(
                    model, bounds, prompt, answer, z, torch.Generator().manual_seed(43)
                )
                control = random_direction(z, 43) * z.norm(dim=-1, keepdim=True)
                for expected, states in zip(ref, (z, control), strict=True):
                    self.assertAlmostEqual(
                        native_score(model, prompt, answer, visual, bounds, states),
                        expected.item(),
                        places=5,
                    )
