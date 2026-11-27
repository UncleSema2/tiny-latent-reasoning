import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from methods.lvr.grpo_pilot import GROUP, LATENTS, replay, rewards, rollout


class LatentPolicyTests(unittest.TestCase):
    def test_cached_rollout_and_teacher_forced_text_scores_agree(self):
        torch.manual_seed(42)
        config = Qwen2_5_VLConfig(
            text_config={
                "vocab_size": 32,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "hidden_size": 24,
                "intermediate_size": 48,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "mrope_section": [2, 2, 2],
                },
            },
            vision_config={
                "depth": 1,
                "hidden_size": 24,
                "intermediate_size": 48,
                "num_heads": 2,
                "out_hidden_size": 24,
            },
            image_token_id=22,
            lvr_start_id=23,
        )
        model = Qwen2_5_VLForConditionalGeneration(config).eval()
        ids = torch.tensor([[1, 22, 10]])
        prompt = ids, torch.arange(3).expand(3, 1, -1), torch.randn(1, 24)
        tokens = [23] + [7] * LATENTS + [8, 2]

        def draw(probs, count):
            return torch.full((GROUP, count), tokens.pop(0), dtype=torch.long)

        with patch("torch.multinomial", side_effect=draw):
            paths = rollout(model, prompt, {2})
        self.assertEqual(len(paths), GROUP)
        logp, mask = replay(model, prompt, paths[0])
        self.assertEqual(mask.tolist(), [True] + [False] * LATENTS + [True, True])
        cached = torch.tensor([t[2] for t in paths[0]])[mask]
        torch.testing.assert_close(logp, cached, atol=1e-5, rtol=1e-5)
        logp.sum().backward()
        self.assertGreater(model.lm_head.weight.grad.norm().item(), 0)

    def test_letter_and_format_rewards_are_independent(self):
        prefix = "<|lvr_start|><|lvr|><|lvr_end|>"
        self.assertEqual(rewards(prefix + " <answer>B</answer>", "B"), (1, 1))
        self.assertEqual(rewards(prefix + " <answer>A</answer>", "B"), (0, 1))
        self.assertEqual(rewards("<answer>B</answer>", "B"), (1, 0))
