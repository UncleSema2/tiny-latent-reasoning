import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from methods.lvr.sglang_model import Qwen2_5_VLForConditionalGeneration


class FixedStepTests(unittest.TestCase):
    def test_interleaved_requests_keep_independent_budgets_and_embeddings(self):
        def embed(ids):
            return ids.float().unsqueeze(1)

        model = Mock(side_effect=lambda *args, input_embeds: input_embeds + 10)
        model.embed_tokens = embed
        decoder = SimpleNamespace(
            remaining=torch.zeros(3, dtype=torch.long),
            previous=torch.zeros(3, 1),
            config=SimpleNamespace(
                lvr_start_id=5, lvr_steps=2, eos_token_id=1, bos_token_id=2
            ),
            model=model,
            lm_head=Mock(),
            logits_processor=lambda ids, *_: SimpleNamespace(
                next_token_logits=torch.zeros(len(ids), 6)
            ),
        )

        def step(slots, ids):
            batch = SimpleNamespace(
                forward_mode=Mock(is_decode=lambda: True),
                req_pool_indices=torch.tensor(slots),
                mrope_positions=None,
            )
            result = Qwen2_5_VLForConditionalGeneration.forward(
                decoder, torch.tensor(ids), None, batch
            )
            return result.next_token_logits

        scores = step([1, 2], [5, 3])
        self.assertTrue(scores[0, 1:3].isneginf().all())
        self.assertTrue(scores[1].isfinite().all())
        step([2, 1], [5, 4])
        self.assertEqual(decoder.previous.flatten().tolist(), [0, 25, 15])
        scores = step([1, 2], [4, 4])
        self.assertTrue(scores[0].isfinite().all())
        self.assertEqual(decoder.remaining.tolist(), [0, 0, 1])
        step([1, 2], [3, 3])
        self.assertEqual(decoder.previous.flatten().tolist(), [0, 13, 35])
