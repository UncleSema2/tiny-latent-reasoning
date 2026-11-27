import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from methods.lvr.training.__main__ import (
    QwenWithLVR,
    StopAfterStep,
    load_base,
    region_indices,
    resumed_recipe,
)


class LVRTrainingTests(unittest.TestCase):
    def test_fsdp_resume_preserves_global_batch_and_records_transition(self):
        original = {"world_size": 8, "accumulation": 8, "fsdp": True, "seed": 42}
        current = {**original, "world_size": 16, "accumulation": 4}
        updated = resumed_recipe(original, current, 500)
        self.assertEqual(
            updated["parallelism_history"],
            [
                {"from_step": 0, "world_size": 8, "accumulation": 8},
                {"from_step": 500, "world_size": 16, "accumulation": 4},
            ],
        )
        self.assertEqual(resumed_recipe(updated, current, 500), updated)
        for change in ({"accumulation": 8}, {"fsdp": False}, {"seed": 9}):
            with self.assertRaisesRegex(ValueError, "original training recipe"):
                resumed_recipe(original, current | change, 500)

    def test_staged_training_matches_uninterrupted_schedule(self):
        class Regression(torch.nn.Linear):
            def forward(self, x, labels):
                return {"loss": F.mse_loss(super().forward(x), labels)}

        data = [
            {"x": torch.tensor([float(i)]), "labels": torch.tensor([2.0 * i])}
            for i in range(8)
        ]

        def make_trainer(path, stop=None):
            torch.manual_seed(42)
            return Trainer(
                model=Regression(1, 1),
                args=TrainingArguments(
                    output_dir=str(path),
                    use_cpu=True,
                    max_steps=4,
                    per_device_train_batch_size=1,
                    learning_rate=0.01,
                    warmup_steps=1,
                    lr_scheduler_type="cosine",
                    save_strategy="no",
                    report_to="none",
                    disable_tqdm=True,
                ),
                train_dataset=data,
                callbacks=[StopAfterStep(stop)] if stop else None,
            )

        with TemporaryDirectory() as folder:
            full = make_trainer(Path(folder) / "full")
            full.train()
            path = Path(folder) / "staged"
            stage = make_trainer(path, stop=2)
            stage.train()
            self.assertEqual(stage.state.global_step, 2)
            self.assertEqual(stage.state.max_steps, 4)
            self.assertTrue((path / "checkpoint-2" / "scheduler.pt").is_file())
            with self.assertRaisesRegex(ValueError, "already reached"):
                make_trainer(path, stop=2).train(resume_from_checkpoint=True)
            resumed = make_trainer(path)
            resumed.train(resume_from_checkpoint=True)
            self.assertEqual(resumed.state.global_step, 4)
            for key, tensor in full.model.state_dict().items():
                torch.testing.assert_close(
                    tensor, resumed.model.state_dict()[key], atol=0, rtol=0
                )

    def test_loads_every_tensor_from_original_qwen_layout(self):
        config = Qwen2_5_VLConfig(
            text_config={
                "vocab_size": 32,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
            },
            vision_config={
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 16,
            },
        )
        base = Qwen2_5_VLForConditionalGeneration(config)
        weights = {
            key.replace("model.language_model.", "model.").replace(
                "model.visual.", "visual."
            ): value.clone()
            for key, value in base.state_dict().items()
        }
        with TemporaryDirectory() as folder:
            config.save_pretrained(folder)
            path = Path(folder) / "model.safetensors"
            save_file(weights, path)
            loaded = load_base(QwenWithLVR, folder, disable_mmap=True)
            for key, value in loaded.state_dict().items():
                torch.testing.assert_close(
                    value, base.state_dict()[key], rtol=0, atol=0
                )
            del weights["model.layers.0.self_attn.q_proj.weight"]
            save_file(weights, path)
            with self.assertRaisesRegex(ValueError, "missing_keys"):
                load_base(QwenWithLVR, folder, disable_mmap=True)

    def test_regions_follow_merged_row_major_grid(self):
        self.assertEqual(region_indices([0.5, 0, 1, 1], 4, 8, 2), [2, 3, 6, 7])
        self.assertEqual(region_indices([-0.1, 0.5, 0.25, 1.1], 4, 8, 2), [4])

    def test_visual_targets_and_shifted_losses_across_images(self):
        features = (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0]]))
        hidden = torch.arange(24, dtype=torch.float32).view(2, 6, 2).requires_grad_()
        language = Mock(return_value=SimpleNamespace(last_hidden_state=hidden))
        model = SimpleNamespace(
            model=language,
            config=SimpleNamespace(lvr_id=2, image_token_id=3),
            get_image_features=Mock(
                return_value=SimpleNamespace(pooler_output=features)
            ),
            get_input_embeddings=lambda: torch.nn.Embedding(5, 2),
            lm_head=torch.nn.Linear(2, 5, bias=False),
        )
        ids = torch.tensor([[3, 3, 1, 2, 2, 4], [3, 1, 2, 4, 0, 0]])
        labels = torch.tensor(
            [[-100, -100, 1, -100, -100, 4], [-100, 1, -100, 4, -100, -100]]
        )
        output = QwenWithLVR.forward(
            model, ids, ids != 0, None, None, torch.tensor([1, 0, 2]), labels
        )
        targets = torch.cat((features[0].flip(0), features[1]))
        embeds = language.call_args.kwargs["inputs_embeds"]
        torch.testing.assert_close(embeds[ids == 2], targets)
        mse = F.mse_loss(hidden[[0, 0, 1], [2, 3, 1]], targets)
        ce = F.cross_entropy(
            model.lm_head(hidden[[0, 0, 1, 1], [1, 4, 0, 2]]),
            torch.tensor([1, 4, 1, 4]),
        )
        torch.testing.assert_close(output["loss"], ce + 0.1 * mse)
        output["loss"].backward()
        self.assertTrue(hidden.grad.isfinite().all())
