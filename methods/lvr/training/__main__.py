"""Region-supervised fine-tuning of Qwen2.5-VL on the LVR Visual CoT release."""

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from filelock import FileLock
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from .data import ANNOTATIONS, IMAGE_REVISION, load_rows, prepare_images

REVISIONS = {
    "3b": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "7b": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
}
TOKENS = ("<|lvr_start|>", "<|lvr|>", "<|lvr_latent_end|>", "<|lvr_end|>")


def region_indices(box, height, width, merge):
    """Project a normalized rectangle onto the resized, spatially merged grid."""
    x0, y0, x1, y1 = box
    left = max(0, min(math.floor(x0 * width), width - 1)) // merge
    top = max(0, min(math.floor(y0 * height), height - 1)) // merge
    right = math.ceil(max(0, min(math.ceil(x1 * width), width)) / merge)
    bottom = math.ceil(max(0, min(math.ceil(y1 * height), height)) / merge)
    indices = [
        y * (width // merge) + x for y in range(top, bottom) for x in range(left, right)
    ]
    if not indices:
        raise ValueError(f"Empty visual region {box} in {height} x {width} grid")
    return indices


def format_prompt(processor, row):
    question, answer = row["conversations"]
    if (question["from"], answer["from"]) != ("human", "gpt") or question[
        "value"
    ].count("<image>") != 1:
        raise ValueError(f"Invalid dialogue for question {row['question_id']}")
    before, after = question["value"].split("<image>")
    content = [
        {"type": "text", "text": before.removesuffix("\n")},
        {"type": "image"},
        {"type": "text", "text": after.removeprefix("\n")},
    ]
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def adaptive_packs(rows, processor, data):
    # Each pack becomes a dynamically padded batch. Four examples shorter
    # than 4096 tokens fit the reference 16384-token limit.
    sizes_file = data / "viscot-image-sizes.json"
    with FileLock(data / ".lvr-data.lock"):
        if not sizes_file.exists():
            sizes = {}
            for name in sorted({row["image"][0] for row in rows}):
                with Image.open(data / "images" / name) as image:
                    sizes[name] = image.size
            tmp = sizes_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(sizes))
            tmp.replace(sizes_file)
    sizes = json.loads(sizes_file.read_text())
    random.Random(42).shuffle(rows)
    packs, short = [], []
    for i in range(0, len(rows), 512):
        batch = rows[i : i + 512]
        # The templates contain one token per image and region placeholder.
        texts = [
            format_prompt(processor, row)
            + row["conversations"][1]["value"].replace("<lvr>", "".join(TOKENS))
            + "<|im_end|>\n"
            for row in batch
        ]
        for row, ids in zip(
            batch,
            processor.tokenizer(
                texts, add_special_tokens=False, verbose=False
            ).input_ids,
            strict=True,
        ):
            w, h = sizes[row["image"][0]]
            h, w = smart_resize(
                h, w, factor=28, min_pixels=128 * 28**2, max_pixels=5120 * 28**2
            )
            region = region_indices(row["bboxes"][0], h // 14, w // 14, 2)
            length = len(ids) + h * w // 28**2 + len(region) - 2
            if length >= 4096:
                packs.append([row])
            else:
                short.append(row)
                if len(short) == 4:
                    packs.append(short)
                    short = []
    if short:
        packs.append(short)
    return packs


class VisualCoTBatch:
    def __init__(self, processor, images, answer_only=False):
        self.answer_only = answer_only
        self.processor = processor
        self.images = images

    def __call__(self, rows):
        prompts, pictures = [], []
        if isinstance(rows[0], list):
            rows = [row for pack in rows for row in pack]
        for row in rows:
            (image,) = row["image"]
            prompts.append(format_prompt(self.processor, row))
            with Image.open(self.images / image) as picture:
                pictures.append(picture.convert("RGB"))
        vision = self.processor.image_processor(images=pictures, return_tensors="pt")
        merge = self.processor.image_processor.merge_size
        ids, labels, rois = [], [], []
        offset = 0
        for row, prompt, (_, h, w) in zip(
            rows, prompts, vision.image_grid_thw.tolist(), strict=True
        ):
            count = h * w // merge**2
            prompt = prompt.replace("<|image_pad|>", "<|image_pad|>" * count)
            prefix = self.processor.tokenizer.encode(prompt, add_special_tokens=False)
            chunks = row["conversations"][1]["value"].split("<lvr>")
            response = chunks[0]
            for box, suffix in zip(row["bboxes"], chunks[1:], strict=True):
                indices = region_indices(box, h, w, merge)
                rois.extend(offset + i for i in indices)
                response += (
                    TOKENS[0]
                    + TOKENS[1] * len(indices)
                    + TOKENS[2]
                    + TOKENS[3]
                    + suffix
                )
            if self.answer_only:
                response = "".join(chunks).lstrip("\n")
            target = self.processor.tokenizer.encode(
                response + "<|im_end|>\n", add_special_tokens=False
            )
            ids.append(torch.tensor(prefix + target))
            labels.append(torch.tensor([-100] * len(prefix) + target))
            offset += count
        pad = self.processor.tokenizer.pad_token_id
        tokens = pad_sequence(ids, batch_first=True, padding_value=pad)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        batch = dict(
            vision,
            input_ids=tokens,
            labels=labels,
            attention_mask=torch.arange(tokens.size(1))[None, :]
            < torch.tensor([len(x) for x in ids])[:, None],
        )

        if self.answer_only:
            batch["mm_token_type_ids"] = (tokens == self.processor.image_token_id).int()
        else:
            labels[
                tokens == self.processor.tokenizer.convert_tokens_to_ids(TOKENS[1])
            ] = -100
            batch["roi_indices"] = torch.tensor(rois)
        return batch


class QwenWithLVR(Qwen2_5_VLForConditionalGeneration):
    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        roi_indices,
        labels,
    ):
        image_features = self.get_image_features(
            pixel_values, image_grid_thw, return_dict=True
        )
        targets = torch.cat(image_features.pooler_output)[roi_indices]
        latent = input_ids == self.config.lvr_id
        embeds = self.get_input_embeddings()(input_ids)
        embeds[latent] = targets.to(embeds.dtype)
        hidden = self.model(
            input_ids=input_ids,
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=(input_ids == self.config.image_token_id).int(),
            mm_encoder_outputs={"image": image_features},
            use_cache=False,
        ).last_hidden_state[:, :-1]
        supervised = labels[:, 1:] != -100
        ce = F.cross_entropy(
            self.lm_head(hidden[supervised]).float(), labels[:, 1:][supervised]
        )
        mse = F.mse_loss(hidden[latent[:, 1:]].float(), targets.float())
        return {"loss": ce + 0.1 * mse, "ce": ce.detach(), "mse": mse.detach()}


# LVR changes the forward pass only. Use Qwen's checkpoint key conversions.
register_checkpoint_conversion_mapping(
    QwenWithLVR.__name__,
    get_checkpoint_conversion_mapping(Qwen2_5_VLForConditionalGeneration.__name__),
)


def load_base(model_class, source, **kwargs):
    model, info = model_class.from_pretrained(
        source, output_loading_info=True, **kwargs
    )
    problems = {key: value for key, value in info.items() if value}
    if problems:
        raise ValueError(f"Incomplete base checkpoint {source}: {problems}")
    return model


class LVRTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(**inputs)
        if self.args.gradient_accumulation_steps == 1 and self.args.logging_steps == 1:
            self.log({k: outputs[k].item() for k in ("ce", "mse")})
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory containing lvr_data/ and images/",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", choices=REVISIONS, default="3b")
    parser.add_argument("--base", type=Path, help="Local base snapshot")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument(
        "--overfit", type=int, help="Use a fixed subset of this many records"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Shard model and optimizer across GPUs with FSDP2",
    )
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="Train ordinary answer SFT on the same examples and adaptive batches",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.overfit and args.batch_size != 1:
        parser.error("Full runs use one adaptive pack per GPU microbatch")
    if (
        min(args.steps, args.batch_size, args.accumulation, args.overfit or 1) < 1
        or args.overfit == 0
    ):
        parser.error("Step, batch and subset sizes must be positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("Use an empty output directory or --resume")
    rows = load_rows(args.data)
    prepare_images(args.data, rows)
    if args.overfit:
        rows = random.Random(42).sample(rows, args.overfit)
    set_seed(42)
    source = args.base or f"Qwen/Qwen2.5-VL-{args.size.upper()}-Instruct"
    processor = AutoProcessor.from_pretrained(source, revision=REVISIONS[args.size])
    processor.image_processor.size = {
        "shortest_edge": 128 * 28**2,
        "longest_edge": 5120 * 28**2,
    }
    processor.tokenizer.add_tokens(list(TOKENS), special_tokens=True)
    if not args.overfit:
        rows = adaptive_packs(rows, processor, args.data)
    model_class = (
        Qwen2_5_VLForConditionalGeneration if args.answer_only else QwenWithLVR
    )
    model = load_base(
        model_class,
        source,
        revision=REVISIONS[args.size],
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    for key, tok in zip(
        ("lvr_start_id", "lvr_id", "lvr_latent_end_id", "lvr_end_id"),
        TOKENS,
        strict=True,
    ):
        setattr(model.config, key, processor.tokenizer.convert_tokens_to_ids(tok))
    if len(processor.tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(processor.tokenizer))
    model.config.lvr_head = model.config.latent_end_token = False
    model.model.visual.requires_grad_(False)
    if not args.fsdp:
        model.model.visual.to(torch.bfloat16)
    training = TrainingArguments(
        output_dir=str(args.output),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        learning_rate=1e-5,
        weight_decay=0.1,
        warmup_steps=math.ceil(args.steps * 0.03),
        lr_scheduler_type="cosine",
        bf16=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=not args.fsdp,
        remove_unused_columns=False,
        seed=42,
        save_strategy="no" if args.overfit else "steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=1,
        report_to="none",
        dataloader_num_workers=0 if args.overfit else 2,
        ddp_find_unused_parameters=False,
        fsdp=args.fsdp,
        fsdp_config={
            "version": 2,
            "activation_checkpointing": True,
            "transformer_layer_cls_to_wrap": ["Qwen2_5_VLDecoderLayer"],
        }
        if args.fsdp
        else None,
    )
    trainer_class = Trainer if args.answer_only else LVRTrainer
    trainer = trainer_class(
        model=model,
        args=training,
        train_dataset=rows,
        processing_class=processor,
        data_collator=VisualCoTBatch(processor, args.data / "images", args.answer_only),
    )
    recipe = {
        "model": str(source),
        "revision": REVISIONS[args.size],
        "annotations": ANNOTATIONS,
        "images_revision": IMAGE_REVISION,
        "seed": 42,
        "steps": args.steps,
        "overfit": args.overfit,
        "batch_size": args.batch_size,
        "accumulation": args.accumulation,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "answer_only": args.answer_only,
        "fsdp": args.fsdp,
    }
    manifest = args.output / "sft_recipe.json"
    if args.resume and json.loads(manifest.read_text()) != recipe:
        raise ValueError("Resume requires the original training recipe")
    if trainer.is_world_process_zero():
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temp = manifest.with_suffix(".tmp")
        temp.write_text(json.dumps(recipe, indent=2) + "\n")
        temp.replace(manifest)
    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
