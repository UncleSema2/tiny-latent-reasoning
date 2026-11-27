"""Calibrate single-GPU latent GRPO with letter and numeric ViRL questions."""

import argparse
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from transformers import set_seed

from . import grpo_pilot as pilot
from .training.__main__ import QwenWithLVR


def batches(annotations, steps):
    raw = annotations.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pilot.ANNOTATIONS_SHA256:
        raise ValueError("Unexpected ViRL annotation digest")
    letters, numbers = [], []
    for row in json.loads(raw):
        if not isinstance(row.get("image"), str):
            continue
        match = re.fullmatch(
            r"<answer>(.+?)</answer>", row["conversations"][1]["value"]
        )
        if match is None:
            continue
        if re.fullmatch("[A-D]", match[1]):
            letters.append(row)
        elif pilot.number(match[1]) is not None:
            numbers.append(row)
    rng = random.Random(42)
    rng.shuffle(letters)
    rng.shuffle(numbers)
    if min(len(letters), len(numbers)) < 2 * steps:
        raise ValueError("Insufficient ViRL rows for the requested number of updates")
    return [letters[i : i + 2] + numbers[i : i + 2] for i in range(0, 2 * steps, 2)]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def input_manifest(checkpoint, images, groups):
    files = sorted(
        p
        for p in checkpoint.iterdir()
        if p.is_file()
        and (
            p.suffix in {".safetensors", ".jinja"}
            or p.name
            in {
                "config.json",
                "generation_config.json",
                "preprocessor_config.json",
                "processor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "merges.txt",
                "vocab.json",
                "model.safetensors.index.json",
            }
        )
    )
    return {
        "recipe": 1,
        "seed": 42,
        "annotations_sha256": pilot.ANNOTATIONS_SHA256,
        "sft_files": {p.name: digest(p) for p in files},
        "groups": [
            [(row["id"], digest(images / row["image"])) for row in group]
            for group in groups
        ],
    }


def validate_resume(previous, recipe, step):
    planned = previous["groups"]
    groups = recipe["groups"]
    if not 0 < step <= len(planned) or step >= len(groups):
        raise ValueError(
            "Checkpoint step must precede the target and belong to the saved plan"
        )
    if previous != {**recipe, "groups": groups[: len(planned)]}:
        raise ValueError(
            "Resume requires identical settings, SFT files and the original batch prefix"
        )


def save(output, step, model, processor, optimizer, recipe):
    target = output / f"checkpoint-{step}"
    with TemporaryDirectory(dir=output, prefix=".saving-") as temp:
        root = Path(temp)
        model.save_pretrained(root)
        processor.save_pretrained(root)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "step": step,
                "recipe": recipe,
            },
            root / "training.pt",
        )
        root.rename(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "annotations", "images", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--save-at", type=int, help="Also save at this intermediate update"
    )
    parser.add_argument(
        "--save-every", type=int, help="Save at multiples of this global step interval"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Calibration checkpoint to restore into a fresh output directory",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.save_at is not None and not 0 < args.save_at < args.steps:
        parser.error("Use positive steps and an intermediate --save-at")
    if args.save_every is not None and args.save_every < 1:
        parser.error("Use a positive --save-every interval")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Use a positive finite learning rate")
    groups = batches(args.annotations, args.steps)
    recipe = input_manifest(args.checkpoint, args.images, groups)
    # Recipe version 1 uses the paper learning rate unless an override is recorded.
    if args.learning_rate != 1e-5:
        recipe["learning_rate"] = args.learning_rate
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    set_seed(42)
    start = 0
    if args.resume:
        state = torch.load(
            args.resume / "training.pt",
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        start = state["step"]
        try:
            validate_resume(state["recipe"], recipe, start)
        except ValueError as exc:
            parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    processor = pilot.load_processor(args.checkpoint)
    reference = pilot.load_policy(args.checkpoint, QwenWithLVR).requires_grad_(False)
    model = pilot.load_policy(args.resume or args.checkpoint, QwenWithLVR)
    # Both policies use features from the same frozen SFT vision tower.
    reference.model.visual = model.model.visual
    optimizer = pilot.make_optimizer(model, args.learning_rate)
    if args.resume:
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        del state
        print(
            f"Restored update {start} from {args.resume}. Target {args.steps}.",
            flush=True,
        )
    for step in range(start + 1, args.steps + 1):
        result = pilot.run_update(
            model, processor, optimizer, groups[step - 1], args.images, reference
        )
        result["step"] = step
        if step in (args.save_at, args.steps) or (
            args.save_every and step % args.save_every == 0
        ):
            before = perf_counter()
            save(args.output, step, model, processor, optimizer, recipe)
            result["save_seconds"] = perf_counter() - before
        with (args.output / "updates.jsonl").open("a") as f:
            print(json.dumps(result, allow_nan=False), file=f)
        print(
            f"step {step}: KL={result['mean_reference_kl']:.6g}, {result['update_seconds']:.2f} seconds",
            flush=True,
        )


if __name__ == "__main__":
    main()
